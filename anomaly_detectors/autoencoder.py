import torch
import numpy as np

class GaussianNoise(torch.nn.Module):
	def __init__(self, stddev):
		super().__init__()
		self.stddev = stddev

	def forward(self, din):
		return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)

# TO DO : Store layers locally for training
class AutoEncoder(torch.nn.Module):
	def __init__(self, latent_dim, conv_layers, lin_layers, conv_out_size=4800, conv_in_channel=32, conv_out_channel=32, channel_growth=-1, image_shape=(1,3,100,100), verbose=False):
		super().__init__()
		self.noise = GaussianNoise(1)
		self.train = True

		if conv_layers:
			conv_sizes = []
			conv_x_sizes = []
			input_shape = image_shape
			if channel_growth == -1:
				channel_growth = (conv_out_channel/conv_in_channel) ** (1/(conv_layers-1))

			# ENCODER
			x = torch.rand(input_shape)
			channel = conv_in_channel
			encoder_layers = []
			if verbose:
				print("Encoder Shapes")
				print('\t',x.shape)
			in_channels = x.shape[1]
			for layer in range(conv_layers-1):
				conv_x_sizes.append(x.shape)
				encoder_layers.append(torch.nn.Conv2d(in_channels, channel, kernel_size=3, stride=2))
				conv_sizes.append((in_channels, channel, 3, 2))
				channel = max(int(channel*channel_growth), 1)
				x = encoder_layers[-1](x)
				encoder_layers.append(torch.nn.LeakyReLU(0.3))
				x = encoder_layers[-1](x)
				if verbose:
					print('\t',x.shape)
				in_channels = x.shape[1]
			conv_x_sizes.append(x.shape)
			in_size = x.shape[2]
			square_size = conv_out_size / 3
			dim_size = int(np.sqrt(square_size))
			encoder_layers.append(torch.nn.Conv2d(in_channels, conv_out_channel, kernel_size=3, stride=max(in_size//dim_size, 1)))
			conv_sizes.append((in_channels, conv_out_channel, 3, max(in_size//dim_size, 1)))
			x = encoder_layers[-1](x)
			conv_x_sizes.append(x.shape)
			encoder_layers.append(torch.nn.LeakyReLU(0.3))
			x = encoder_layers[-1](x)
			conv_out_x_size = x.shape
			if verbose:
				print('\t',x.shape)
		if lin_layers:
			encoder_layers.append(torch.nn.AvgPool2d(x.shape[-1])) # Flatten through AvgPool
			x = encoder_layers[-1](x)
			if verbose:
				print('\t',x.shape)
			encoder_layers.append(torch.nn.Flatten())
			x = encoder_layers[-1](x)
			lin_input_shape = x.shape[1]
			if verbose:
				print('\t',x.shape)
			input_shape = x.shape[1]
			shape_diff = input_shape - latent_dim
			shape_ratio = shape_diff // lin_layers
			for layer in range(lin_layers-1):
				encoder_layers.append(torch.nn.Linear(input_shape, input_shape - shape_ratio))
				x = encoder_layers[-1](x)
				encoder_layers.append(torch.nn.LeakyReLU(0.1))
				x = encoder_layers[-1](x)
				if verbose:
					print('\t',x.shape)
				input_shape = input_shape - shape_ratio

			encoder_layers.append(torch.nn.Linear(input_shape, latent_dim))
			x = encoder_layers[-1](x)
			encoder_layers.append(torch.nn.LeakyReLU(0.1))
			x = encoder_layers[-1](x)
			if verbose:
				print('\t',x.shape)

		self.encoder = torch.nn.Sequential(*encoder_layers)

		conv_sizes.reverse()
		conv_x_sizes.reverse()

		# DECODER
		decoder_layers = []
		if verbose:
			print("Decoder Shapes")
			print('\t',x.shape)

		if lin_layers:
			in_channels = x.shape[1]
			shape_diff = lin_input_shape - in_channels
			shape_ratio = shape_diff // lin_layers
			for layer in range(lin_layers-1):
				decoder_layers.append(torch.nn.Linear(in_channels, in_channels+shape_ratio))
				x = decoder_layers[-1](x)
				decoder_layers.append(torch.nn.LeakyReLU(0.1))
				x = decoder_layers[-1](x)
				if verbose:
					print('\t',x.shape)
				in_channels = in_channels + shape_ratio

			decoder_layers.append(torch.nn.Linear(in_channels, lin_input_shape))
			x = decoder_layers[-1](x)
			decoder_layers.append(torch.nn.LeakyReLU(0.1))
			x = decoder_layers[-1](x)
			if verbose:
				print('\t',x.shape)

		if conv_layers:
			for layer in range(conv_layers-1):
				target = conv_sizes[layer]
				target_x = conv_x_sizes[layer]
				op = max(target_x[-1] - ((x.shape[-1] - 1) * target[3] + (target[2]-1) + 1), 0)
				decoder_layers.append(torch.nn.ConvTranspose2d(x.shape[1], target[0], kernel_size=target[2], stride=target[3], output_padding=op))
				x = decoder_layers[-1](x)
				decoder_layers.append(torch.nn.BatchNorm2d(x.shape[1]))
				x = decoder_layers[-1](x)
				decoder_layers.append(torch.nn.LeakyReLU(0.3))
				x = decoder_layers[-1](x)
				if verbose:
					print('\t',x.shape)

			target = conv_sizes[-1]
			target_x = image_shape
			op = target_x[-1] - ((x.shape[-1] - 1) * target[3] + (target[2]-1) + 1)
			decoder_layers.append(torch.nn.ConvTranspose2d(target[1], target[0], kernel_size=target[2], stride=target[3], output_padding=op))
			x = decoder_layers[-1](x)
			decoder_layers.append(torch.nn.BatchNorm2d(x.shape[1]))
			x = decoder_layers[-1](x)
			decoder_layers.append(torch.nn.Sigmoid())
			x = decoder_layers[-1](x)
			if verbose:
				print('\t',x.shape)

		self.decoder = torch.nn.Sequential(*decoder_layers)

	def to(self, device):
		self.encoder.to(device)
		self.decoder.to(device)

	def eval(self):
		self.train = False

	def forward(self, x):
		x = self.encoder(x)
		if self.train:
			x = self.noise(x)
		return self.decoder(x)


class Encoder(torch.nn.Module):
	def __init__(self, latent_dim, conv_layers, lin_layers, conv_out_size=4800, conv_out_channels=3, input_shape=(1,3,100,100), verbose=False):
		super().__init__()
		self.latent_dim = latent_dim
		self.conv_layers = conv_layers
		self.lin_layers = lin_layers
		self.conv_out_channels = conv_out_channels
		self.conv_out_size = conv_out_size
		self.verbose = verbose
		self.layers = []
		self.input_shape = input_shape
		self.lin_in_shape = None
		self.last_conv_size = None
		self.model = self.build_model()
		self._model_metadata = dict()

	def build_model(self):
		if self.verbose:
			print("Encoder shapes")
		x = torch.rand(self.input_shape)
		if self.verbose:
			print(x.shape)
		in_channels = x.shape[1]
		for layer in range(self.conv_layers-1):
			self.layers.append(torch.nn.Conv2d(in_channels, 32, kernel_size=3))
			x = self.layers[-1](x)
			self.layers.append(torch.nn.LeakyReLU(0.3))
			x = self.layers[-1](x)
			if self.verbose:
				print(x.shape)
			in_channels = x.shape[1]
		in_size = x.shape[2]
		square_size = self.conv_out_size / 3
		dim_size = int(np.sqrt(square_size))
		self.layers.append(torch.nn.Conv2d(in_channels, self.conv_out_channels, kernel_size=3, stride=in_size//dim_size))
		self.last_conv_size = (in_channels, self.conv_out_channels, 3, in_size//dim_size)
		x = self.layers[-1](x)
		self.layers.append(torch.nn.LeakyReLU(0.3))
		x = self.layers[-1](x)
		if self.verbose:
			print(x.shape)
		self.layers.append(torch.nn.Flatten())
		x = self.layers[-1](x)
		self.lin_in_shape = x.shape
		if self.verbose:
			print(x.shape)
		input_shape = x.shape[1]
		shape_diff = input_shape - self.latent_dim
		shape_ratio = shape_diff // self.lin_layers
		for layer in range(self.lin_layers-1):
			self.layers.append(torch.nn.Linear(input_shape, input_shape  - shape_ratio))
			x = self.layers[-1](x)
			self.layers.append(torch.nn.LeakyReLU(0.1))
			x = self.layers[-1](x)
			if self.verbose:
				print(x.shape)
			input_shape = input_shape - shape_ratio
			
		self.layers.append(torch.nn.Linear(input_shape, self.latent_dim))
		x = self.layers[-1](x)
		if self.verbose:
			print(x.shape)
		self.layers.append(torch.nn.LeakyReLU(0.1))

		return torch.nn.Sequential(*self.layers)

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = layer(x)
		return x
		# return self.model(x)

	def to(self, device):
		self.model.to(device)
		for layer in self.layers:
			layer.to(device)

	def save(self, save_path):
		"""
		Save the model to specified path.

		Parameters:
		save_path [str]: Path to directory to save model files too.
		Returns:
		[str]: Path to the folder the model files were saved too.
		"""     
		save_name = save_path.split(os.path.sep)[-1]

		if not os.path.isdir(save_path):
			os.makedirs(save_path)

		# save model weights.
		model_weights = self.model.state_dict()
		torch.save(model_weights, os.path.join(save_path, f"{save_name}_weights.pth"))

		# save model metadata.
		self._model_metadata["weight_hash"] = sha256(str(model_weights).encode()).hexdigest()
		with open(os.path.join(save_path, f"{save_name}_metadata.json"), "w") as f:
			json.dump(self._model_metadata, f, indent=1)

		return save_path

class Decoder(torch.nn.Module):
	def __init__(self, latent_dim, conv_layers, lin_layers, conv_in_size=4800, lin_out_dim=6627, first_conv_size=None, image_shape=(3,100,100), verbose=False):
		super().__init__()
		self.latent_dim = latent_dim
		self.conv_layers = conv_layers
		self.lin_layers = lin_layers
		self.lin_out_dim = lin_out_dim
		self.verbose = verbose
		self.layers = []
		self.conv_in_size = conv_in_size
		self.first_conv_size = first_conv_size
		self.image_shape = image_shape
		self.model = self.build_model()
		self._model_metadata = dict()

	def build_model(self):
		if self.verbose:
			print("Decoder shapes")
		x = torch.rand(1, self.latent_dim)
		if self.verbose:
			print(x.shape)
		in_channels = x.shape[1]
		shape_diff = self.lin_out_dim - in_channels
		shape_ratio = shape_diff // self.lin_layers
		for layer in range(self.lin_layers-1):
			self.layers.append(torch.nn.Linear(in_channels, in_channels + shape_ratio))
			x = self.layers[-1](x)
			self.layers.append(torch.nn.LeakyReLU(0.1))
			x = self.layers[-1](x)
			if self.verbose:
				print(x.shape)
			in_channels = in_channels + shape_ratio

		self.layers.append(torch.nn.Linear(in_channels, self.lin_out_dim))
		x = self.layers[-1](x)
		self.layers.append(torch.nn.LeakyReLU(0.1))
		x = self.layers[-1](x)
		if self.verbose:
			print(x.shape)
		
		in_channels = 3
		in_dim = x.shape[1] // 3
		in_dim = int(np.sqrt(in_dim))
		self.layers.append(torch.nn.Unflatten(1, (in_channels, in_dim, in_dim)))
		x = self.layers[-1](x)
		if self.verbose:
			print(x.shape)

		in_channels = x.shape[1]
		for layer in range(self.conv_layers-1):
			if layer == 0:
				if not self.first_conv_size:
					self.layers.append(torch.nn.ConvTranspose2d(in_channels, 32, kernel_size=3, stride=2, output_padding=1))
				else:
					self.layers.append(torch.nn.ConvTranspose2d(self.first_conv_size[1], self.first_conv_size[0], kernel_size=self.first_conv_size[2], stride=self.first_conv_size[3], output_padding=1 if self.image_shape[-1]%2==0 else 0))
			else:
				self.layers.append(torch.nn.ConvTranspose2d(in_channels, 32, kernel_size=3))
			x = self.layers[-1](x)
			self.layers.append(torch.nn.LeakyReLU(0.3))
			x = self.layers[-1](x)
			if self.verbose:
				print(x.shape)
			in_channels = x.shape[1]

		self.layers.append(torch.nn.ConvTranspose2d(in_channels, 3, kernel_size=3))
		x = self.layers[-1](x)
		self.layers.append(torch.nn.LeakyReLU(0.3))
		x = self.layers[-1](x)
		if self.verbose:
			print(x.shape)
		
		return torch.nn.Sequential(*self.layers)

	def forward(self, x):
		return self.model(x)

	def to(self, device):
		self.model.to(device)
		for layer in self.layers:
			layer.to(device)

	def save(self, save_path):
		"""
		Save the model to specified path.

		Parameters:
		    save_path [str]: Path to directory to save model files too.
		Returns:
		    [str]: Path to the folder the model files were saved too.
		"""     
		save_name = save_path.split(os.path.sep)[-1]

		if not os.path.isdir(save_path):
			os.makedirs(save_path)

		# save model weights.
		model_weights = self.model.state_dict()
		torch.save(model_weights, os.path.join(save_path, f"{save_name}_weights.pth"))

		# save model metadata.
		self._model_metadata["weight_hash"] = sha256(str(model_weights).encode()).hexdigest()
		with open(os.path.join(save_path, f"{save_name}_metadata.json"), "w") as f:
			json.dump(self._model_metadata, f, indent=1)

		return save_path