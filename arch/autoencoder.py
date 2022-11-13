import torch
from model_params import (conv1, conv2, conv3,
						  maxpool1, maxpool2, maxpool3,
						  dropout1, dropout2,
						  lin1, lin2,
						  out_dim,
						  decode_conv1, decode_conv2, decode_conv3,
						  decode_unflatten,
						  decode_lin1,
						  decode_lin2)

class Encoder(torch.nn.Module):
	def __init__(self, latent_dim):
		super().__init__()
		self.latent_dim = latent_dim

	def encode(self, x):
		x = conv1(x)
		x = torch.nn.ReLU(True)(x)
		x = maxpool1(x)
		x = conv2(x)
		x = torch.nn.ReLU(True)(x)
		x = maxpool2(x)
		x = conv3(x)
		x = torch.nn.ReLU(True)(x)
		x = maxpool3(x)
		x = dropout1(x)
		x = x.reshape(-1)
		x = lin1(x)
		x = torch.nn.ReLU(True)(x)
		x = dropout2(x)
		x = lin2(x)
		x = torch.nn.ReLU(True)(x)
		x = torch.nn.Linear(out_dim, self.latent_dim)(x)
		x = torch.nn.Sigmoid()(x)
		return x

	def forward(self, x):
		x = self.encode(x)
		return x

class Decoder(torch.nn.Module):
	def __init__(self, latent_dim):
		super().__init__()
		self.latent_dim = latent_dim

	def decode(self, x):
		x = decode_lin1(x)
		x = torch.nn.ReLU(True)(x)
		x = decode_lin2(x)
		x = torch.nn.ReLU(True)(x)
		x = decode_unflatten(x)
		x = decode_conv1(x)
		x = torch.nn.ReLU(True)(x)
		x = decode_conv2(x)
		x = torch.nn.ReLU(True)(x)
		x = decode_conv3(x)
		x = torch.nn.Sigmoid()(x)
		return x

	def forward(self, x):
		x = self.decode(x)
		return x