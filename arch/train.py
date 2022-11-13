from autoencoder import Encoder, Decoder
import torch

def get_arg_parser():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--latent-dim", type=int, default=200, help="Latent space dimension")
	return parser

def main(args):

	encoder = Encoder(args.latent_dim)
	decoder = Decoder(args.latent_dim)
	x = torch.rand((3,400,400))
	print(x.shape)
	x = encoder(x)
	print(x.shape)
	x = decoder(x)
	print(x.shape)

if __name__ == "__main__":
	main(get_arg_parser().parse_args())