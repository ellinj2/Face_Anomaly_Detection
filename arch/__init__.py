import torch
from model_params import (conv1, conv2, conv3,
						  maxpool1, maxpool2, maxpool3,
						  dropout1, dropout2,
						  lin1, lin2,
						  out_dim)

embedding_model = torch.nn.Sequential(
	conv1,
	maxpool1,
	conv2,
	maxpool2,
	conv3,
	maxpool3,
	dropout1,
	lin1,
	dropout2,
	lin2,
	)

classifying_model = torch.nn.Sequential(
	con1,
	maxpool1,
	conv2,
	maxpool2,
	conv3,
	maxpool3,
	dropout1,
	lin1,
	dropout2,
	lin2,
	torch.nn.Linear(out_dim, 1, activation="sigmoid"))