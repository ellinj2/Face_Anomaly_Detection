import os
import numpy as np
import cv2
import torch
import glob
import pandas as pd
#import config

from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2


class FaceDataset(Dataset):
	"""
	Dataset facillitating the loading of images for the network

	Attributes:
		dataframe
	"""
	def __init__(self, dataframe, image_dir, transforms=None):
		"""
		Initializes FaceDataset Instance

		Parameters:
			dataframe - pandas dataframe containing image metadata

		"""
		super().__init__()

		self.num_ids = dataframe.shape[0]
		self.df = dataframe
		self.image_dir = image_dir
		self.transforms = transforms

	def __getitem__(self, index: int):
		"""
		Gets data associated with desired image

		Parameters:
			index - the index of the desired image. Integer version of it's file no.

		Returns:
			image - CV2 object representation of image
			landmarks - the facial landmarks associated with that image
			index - the index of the image in the dataframe
		"""
		
		records = self.df.loc[index]

		image = cv2.imread("{0}/{1:05}/{2:05}.png".format(self.image_dir,index-(index%1000),index),cv2.IMREAD_COLOR)

		landmarks = records["in_the_wild"]["face_landmarks"]

		return image, landmarks, index


	def __len__(self):
		"""
		Gets the length of the dataset

		"""
		return self.num_ids


if __name__ == '__main__':
	# Big file - 2 minute load time
	#df = pd.read_json("../data/ffhq-dataset-v2.json").T

	# Test file - First 10 images
	df = pd.read_json("../data/test.json").T
	
	FD = FaceDataset(df,"../data/in-the-wild-images/train")
	for i in range(10):
		img, landmarks, index = FD.__getitem__(i)
		cv2.imwrite("{:05}.png".format(index),img)
		print("{}: {}".format(index, landmarks))
