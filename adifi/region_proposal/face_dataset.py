import os
from glob import glob
from PIL import Image
import scipy.io

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FFHQFaceDataset(Dataset):
	"""
	FFHQ Dataset facillitating the loading of images for the network

	Attributes:
		dataframe - the pandas dataframe that stores all of the metadata

	"""
	def __init__(self, dataframe, image_dir):
		"""
		Initializes FaceDataset Instance

		Parameters:
			dataframe - pandas dataframe containing image metadata

		"""
		super().__init__()

		self.num_ids = dataframe.shape[0]
		self.df = dataframe
		self.image_dir = image_dir
		self.transform = transforms.ToTensor()

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

		image = Image.open("{0}/{1:05}/{2:05}.png".format(self.image_dir,index-(index%1000),index))
		tensor = self.transform(image)

		landmarks = records["in_the_wild"]["face_landmarks"]

		return tensor, landmarks, index


	def __len__(self):
		"""
		Gets the length of the dataset
		"""
		return self.num_ids


class WFFaceDataset(Dataset):
	"""
	Wider Face Dataset loader for network

	Attributes:
		df - the pandas dataframe that holds all of the data

	"""
	def __init__(self, data_path):
		"""
		Initializes WIDER FACE dataset class.

		Arguments:
			data_path - Path to image directory with a .mat annotation file.

		Code inspired by:
			https://github.com/twmht/python-widerface/blob/master/wider/loader.py
		"""
		annotation_mat = glob(os.path.join(data_path, "*.mat"))
		if len(annotation_mat) != 1:
			raise Exception(f"{data_path} should contain one mat file.")

		annotation_mat = annotation_mat[0]
		
		self.image_paths = []
		self.image_bboxs = []
		self.transform = transforms.ToTensor()

		f = scipy.io.loadmat(annotation_mat)
		event_list = f.get('event_list')
		file_list = f.get('file_list')
		face_bbox_list = f.get('face_bbx_list')

		for event_idx, event in enumerate(event_list):
			directory = event[0][0]
			for image_idx, image in enumerate(file_list[event_idx][0]):
				image_name = image[0][0]
				face_bbox = face_bbox_list[event_idx][0][image_idx][0]

				bboxes = []
				for i in range(face_bbox.shape[0]):
					xmin = int(face_bbox[i][0])
					ymin = int(face_bbox[i][1])
					xmax = int(face_bbox[i][2]) + xmin
					ymax = int(face_bbox[i][3]) + ymin

					# removes broken bounding boxes.
					if xmin < xmax and ymin < ymax:
						bboxes.append([xmin, ymin, xmax, ymax])

				image_path = os.path.join(data_path, directory, image_name + ".jpg")
				if not os.path.isfile(image_path):
					raise Exception(f"{image_path} is missing from dataset.")

				bboxes = torch.Tensor(bboxes).type(torch.float32).reshape(len(bboxes), 4)

				self.image_paths.append(image_path)
				self.image_bboxs.append(bboxes)

	def __getitem__(self, index: int):
		"""
		Gets an item from the dataset at index: index

		Arguments:
			index - the index of the piece of data

		Returns:
			image - the tensor containing the image, scaled from 0 to 1
			bbox - a tensor of shape(N,4), where N is the number of valid bounding
							boxes
		"""
		image_path = self.image_paths[index]
		image_bboxs = self.image_bboxs[index]

		image = Image.open(image_path)
		image = self.transform(image)

		return image, image_bboxs

	def __len__(self):
		"""
		Gets the length of the Dataset
		"""
		return len(self.image_paths)
