import os
import scipy.io
import numpy as np
import cv2
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
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

		image_bgr = cv2.imread("{0}/{1:05}/{2:05}.png".format(self.image_dir,index-(index%1000),index),cv2.IMREAD_COLOR)

		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

		tensor = self.transform(image_rgb)

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
	def __init__(self, filename, label_file, imgdir):
		"""
		Initializes the class

		Arguments:
			filename - the name of the boundingbox file
			label_file - the name of label file
			imgdir - the directory to search for images in
		"""
		self.datafile = filename
		self.label_file = label_file
		self.image_dir = imgdir
		# self.prog_bar_width = 50
		# self.prog_started = False
		# self.prog_started = False

		self.image_ids = []
		self.image_to_bbox_dict = {}
		self.read_data()
		self.image_ids = self.image_ids[:] # TEMPORARY, DELETE ONCE NMS ERROR IS RESOLVED
		self.transform = transforms.ToTensor()

	def __getitem__(self, index: int):
		"""
		Gets an item from the dataset at index: index

		Arguments:
			index - the index of the piece of data

		Returns:
			tensor - the tensor containing the image, scaled from 0 to 1
			bbox - a tensor of shape(N,4), where N is the number of valid bounding
							boxes
			image_id - the name of the image file
		"""
		image_id = self.image_ids[index]
		bbox = self.image_to_bbox_dict[image_id]
		# row = self.df[self.df["Name"] == image_id]
		# bbox = row.BBox.values[0]
		image_bgr = cv2.imread("{0}/{1}".format(self.image_dir,image_id),cv2.IMREAD_COLOR)
		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		tensor = self.transform(image_rgb)

		return tensor, bbox, image_id

	def __len__(self):
		"""
		Gets the length of the Dataset
		"""
		return len(self.image_ids)

	def prog_bar(self, x, tot):
		"""
		I made a progress bar because it got pretty slow at one point. Enjoy!
		"""
		perc = x/float(tot)
		progress = int(perc*self.prog_bar_width)
		if self.prog_started:
			print("["+progress*'#'+(self.prog_bar_width-progress)*"_"+"]",end="\r")
		else:
			print("[Beginning Loading File]")
			print("["+progress*'#'+(self.prog_bar_width-progress)*"_"+"]",end="\r")
			self.prog_started = True

	def read_data(self):
		"""
		Reads bbox metadata out of txt file, collects image names and bboxes
		into dataframe.
		Code inspired by:
			https://github.com/twmht/python-widerface/blob/master/wider/loader.py
		"""
		f = scipy.io.loadmat(self.label_file)
		event_list = f.get('event_list')
		file_list = f.get('file_list')
		face_bbx_list = f.get('face_bbx_list')

		for event_idx, event in enumerate(event_list):
			directory = event[0][0]
			for im_idx, im in enumerate(file_list[event_idx][0]):
				im_name = im[0][0]
				face_bbx = face_bbx_list[event_idx][0][im_idx][0]

				bboxes = []

				for i in range(face_bbx.shape[0]):
					xmin = int(face_bbx[i][0])
					ymin = int(face_bbx[i][1])
					xmax = int(face_bbx[i][2]) + xmin
					ymax = int(face_bbx[i][3]) + ymin

					# removes broken bounding boxes.
					if xmin < xmax and ymin < ymax:
						bboxes.append([xmin, ymin, xmax, ymax])

				image_path = os.path.join(directory, im_name + ".jpg")
				self.image_ids.append(image_path)
				self.image_to_bbox_dict[image_path] = torch.Tensor(bboxes).type(torch.float32) if len(bboxes) != 0 else torch.zeros((0,4), dtype=torch.float32)

		# with open(self.datafile,"r") as f:
		# 	all_data = f.read()
		# 	all_data = all_data.strip("\n").split("\n")
		# 	self.df = pd.DataFrame(columns=["Name","BBox"])
		# 	i = 0
		# 	while i < len(all_data):
		# 		img_name = all_data[i]
		# 		i+=1
		# 		num_boxes = int(all_data[i])
		# 		i+=1
		# 		boxes = np.zeros(shape=(num_boxes,4),dtype=np.float32)
		# 		for j in range(num_boxes):
		# 			box_data = all_data[i+j]
		# 			box_data = box_data.strip().split(" ")
		# 			x1, y1, w, h = [float(int(b)) for b in box_data[:4]]
		# 			if w != 0 and h != 0 and x1+w != 0 and y1+h != 0:
		# 				boxes[j][0] = x1
		# 				boxes[j][1] = y1
		# 				boxes[j][2] = x1 + w
		# 				boxes[j][3] = y1 + h
		# 		if num_boxes == 0:
		# 			i += 1
		# 		else:
		# 			i+=num_boxes

		# 		self.image_ids.append(img_name)
		# 		self.image_to_bbox_dict[img_name] = torch.from_numpy(boxes)
		# 		new_row = pd.Series({'Name':img_name, 'BBox':torch.from_numpy(boxes)})
		# 		self.df = pd.concat([self.df,new_row.to_frame().T],ignore_index=True)
		# 		self.prog_bar(i,len(all_data))
		# print()

if __name__ == '__main__':
	# Just a quick little test you can try
	test_ffhq = False
	test_wf = True

	if test_ffhq:
		# Big file - 2 minute load time
		#df = pd.read_json("../data/in-the-wild-images/ffhq-dataset-v2.json").T

		# Test file - First 10 images
		df = pd.read_json("../data/in-the-wild-images/test.json").T
		
		FD = FFHQFaceDataset(df,"../data/in-the-wild-images/train")
		for i in range(10):
			tensor, landmarks, index = FD.__getitem__(i)
			print("{}: {}: {}".format(index, tensor.shape, landmarks))
	
	if test_wf:
		FD = WFFaceDataset("../data/wider_face/wider_face_train_bbx_gt.txt","../data/wider_face/train")
		for i in range(10):
			tensor, bbox, name = FD.__getitem__(i)
			print("{}:\n{}\n{}\n".format(name,tensor.shape,bbox))