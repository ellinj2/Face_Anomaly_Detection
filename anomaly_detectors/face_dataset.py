import torch
import torchvision
import glob
import pandas as pd
import cv2

class RFFaceDataset(torch.utils.data.Dataset):
	"""
	Real and Fake Faces Dataset loader for network

	Attributes:
		df - the pandas dataframe that holds all of the data
			Column:	ID			Label					Difficulty					...
			Values:	Image ID	1 if real, 0 if fake	Difficulty of fake image	1 if modified
	"""
	def __init__(self, imgdir):
		"""
		Initialize the class

		Arguments:
			imgidr - path to image directory
		"""
		self.image_dir = imgdir
		self.image_ids = []
		self.labels = []
		self.difficulties = []
		self.left_eye = []
		self.right_eye = []
		self.nose = []
		self.mouth = []
		self.transform = torchvision.transforms.ToTensor()
		self.read_data()

	def read_data(self):
		real_images = glob.glob(f"{self.image_dir}/training_real/*.jpg")
		fake_images = glob.glob(f"{self.image_dir}/training_fake/*.jpg")

		for image in real_images:
			_id = image.split(self.image_dir)[-1]
			self.image_ids.append(_id)
			image = image.split('/')[-1].split('\\')[-1]
			self.labels.append(1)
			self.difficulties.append("Real")
			self.left_eye.append(0)
			self.right_eye.append(0)
			self.nose.append(0)
			self.mouth.append(0)

		for image in fake_images:
			_id = image.split(self.image_dir)[-1]
			self.image_ids.append(_id)
			self.labels.append(0)
			image = image.split('/')[-1].split('\\')[-1]
			difficulty, _, face = image.split('_')
			left_eye = right_eye = nose = mouth = 0
			try:
				left_eye = int(face[0])
			except:
				print(f"Warning: {image} does not have proper left-eye mapping")
			try:
				right_eye = int(face[1])
			except:
				print(f"Warning: {image} does not have proper right-eye mapping")
			try:
				nose = int(face[2])
			except:
				print(f"Warning: {image} does not have proper nose mapping")
			try:
				mouth = int(face[3])
			except:
				print(f"Warning: {image} does not have proper mouth mapping")
			
			self.difficulties.append(difficulty.title())
			self.left_eye.append(int(left_eye))
			self.right_eye.append(int(right_eye))
			self.nose.append(int(nose))
			self.mouth.append(mouth)

		self.df = pd.DataFrame({"ID" : self.image_ids,
								"Label" : self.labels,
								"Difficulty" : self.difficulties,
								"Left Eye" : self.left_eye,
								"Right Eye" : self.right_eye,
								"Nose" : self.nose,
								"Mouth" : self.mouth})

	def __len__(self):
		return len(self.image_ids)

	def __getitem__(self, index):
		image_id = self.image_ids[index]
		label = self.labels[index]
		image_rgb = cv2.cvtColor(cv2.imread(f"{self.image_dir}{image_id}", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
		target = [col[index] for col in [self.left_eye, self.right_eye, self.nose, self.mouth]]

		return self.transform(image_rgb), label, self.difficulties[index], target