import os
import torch

from region_proposal import RegionProposalNetwork
from anomaly_classifier import AnomalyClassifier

class ADIFI:
	def __init__(self, load_path):
		rpn_path = os.path.join(load_path, "region_proposal_model")
		ad_path = os.path.join(load_path, "anomaly_detector_model")

		self.rpn_model = RegionProposalNetwork(load_path=rpn_path)
		self.cls_model = AnomalyClassifier(load_path=ad_path)

		self.to(torch.device('cpu')) # defaults to CPU.

	def to(self, device):
		"""
		Loads and performs computations on the model and input data to specified device.

		Parameters:
			device [str]: Name of the device to load model too.
		"""
		try:
			self.rpn_model.to(device)
			self.cls_model.to(device)
			self.device = device
		except Exception as e:
			raise Exception(e)

	def preprocess(self, X):
		return self.rpn_model.preprocess(X)

	def detect(self, X):
		y_props = self.rpn_model.propose(X)

		y_hats = []
		for x, y_prop in zip(X, y_props):
			bbox = y_prop["boxes"]
			x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
				
			X_propose = x[y1:y2,x1:x2]

			X_pprocess = self.cls_model.preprocess(X_propose)
			y_cls = self.cls_model.classify(X_pprocess)

			y_prop["label"] = y_cls

			y_hats.append(y_prop)
			
		return y_hats
	
	def evaluate(self):
		pass