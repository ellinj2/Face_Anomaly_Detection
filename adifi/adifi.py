import os
import torch

from region_proposal import RegionProposalNetwork
from anomaly_classifier import AnomalyClassifier

class ADIFI:
	def __init__(self, load_path, epsilon=0.5):
		rpn_path = os.path.join(load_path, "region_proposal_model")
		ad_path = os.path.join(load_path, "anomaly_detector_model.pth")

		self.rpn_model = RegionProposalNetwork(load_path=rpn_path)
		self.cls_model = AnomalyClassifier(load_path=ad_path)
		self.epsilon = epsilon

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
			x1, y1, x2, y2 = bbox[:, 0].to(torch.int64).tolist(), bbox[:, 1].to(torch.int64).tolist(), bbox[:, 2].to(torch.int64).tolist(), bbox[:, 3].to(torch.int64).tolist()
	
			X_propose = [x[:,y1[i]:y2[i],x1[i]:x2[i]] for i in range(len(x1))]
			if len(X_propose) == 0:
				continue

			X_pprocess = torch.cat([self.cls_model.preprocess(X_p[None,:,:,:]) for X_p in X_propose])
			
			y_cls = self.cls_model.classify(X_pprocess)
			lab = torch.argmax(y_cls, axis=1)
			y_cls = torch.Tensor([y_cls[i,lab[i]] if lab[i]==0 else 1-y_cls[i,lab[i]] for i in range(len(lab))])

			y_prop["scores"] = y_cls
			y_prop["labels"] = (-torch.log(y_cls) >= self.epsilon).type(torch.int64)

			y_hats.append(y_prop)
			
		return y_hats
	
	def evaluate(self):
		pass