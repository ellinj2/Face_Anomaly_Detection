class AnomalyDetectorModel:
	"""
	Model for determining anomaly likelihood from input data

	Attributes:
		detector [various] : detection model to be used

	Methods:
		__call__ : passes input data into self.detector for inferencing
		fit : passes input data into self.detector for training
	"""
	def __init__(self, detector, **kwargs):
		"""
		Initializes AnomalyDetectorModel instance

		Parameters:
			detector [str, class exposing __call__ and fit] : class or string representation for detector model to be mapped to constants.DETECTOR_MAP
			kwargs : dictionary of arguments to be passed to API functions
		"""
		self.detector = detector(**kwargs) if type(detector) is not str else DETECTOR_MAP[detector](**kwargs)

	def __call__(self, X, **kwargs):
		"""
		Pass input data through detector

		Parameters:
			X [pytorch.tensor or similar] : Input data to be processed
			kwargs : keyword arguments to be passed to self.detector

		Returns:
			pytorch.tensor or similar : Return from self.detector
		"""

	def fit(self, X, **kwargs):
		"""
		Fit self.detector on input data

		Parameters:
			X [pytorch.tensor or similar] : Input data to be processed
			kwargs : keyword arguments to be passed to self.detector training (could include Y data)
		"""

class AnomalyDetectorLoss:
	"""
	Loss functions for anomaly detection in-situ for the embedding model

	Attributes:
		function [callable] : Function used to compute loss
		args [dict] : Specific keyword arguments for loss computation

	Methods:
		__init__ : Initializes attributes
		__call__ : Calls loss function
	"""
	def __init__(self, f, **kwargs):
		"""
		Initializes loss function

		Parameters:
			f [str, callable] : Loss function to be called. If type is str, will pull function from constants.LOSS_MAP
			kwargs : keyword arguments to be passed to self.__call__
		"""
		self.f = f if type(f) is not str else LOSS_MAP[f]
		self.args = **kwargs

	def __call__(self, **kwargs):
		"""
		Passes values through self.f to compute loss

		Parameters:
			kwargs : keyword arguments to be passed to self.f
		"""
		return self.f(self.args | **kwargs)