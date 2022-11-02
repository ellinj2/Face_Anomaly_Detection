class Pipeline:
	def __init__(self, models=[], **kwargs):
		self.models = models
		self.args = kwargs

	def __call__(self, X, **kwargs):
		for model in self.models:
			X = model(X, **kwargs)
		return X