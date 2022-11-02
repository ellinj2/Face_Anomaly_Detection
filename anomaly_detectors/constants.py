import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
LOSS_MAP = {
	"contrastive flow" : lambda x,y : x - y,
	"likelihood ratio" : lambda x,y : x / y,
	"log-likelihood ratio" : lambda x,y : np.log(x/y)
}
DETECTOR_MAP = {
	"one-class SVM" : OneClassSVM,
	"isolation forest" : IsolationForest,
	"local outlier factor" : LocalOutlierFactor
}