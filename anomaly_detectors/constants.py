LOSS_MAP = {
	"contrastive flow" : lambda x,y : x - y,
	"likelihood ratio" : lambda x,y : x / y,
}
DETECTOR_MAP = {
	
}