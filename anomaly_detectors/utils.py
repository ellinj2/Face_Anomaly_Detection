import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def getDistribution(data):
	return -1 * np.log(data)

def histogramify(real_data, fake_data, title, img_name):
	total_dist = np.concatenate((np.array(real_data), np.array(fake_data)))
	min_dist = np.min(total_dist)
	max_dist = np.max(total_dist)
	min_x = min_dist - 0.25 * (max_dist - min_dist)
	max_x = max_dist + 0.25 * (max_dist - min_dist)
	sns.histplot(data=real_data,color='g',stat="density",label='Real',kde=False)
	sns.histplot(data=fake_data,color='r',stat="density",label='Fake',kde=False)
	plt.xlim(min_x,max_x)
	plt.legend()
	plt.xlabel('Log-likelihood')
	plt.title(title)
	plt.savefig(f'{img_name}.png')
	plt.clf()