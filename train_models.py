from time import time
from datetime import datetime

from config import *
from measurements import *
from tools import *


def main():
	# Load images
	images = import_images(IMG_FOLDER_PATH,N_IMAGES_LIMIT,True)

	# Generate watermark
	watermark = generate_watermark(MARK_SIZE)

	show_threshold = False
	attacks = []
	# Get list of attacks, so that the models are trained with images attacked in the same way
	for _ in images:
		for _ in range(0, RUNS_PER_IMAGE):
			attacks.append(get_random_attacks(randint(1, MAX_N_ATTACKS)))

	work = []
	# TODO: Avoid retraining models already trained, or implement logic to continue training already trained models with new samples
	
	alpha_range = np.arange(0.1, 0.4 , 0.1) * DEFAULT_ALPHA
	for alpha in alpha_range:
		alpha = int(alpha)
		for level in [DWT_LEVEL-1,DWT_LEVEL,DWT_LEVEL+1]:
			for subband in [["LL"], ["HL","LH"]]:
				work.append((images, embed_watermark, extract_watermark, watermark, alpha, level, subband, attacks, show_threshold))
	
	alpha_range = np.arange(0.5, 1, 0.2) * ALPHA_TN
	beta_range = np.arange(0.01, BETA+0.1, 0.04)
	for alpha in alpha_range:
		alpha = round(alpha,2)
		for beta in beta_range:
			beta = round(beta,2)
			work.append((images,embed_watermark_tn, extract_watermark_tn, watermark, alpha, beta, attacks, show_threshold))
	
	result = multiprocessed_workload(create_model,work)
	print(result)


if __name__ == '__main__':
	st = time()
	main()
	et = time()
	print(et-st)