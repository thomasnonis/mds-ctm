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
	alpha_range = np.arange(0.1, 1, 0.1) * DEFAULT_ALPHA
	for alpha in alpha_range:
		alpha = int(alpha)
		for level in [DWT_LEVEL-1,DWT_LEVEL,DWT_LEVEL+1]:
			for subband in [["LL"]]:
				work.append((images, watermark, alpha, level, subband, attacks, show_threshold))
		
	result = multiprocessed_workload(create_model,work)
	print(result)


if __name__ == '__main__':
	st = time()
	main()
	et = time()
	print(et-st)