from time import time
from datetime import datetime

from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *

start_time = time()
print('Starting...')

# Load images
images = import_images(IMG_FOLDER_PATH)
n_images = min(len(images), N_IMAGES_LIMIT) # set to a lower number to limit the number of images to process
watermark = generate_watermark(MARK_SIZE)

#(Image, Attacks List, WPSNR, SIM)
optimum = (None, None, -999999, 999999)

show_threshold = True

watermarked_images = []

for original_img, img_name in images[:n_images]:
	print('Elaborating image %s' % img_name)
	watermarked_img = embed_watermark(original_img, img_name, watermark, DEFAULT_ALPHA)

	watermarked_images.append((original_img, watermarked_img, img_name))

threshold, tpr, fpr = compute_thr_multiple_images(watermarked_images, watermark, show_threshold)


f = open('threshold.txt', 'w')
f.write(str(threshold))
f.close()

f = open('results.txt', 'a')
f.write('''Date-Time: {}
	Threshold: {}
	TPR: {}
	TARGET_FPR: {}
	FPR: {}
	Alpha: {}
	n_images: {}
	MAX_N_ATTACKS: {}
	N_AVAILABLE_ATTACKS: {}
	RUNS_PER_IMAGE: {}
	N_FALSE_WATERMARKS_GENERATIONS: {}
	==================================\n
	'''.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), threshold, tpr, TARGET_FPR, fpr, DEFAULT_ALPHA, n_images, MAX_N_ATTACKS, N_AVAILABLE_ATTACKS, RUNS_PER_IMAGE, N_FALSE_WATERMARKS_GENERATIONS))
f.close()