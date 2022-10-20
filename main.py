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
images = import_images(IMG_FOLDER_PATH,N_IMAGES_LIMIT,True)

watermark = generate_watermark(MARK_SIZE)

#(Image, Attacks List, WPSNR, SIM)
optimum = (None, None, -999999, 999999)

show_threshold = True

watermarked_images = []

alpha = DEFAULT_ALPHA
level = DWT_LEVEL
subband = DEFAULT_SUBBAND
# Take ten random images
for original_img, img_name in images:

	watermarked_img = embed_watermark(original_img, img_name, watermark, alpha, level, subband)

	watermarked_images.append((original_img, watermarked_img, img_name))

(threshold, tpr, fpr) = compute_thr_multiple_images(watermarked_images, watermark, level, subband, show_threshold)


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
	'''.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), threshold, tpr, TARGET_FPR, fpr, DEFAULT_ALPHA, len(images), MAX_N_ATTACKS, N_AVAILABLE_ATTACKS, RUNS_PER_IMAGE, N_FALSE_WATERMARKS_GENERATIONS))
f.close()