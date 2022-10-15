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

watermarked_images = []

for original_img, img_name in images[:n_images]:
	print('Loading image %s' % img_name)
	watermarked_img = embed_watermark(original_img, img_name, watermark, DEFAULT_ALPHA)

	watermarked_images.append((original_img, watermarked_img, img_name))

threshold, tpr, fpr = compute_thr_multiple_images(watermarked_images, watermark, False)


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

plt.show()

'''
# extracted_watermark = extract_watermark(original_img, img_name, watermarked_img, DWT_LEVEL)
for _ in range(0, RUNS_PER_IMAGE):
attacked_img, attacks_list = random_attacks(watermarked_img)

attacked_watermark = extract_watermark(original_img, img_name, attacked_img)


# TODO: Save results to file for analyzing best and worst attack strategies and determine performance

print('======================')
print('Image: %s' % img_name)
print('Attacks: %s' % attacks_list)
print('WPSNR original_img-watermarked_img: %.2f[dB]' % wpsnr(original_img, watermarked_img))
print('WPSNR original_img-attacked: %.2f[dB]' % wpsnr(original_img, attacked_img))
print('WPSNR watermarked_img-attacked: %.2f[dB]' % wpsnr(watermarked_img, attacked_img))

# TODO: train for similarity and various parameters

wpsnr_m = wpsnr(original_img, attacked_img)
sim = similarity(watermark, attacked_watermark)

# TODO: does it make sense to run multiple random watermarks through sim for each image?

if wpsnr_m > optimum[2] and sim < optimum[3]:
	optimum = (img_name, attacks_list, wpsnr_m, sim)
	print('======================')
	print('New Optimum: %s' % str(optimum))
	print('Image: %s' % img_name)
	print('Attacks: %s' % attacks_list)
	print('WPSNR original_img-watermarked_img: %.2f[dB]' % wpsnr(original_img, watermarked_img))
	print('WPSNR original_img-attacked: %.2f[dB]' % wpsnr_m)
	print('WPSNR watermarked_img-attacked: %.2f[dB]' % wpsnr(watermarked_img, attacked_img))
	print('SIM: %.4f' % sim)

show_images([
	(original_img, "Original image"),
	(watermarked_img,"Watermarked image"),
	(attacked_img, "Attacked"),
	(watermark, "Watermark"),
	(extracted_watermark, "Extracted watermark"),
	(attacked_watermark, "Attacked Watermark"),
	(watermarked_img - original_img, "Watermarked - Original"),
	(attacked_img - original_img, "Attacked - Original"),
	], 3, 3)


show_images([(extracted, "Extracted watermak"), (attacked, "Attacked"), (extracted_watermark, "Watermark attacked")], 1, 3)
'''