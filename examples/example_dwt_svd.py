import sys
from time import time
from collections import defaultdict

from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *

start_time = time()
print('Starting...')

# Load images
images = import_images("../images/")
n_images = min(len(images), N_IMAGES_LIMIT) # set to a lower number to limit the number of images to process
watermark = generate_watermark(MARK_SIZE)

#(Image, Attacks List, WPSNR, SIM)
# optimum = (None, None, -999999, 999999)
results = defaultdict(dict)

for original_img, img_name in images[:N_IMAGES_LIMIT]:
	watermarked_img = embed_watermark_lh_hl(original_img, img_name, watermark, DEFAULT_ALPHA)#embed_watermark(original_img, img_name, watermark, DEFAULT_ALPHA) # We should try with different alpha too!
	results[img_name] = {
		#"watermarked_img" : watermarked_img,
		"WPSNR": -999999,
		"SIM": 99999
	}

	# extracted_watermark = extract_watermark(original_img, img_name, watermarked_img, DWT_LEVEL)

	for _ in range(0, RUNS_PER_IMAGE):
		attacked_img, attacks_list = do_random_attacks(watermarked_img, get_random_attacks(1))

		attacked_watermark = extract_watermark_lh_hl(original_img, img_name, attacked_img)

		# TODO: Save results to file for analyzing best and worst attack strategies and determine performance
		'''
		print('======================')
		print('Image: %s' % img_name)
		print('Attacks: %s' % attacks_list)
		print('WPSNR original_img-watermarked_img: %.2f[dB]' % wpsnr(original_img, watermarked_img))
		print('WPSNR original_img-attacked: %.2f[dB]' % wpsnr(original_img, attacked_img))
		print('WPSNR watermarked_img-attacked: %.2f[dB]' % wpsnr(watermarked_img, attacked_img))
		'''

		# TODO: train for similarity and various parameters
		wpsnr_m = wpsnr(original_img, attacked_img)
		sim = similarity(watermark, attacked_watermark)

		# TODO: does it make sense to run multiple random watermarks through sim for each image?

		if wpsnr_m >= 35 and wpsnr_m > results[img_name]["WPSNR"] and sim < results[img_name]["SIM"]:
			#optimum = (img_name, attacks_list, wpsnr_m, sim)
			results[img_name]["Attacks"] = attacks_list
			results[img_name]["WPSNR"] = wpsnr_m
			results[img_name]["SIM"] = sim
			print('======================')
			print('New Optimum for image %s %s' % (img_name,str(results[img_name])))
			print('Image: %s' % img_name)
			print('Attacks: %s' % attacks_list)
			print('WPSNR original_img-watermarked_img: %.2f[dB]' % wpsnr(original_img, watermarked_img))
			print('WPSNR original_img-attacked: %.2f[dB]' % wpsnr_m)
			print('WPSNR watermarked_img-attacked: %.2f[dB]' % wpsnr(watermarked_img, attacked_img))
			print('SIM: %.4f' % sim)

		'''
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

print(results)
# Compute threshold
img_folder_path = '../sample-images/'
images = import_images(img_folder_path)
paths = os.listdir(img_folder_path)

w_images = []
for i in range(len(images)):
	#print(images[i][0])
	w_img = embed_watermark(images[i][0], paths[i], watermark, DEFAULT_ALPHA)
	w_images.append(images[i])

t = compute_thr_multiple_images(w_images, watermark, True)
print("Threshold: ", t)