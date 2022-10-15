import sys
from time import time

sys.path.append("..")
from config import *
from attacks import *
from measurements import *
from transforms import *
from embedding import *
from tools import *

def embed_watermark(original_img, img_name, watermark, level, alpha):
	coeffs = wavedec2d(original_img, level)
	ll = coeffs[0]

	ll_svd, svd_key = embed_into_svd(ll, watermark, alpha)

	save_parameters(img_name, alpha, svd_key)

	coeffs[0] = ll_svd
	return waverec2d(coeffs)

def extract_watermark(original_img, img_name, watermarked_img, level):
	(_, alpha, svd_key) = read_parameters(img_name)

	return detection_dwt_svd(original_img, watermarked_img, alpha, level, MARK_SIZE, svd_key)

start_time = time()
print('Starting...')

# Load images
images = import_images(IMG_FOLDER_PATH)
N_IMAGES_LIMIT = len(images) # set to a lower number to limit the number of images to process
watermark = generate_watermark(1024).reshape((MARK_SIZE, MARK_SIZE)) # So that it is a square

#(Image, Attacks List, WPSNR, SIM)
optimum = (None, None, -999999, 999999)

for original_img, img_name in images[:N_IMAGES_LIMIT]:
	watermarked_img = embed_watermark(original_img, img_name, watermark, DWT_LEVEL, DEFAULT_ALPHA)

	# extracted_watermark = extract_watermark(original_img, img_name, watermarked_img, DWT_LEVEL)

	for _ in range(0, RUNS_PER_IMAGE):
		attacked_img, attacks_list = random_attacks(watermarked_img)

		attacked_watermark = extract_watermark(original_img, img_name, attacked_img, DWT_LEVEL)

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