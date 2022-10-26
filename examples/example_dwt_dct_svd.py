import sys
from time import time
from collections import defaultdict

import numpy as np

sys.path.append('..')
from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *

start_time = time()
print('Starting...')

# Load images
n_images = min(1,N_IMAGES_LIMIT)
images = import_images('../'+IMG_FOLDER_PATH, n_images, True)
watermark = generate_watermark(MARK_SIZE)

def embed_watermark_dct(original_img: np.ndarray, img_name: str, watermark: np.ndarray, alpha: float, level,
					subbands: list) -> np.ndarray:
	"""Embeds a watermark into the S component of the SVD decomposition of an image's LL DWT subband

	Args:
		original_img (np.ndarray): Image in which to embed the watermark
		img_name (str): Name of the image
		watermark (np.ndarray): Watermark to embed
		alpha (float): Watermark embedding strength coefficient
		subbands (list): List of subbands where to embed the watermark

	Returns:
		np.ndarray: Watermarked image
	"""
	coeffs = wavedec2d(original_img, level)

	for subband in subbands:
		band = None
		if subband == "LL":
			band = coeffs[0]
		elif subband == "HL":
			band = coeffs[1][0]
		elif subband == "LH":
			band = coeffs[1][1]
		elif subband == "HH":
			band = coeffs[1][2]
		else:
			raise Exception(f"Subband {subband} does not exist")

		# print(band)

		band = dct(dct(band, axis=0, norm='ortho'), axis=1, norm='ortho')

		band_svd, svd_key = embed_into_svd(band, watermark, alpha)
		save_parameters(img_name + '_' + subband + str(level), alpha, svd_key)

		band_svd = idct(idct(band_svd, axis=1, norm='ortho'),axis=0, norm='ortho')

		if subband == "LL":
			coeffs[0] = band_svd
		elif subband == "HL":
			coeffs[1] = (band_svd, coeffs[1][1], coeffs[1][2])
		elif subband == "LH":
			coeffs[1] = (coeffs[1][0], band_svd, coeffs[1][2])
		elif subband == "HH":
			coeffs[1] = (coeffs[1][0], coeffs[1][1], band_svd)
		else:
			raise Exception(f"Subband {subband} does not exist")

	watermark = waverec2d(coeffs)
	print(wpsnr(watermark, original_img))
	plt.figure()
	plt.subplot(121)
	plt.title("Original")
	plt.imshow(original_img, cmap='gray')
	plt.subplot(122)
	plt.title("Watermarked")
	plt.imshow(watermark, cmap='gray')
	plt.show()
	return waverec2d(coeffs)


def extract_watermark(original_img: np.ndarray, img_name: str, watermarked_img: np.ndarray, level: int,
					  subbands: list) -> np.ndarray:
	"""Extracts the watermark from a watermarked image by appling the reversed embedding algorithm,
	provided that the proper configuration file and the original, unwatermarked, image are available.

	Args:
		original_img (np.ndarray): Original unwatermarked image
		img_name (str): Name of the image
		watermarked_img (np.ndarray): Image from which to extract the watermark
		subbands (list): List of subbands where to extract the watermark

	Returns:
		np.ndarray: Extracted watermark
	"""

	original_coeffs = wavedec2d(original_img, level)
	watermarked_coeffs = wavedec2d(watermarked_img, level)
	watermarks = []
	for subband in subbands:
		original_band = None
		watermarked_band = None
		if subband == "LL":
			original_band = original_coeffs[0]
			watermarked_band = watermarked_coeffs[0] # dct(dct(watermarked_coeffs[0], axis=0, norm='ortho'), axis=1, norm='ortho')
		elif subband == "HL":
			original_band = original_coeffs[1][0]
			watermarked_band = watermarked_coeffs[1][0] # dct(dct(watermarked_coeffs[1][0], axis=0, norm='ortho'), axis=1, norm='ortho')
		elif subband == "LH":
			original_band = original_coeffs[1][1]
			watermarked_band = watermarked_coeffs[1][1]# dct(dct(watermarked_coeffs[1][1], axis=0, norm='ortho'), axis=1, norm='ortho')
		elif subband == "HH":
			original_band = original_coeffs[1][2]
			watermarked_band = watermarked_coeffs[1][2] # dct(dct(watermarked_coeffs[1][2], axis=0, norm='ortho'), axis=1, norm='ortho')
		else:
			raise Exception(f"Subband {subband} does not exist")

		original_band_u, original_band_s, original_band_v = np.linalg.svd(original_band)
		original_band_s = np.diag(original_band_s)

		watermarked_band_u, watermarked_band_s, watermarked_band_v = np.linalg.svd(watermarked_band)
		watermarked_band_s = np.diag(watermarked_band_s)

		(_, alpha, svd_key) = read_parameters(img_name + '_' + subband + str(level))
		# original_s_ll_d_u, original_s_ll_d_s, original_s_ll_d_v
		s_band_d = svd_key[0] @ watermarked_band_s @ svd_key[1]

		# Initialize the watermark matrix
		watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)

		# Extract the watermark

		for i in range(0, MARK_SIZE):
			for j in range(0, MARK_SIZE):
				watermark[i][j] = (s_band_d[i][j] - original_band_s[i][j]) / alpha
		watermarks.append(watermark)

	final_watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)

	for watermark in watermarks:
		final_watermark += watermark
	final_watermark = final_watermark / len(subbands)

	plt.figure()
	plt.subplot(121)
	plt.title("Final Watermark")
	plt.imshow(final_watermark, cmap='gray')
	plt.show()
	return final_watermark

subbands = [['HL','LH']]
levels = [DWT_LEVEL]
alpha_range = np.arange(0.1, 0.3, 0.1) * DEFAULT_ALPHA
watermarked_imgs = defaultdict(dict)
print("Total work: ", n_images * len(subbands) * len(alpha_range) * len(levels))
for original_img, img_name in images:
	watermarked_imgs[img_name]['original_img'] = original_img
	watermarked_imgs[img_name]['watermarked'] = {}
	for level in levels:
		watermarked_imgs[img_name]['watermarked'][level] = {}
		for subband in subbands:
			watermarked_imgs[img_name]['watermarked'][level]['-'.join(subband)] = {}
			d = {}
			for alpha in alpha_range:
				alpha = int(alpha)
				watermarked_img = embed_watermark_dct(original_img, img_name, watermark, alpha, level, subband)
				d[alpha] = watermarked_img
			watermarked_imgs[img_name]['watermarked'][level]['-'.join(subband)] = d
plt.figure()
for img in watermarked_imgs:
	original_img = watermarked_imgs[img]['original_img']

	for level in watermarked_imgs[img]['watermarked']:
		for subband in watermarked_imgs[img]['watermarked'][level]:
			ys = []
			for alpha in watermarked_imgs[img]['watermarked'][level][subband]:
				watermarked_image = watermarked_imgs[img]['watermarked'][level][subband][alpha]
				ys.append(wpsnr(original_img, watermarked_image))
			plt.plot(alpha_range, ys, lw=2, label=img+'_'+subband+'_'+str(level))

plt.xlabel('Alpha')
plt.ylabel('WPSNR')
plt.title('WPSNR comparison of images with a watermark embedded and with different values of alpha and different subbands')
plt.legend(loc='upper right')
plt.show()

plt.figure()
attacks_list = get_random_attacks(1)

print(describe_attacks(attacks_list))
for img in watermarked_imgs:
	original_img = watermarked_imgs[img]['original_img']

	for level in watermarked_imgs[img]['watermarked']:
		for subband in subbands:
			xs = []
			ys = []
			for alpha in watermarked_imgs[img]['watermarked'][level]['-'.join(subband)]:
				watermarked_image = watermarked_imgs[img]['watermarked'][level]['-'.join(subband)][alpha]
				attacked_img, _ = do_attacks(watermarked_image, attacks_list)
				extracted_watermark = extract_watermark(original_img, img_name, attacked_img,level,subband)
				xs.append(wpsnr(original_img,attacked_img))
				ys.append(similarity(watermark, extracted_watermark))
			plt.plot(xs, ys, lw=2, label=img+'_'+'-'.join(subband)+'_'+str(level))

plt.xlabel('WPSNR')
plt.ylabel('Similarity')
plt.title('WPSNR-Similarity of attacked images at different levels ')
plt.legend(loc='upper right')
plt.show()

"""
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

t = compute_thr_multiple_images(w_images, watermark, '../images/', True)
print("Threshold: ", t)
"""