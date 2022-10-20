import os
import numpy as np
import cv2
import pickle
from transforms import wavedec2d, waverec2d
from config import *
import matplotlib.pyplot as plt
import random


def wpsnr_to_mark(wpsnr: float) -> int:
	"""Convert WPSNR to a competition mark

	Args:
		wpsnr (float): the WPSNR value in dB

	Returns:
		int: The mark that corresponds to the WPSNR value according to the competition rules
	"""
	if wpsnr >= 35 and wpsnr < 50:
		return 1
	if wpsnr >= 50 and wpsnr < 54:
		return 2
	if wpsnr >= 54 and wpsnr < 58:
		return 3
	if wpsnr >= 58 and wpsnr < 62:
		return 4
	if wpsnr >= 62 and wpsnr < 66:
		return 5
	if wpsnr >= 66:
		return 6
	return 0


def generate_watermark(size_h: int, size_v: int = 0, save: bool = False) -> np.ndarray:
	"""Generates a random watermark of size (size_h, size_v)

	Generates a random watermark of size (size_h, size_v) if size_v is specified,
	otherwise generates a square watermark of size (size_h, size_h)

	Args:
		size_h (int): Horizontal size
		size_v (int, optional): Vertical size. Defaults to size_h.

	Returns:
		np.ndarray: Random watermark of the desired size
	"""
	if size_v == 0:
		size_v = size_h
	
	# Generate a watermark
	mark = np.random.uniform(0.0, 1.0, size_v * size_h)
	mark = np.uint8(np.rint(mark))
	if save is True:
		np.save('mark.npy', mark)
	return mark.reshape((size_v, size_h))


def show_images(list_of_images: list, rows: int, columns: int, show: bool = True) -> None:
	"""Plot a list of images in a grid of size (rows, columns)

	The list of images must be a list of tuples (image, title), such as:
	[(watermarked, "Watermarked"), (attacked, "Attacked"), ...]

	Args:
		list_of_images (list): List of (image: list, title: str) tuples
		rows (int): number of rows in the grid
		columns (int): number of columns in the grid
		show (bool, optional): Whether to plt.show() the images within the function or let the user plt.show() at a different time. Defaults to True.
	"""
	for (i,(image,label)) in enumerate(list_of_images):
		plt.subplot(rows,columns,i+1)
		plt.title(list_of_images[i][1])
		plt.imshow(list_of_images[i][0], cmap='gray')

	if show is True:
		plt.show()


def embed_into_svd(img: np.ndarray, watermark: list, alpha: float) -> tuple:
	"""Embeds the watermark into the S component of the SVD decomposition of the image

	Args:
		img (np.ndarray): Image in which to embed the watermark
		watermark (list): Watermark to embed
		alpha (float): Embedding strength coefficient

	Returns:
		tuple: (Watermarked image: np.ndarray, SVD key matrices: tuple)
	"""
	(svd_u, svd_s, svd_v) = np.linalg.svd(img)

	# Convert S from a 1D vector to a 2D diagonal matrix
	svd_s = np.diag(svd_s)

	# Embed the watermark in the SVD matrix
	for x in range(0, watermark.shape[0]):
		for y in range(0, watermark.shape[1]):
			svd_s[x][y] += alpha * watermark[x][y]

	(svd_s_u, svd_s_s, svd_s_v) = np.linalg.svd(svd_s)

	# Convert S from a 1D vector to a 2D diagonal matrix
	svd_s_s = np.diag(svd_s_s)

	# Recompose matrices from SVD decomposition
	watermarked = svd_u @ svd_s_s @ svd_v
	# key = svd_s_u @ svd_s @ svd_s_v

	return (watermarked, (svd_s_u, svd_s_v))

def extract_from_svd(img, svd_key, alpha):	
	# Perform SVD decomposition of image
	svd_w_u, svd_w_s, svd_w_v = np.linalg.svd(img)

	# Convert S from a 1D vector to a 2D diagonal matrix
	svd_w_s = np.diag(svd_w_s)

	# Reconstruct S component using embedding key components
	s_ll_d = svd_key[0] @ svd_w_s @ svd_key[1]

	# Initialize the watermark matrix
	watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)
	
	# Extract the watermark
	for i in range(0,MARK_SIZE):
		for j in range(0,MARK_SIZE):
			watermark[i][j] = (s_ll_d[i][j] - svd_w_s[i][j]) / alpha

	return watermark

def save_parameters(img_name: str, alpha: float, svd_key: tuple) -> None:
	"""Saves the necessary parameters for the detection into parameters/<img_name>_parameters.txt

	Args:
		img_name (str): Name of the image
		alpha (float): Embedding strength coefficient
		svd_key (tuple): Tuple containing the SVD key matrices for the reverse algorithm
	"""
	if not os.path.isdir('parameters/'):
		os.mkdir('parameters/')
	f = open('parameters/' + img_name + '_parameters.txt', 'wb')
	pickle.dump((img_name, alpha, svd_key), f, protocol=2)
	f.close()

def read_parameters(img_name: str) -> tuple:
	"""Retrieves the necessary parameters for the detection from parameters/<img_name>_parameters.txt

	Args:
		img_name (str): Name of the image

	Returns:
		tuple: (Name of the image: str, Embedding strength coefficient: float, SVD key matrices for the reverse algorithm: np.ndarray)
	"""
	# print("IMGNAME: ", img_name)
	f = open('parameters/' + img_name + '_parameters.txt', 'rb')
	(img_name, alpha, svd_key) = pickle.load(f)
	f.close()
	return img_name, alpha, svd_key

def import_images(img_folder_path: str, num_images: int, shuffle:bool=False) -> list:
	"""Loads a list of all images contained in a folder and returns a list of (image, name) tuples
	Args:
		img_folder_path (str): Relative path to the folder containing the images (e.g. 'images/')
	Returns:
		list: List of (image, name) tuples
	"""
	if not os.path.isdir(img_folder_path):
		exit('Error: Images folder not found')
	
	images = []
	paths = os.listdir(img_folder_path)
	if shuffle:
		random.shuffle(paths)
	for img_filename in paths[:num_images]:
		# (image, name)
		images.append((cv2.imread(img_folder_path + img_filename, cv2.IMREAD_GRAYSCALE), img_filename.split('.')[-2]))


	print('Loaded', num_images, 'image' + ('s' if num_images > 1 else ''))
	
	return images

def extract_watermark(original_img: np.ndarray, img_name: str, watermarked_img: np.ndarray, level: int, subbands: list) -> np.ndarray:
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
			watermarked_band = watermarked_coeffs[0]
		elif subband == "HL":
			original_band = original_coeffs[1][0]
			watermarked_band = watermarked_coeffs[1][0]
		elif subband == "LH":
			original_band = original_coeffs[1][1]
			watermarked_band = watermarked_coeffs[1][0]
		elif subband == "HH":
			original_band = original_coeffs[1][2]
			watermarked_band = watermarked_coeffs[1][0]
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
		
		for i in range(0,MARK_SIZE):
			for j in range(0,MARK_SIZE):
				watermark[i][j] = (s_band_d[i][j] - original_band_s[i][j]) / alpha
		watermarks.append(watermark)
	
	final_watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)
	
	for watermark in watermarks:
		final_watermark += watermark
	final_watermark = final_watermark / len(subbands)

	# NOTE: Danger zone!
	for i in range(0, MARK_SIZE):
		for j in range(0, MARK_SIZE):
			if final_watermark[i][j] >= 0.5: # Threshold from paper
				final_watermark[i][j] = 1
			else:
				final_watermark[i][j] = 0

	return final_watermark


def embed_watermark(original_img: np.ndarray, img_name: str, watermark: np.ndarray, alpha: float, level, subbands: list) -> np.ndarray:
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

		band_svd, svd_key = embed_into_svd(band, watermark, alpha)
		save_parameters(img_name + '_' + subband + str(level), alpha, svd_key)

		if subband == "LL":
			coeffs[0] = band_svd
		elif subband == "HL":
			coeffs[1] = (band_svd, coeffs[1][1], coeffs[1][2])
		elif subband == "LH":
			coeffs[1] = (coeffs[1][0], band_svd, coeffs[1][2])
			band = coeffs[1][1]
		elif subband == "HH":
			coeffs[1] = (coeffs[1][0], coeffs[1][1], band_svd)
		else:
			raise Exception(f"Subband {subband} does not exist")
		
	return waverec2d(coeffs)

def make_dwt_image(img_coeffs: list) -> np.ndarray:
	"""Creates a DWT image from a given set of DWT coefficients

	Args:
		img (np.ndarray): DWT coefficients

	Returns:
		np.ndarray: DWT image
	"""
	levels = len(img_coeffs) - 1
	original_size = img_coeffs[0].shape[0] * (2 ** levels)
	img = np.zeros((original_size, original_size), dtype=np.float64)
	size = 0
	i = levels
	for level in range(1, levels+1):	
		size = int(original_size / (2 ** level))
		img[size:size*2, 0:size] = img_coeffs[i][0]
		img[0:size, size:size*2] = img_coeffs[i][1]
		img[size:size*2, size:size*2] = img_coeffs[i][2]
		i -= 1

	size = int(original_size / (2 ** levels))
	img[0:size, 0:size] = img_coeffs[0]

	return img