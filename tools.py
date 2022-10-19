import os
import numpy as np
import cv2
import pickle
from transforms import wavedec2d, waverec2d
from config import MARK_SIZE, DWT_LEVEL
import matplotlib.pyplot as plt

from config import *

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
	f = open('parameters/' + img_name + '_parameters.txt', 'rb')
	(img_name, alpha, svd_key) = pickle.load(f)
	f.close()
	return img_name, alpha, svd_key

def import_image(img_path: str) -> list:
	images = []
	images.append((cv2.imread(IMG_FOLDER_PATH + img_path, cv2.IMREAD_GRAYSCALE), img_path.split('.')[-2]))
	return images

def import_images(img_folder_path: str) -> list:
	"""Loads a list of all images contained in a folder and returns a list of (image, name) tuples

	Args:
		img_folder_path (str): Relative path to the folder containing the images (e.g. 'images/')

	Returns:
		list: List of (image, name) tuples
	"""
	if not os.path.isdir(img_folder_path):
		exit('Error: Images folder not found')
	
	img_filenames = os.listdir(img_folder_path)
	images = []
	for i in range(len(img_filenames)):
		if i == N_IMAGES_LIMIT:
			break
		# (image, name)
		images.append(import_image(img_filenames[i])[0])

	n_images = len(images)
	print('Loaded', n_images, 'image' + ('s' if n_images > 1 else ''))
	return images

def embed_watermark(original_img: np.ndarray, img_name: str, watermark: np.ndarray, alpha: float) -> np.ndarray:
	"""Embeds a watermark into the S component of the SVD decomposition of an image's LL DWT subband

	Args:
		original_img (np.ndarray): Image in which to embed the watermark
		img_name (str): Name of the image
		watermark (np.ndarray): Watermark to embed
		alpha (float): Watermark embedding strength coefficient

	Returns:
		np.ndarray: Watermarked image
	"""
	img_coeffs = wavedec2d(original_img, DWT_LEVEL)
	
	from measurements import nvf
	nvf_img = nvf(original_img, 75, 3)
	nvf_coeffs = wavedec2d(nvf_img, DWT_LEVEL)

	masked_coeffs = nvf_coeffs.copy()
	masked_coeffs[0] = nvf_coeffs[0] * img_coeffs[0]
	for i in range(1, len(img_coeffs)):
		masked_coeffs[i] = (nvf_coeffs[i][0] * img_coeffs[i][0], nvf_coeffs[i][1] * img_coeffs[i][1], nvf_coeffs[i][2] * img_coeffs[i][2])

	watermarked_img_coeffs = []
	svd_keys = []

	watermarked_img_coeffs.append(img_coeffs[0])

	for i in range(1, len(img_coeffs)):
		watermarked_h, svd_key = embed_into_svd(masked_coeffs[i][0], watermark, DEFAULT_ALPHA)
		svd_keys.append(svd_key)

		watermarked_v, svd_key = embed_into_svd(masked_coeffs[i][1], watermark, DEFAULT_ALPHA)
		svd_keys.append(svd_key)

		watermarked_d, svd_key = embed_into_svd(masked_coeffs[i][2], watermark, DEFAULT_ALPHA)
		svd_keys.append(svd_key)

		watermarked_img_coeffs.append((watermarked_h, watermarked_v, watermarked_d))


	save_parameters(img_name, masked_coeffs, svd_keys)

	return waverec2d(img_coeffs)

def extract_watermark(original_img: np.ndarray, img_name: str, watermarked_img: np.ndarray) -> np.ndarray:
	"""Extracts the watermark from a watermarked image by appling the reversed embedding algorithm,
	provided that the proper configuration file and the original, unwatermarked, image are available.

	Args:
		original_img (np.ndarray): Original unwatermarked image
		img_name (str): Name of the image
		watermarked_img (np.ndarray): Image from which to extract the watermark

	Returns:
		np.ndarray: Extracted watermark
	"""
	# Extract DWT coefficients from original and watermarked images
	original_coeffs = wavedec2d(original_img, DWT_LEVEL)
	watermarked_coeffs = wavedec2d(watermarked_img, DWT_LEVEL)

	# Build the alpha matrix (mask)
	'''
	from measurements import nvf
	nvf_img = nvf(original_img, 75, 3)
	nvf_coeffs = wavedec2d(nvf_img, DWT_LEVEL)

	nvf_mask = nvf_coeffs.copy()
	for i in range(1, len(original_coeffs)):
		nvf_mask[i] = (nvf_coeffs[i][0] * nvf_img[i][0], nvf_coeffs[i][1] * nvf_img[i][1], nvf_coeffs[i][2] * nvf_img[i][2])
	'''

	# NVF mask can actually be reconstructed, should not need to save it
	(_, nvf_mask, svd_key) = read_parameters(img_name)

	# Create a list with the same format as returned by wavedec2d()
	watermarked_img_coeffs = []
	# watermarked_img_coeffs.append(original_coeffs[0])

	# Import key
	max_sim = -99999
	n_coeffs = len(original_coeffs)
	extracted_watermarks = np.ones([MARK_SIZE, MARK_SIZE, 3 * n_coeffs], dtype=np.float64)
	idx = 0
	for i in range(1, n_coeffs):
		for j in range(3):
			extracted_watermarks[:,:,j] = extract_from_svd(watermarked_coeffs[i][j], svd_key[idx], DEFAULT_ALPHA)
			# plt.figure()
			# plt.imshow(extracted_watermarks[:,:,j], cmap='gray')
			# plt.title('Extracted watermark {} {}'.format(i, j))
			idx += 1

	extracted_watermark = np.ones((MARK_SIZE, MARK_SIZE))
	# Would be better to perform similarity with original watermark and choose best one
	extracted_watermark = np.average(extracted_watermarks, axis=2)

	return extracted_watermark

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