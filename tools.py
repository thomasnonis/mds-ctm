import os
import numpy as np
import cv2
import pickle
from transforms import wavedec2d, waverec2d
from config import MARK_SIZE, DWT_LEVEL
import matplotlib.pyplot as plt
import concurrent.futures
import traceback


from config import *
# from attacks import do_random_attacks, get_random_attacks
# from measurements import similarity
# from random import randint, random

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
	plt.figure(figsize=(15, 6))
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
	return img_name, float(alpha), svd_key

def import_images(img_folder_path: str) -> list:
	"""Loads a list of all images contained in a folder and returns a list of (image, name) tuples

	Args:
		img_folder_path (str): Relative path to the folder containing the images (e.g. 'images/')

	Returns:
		list: List of (image, name) tuples
	"""
	if not os.path.isdir(img_folder_path):
		exit('Error: Images folder not found')
	
	images = []
	for img_filename in os.listdir(img_folder_path):
		# (image, name)
		images.append((cv2.imread(img_folder_path + img_filename, cv2.IMREAD_GRAYSCALE), img_filename.split('.')[-2]))

	n_images = len(images)
	print('Loaded', n_images, 'image' + ('s' if n_images > 1 else ''))
	
	return images

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
	(_, alpha, svd_key) = read_parameters(img_name)
	original_coeffs = wavedec2d(original_img, DWT_LEVEL)
	original_LL = original_coeffs[0]

	original_ll_u, original_ll_s, original_ll_v = np.linalg.svd(original_LL)
	original_ll_s = np.diag(original_ll_s)
	
	watermarked_coeffs = wavedec2d(watermarked_img, DWT_LEVEL)
	watermarked_LL = watermarked_coeffs[0]

	watermarked_ll_u, watermarked_ll_s, watermarked_ll_v = np.linalg.svd(watermarked_LL)
	watermarked_ll_s = np.diag(watermarked_ll_s)
	
	# original_s_ll_d_u, original_s_ll_d_s, original_s_ll_d_v 
	s_ll_d = svd_key[0] @ watermarked_ll_s @ svd_key[1]

	# Initialize the watermark matrix
	watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)
	
	# Extract the watermark
	for i in range(0,MARK_SIZE):
		for j in range(0,MARK_SIZE):
			watermark[i][j] = (s_ll_d[i][j] - original_ll_s[i][j]) / alpha

	return watermark

'''
def attack_and_extract_and_sim(img_name, original_img, watermarked_img, original_watermark ):
	attacks_list = get_random_attacks(randint(1, MAX_N_ATTACKS))
	attacked_img, attacks_list = do_random_attacks(watermarked_img,attacks_list)

	extracted_watermark = extract_watermark(original_img, img_name, attacked_img)


	sim = similarity(original_watermark, extracted_watermark)
	return sim, extracted_watermark
'''

def embed_watermark(original_img: np.ndarray, img_name: str, watermark: np.ndarray, alpha: float, order_of_execution: int=-1) -> np.ndarray:
	"""Embeds a watermark into the S component of the SVD decomposition of an image's LL DWT subband

	Args:
		original_img (np.ndarray): Image in which to embed the watermark
		img_name (str): Name of the image
		watermark (np.ndarray): Watermark to embed
		alpha (float): Watermark embedding strength coefficient

	Returns:
		np.ndarray: Watermarked image
	"""
	coeffs = wavedec2d(original_img, DWT_LEVEL)
	ll = coeffs[0]

	ll_svd, svd_key = embed_into_svd(ll, watermark, alpha)

	save_parameters(img_name, alpha, svd_key)

	coeffs[0] = ll_svd
	if order_of_execution == -1:
		return waverec2d(coeffs)
	else:
		return waverec2d(coeffs), order_of_execution

def multithreaded_workload(function, work):
	with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
		future_to_report = {executor.submit(function, *unit_of_work, order_of_execution): unit_of_work for order_of_execution,unit_of_work in enumerate(work)}

	tmp_results = []
	for future in concurrent.futures.as_completed(future_to_report):
		result = future_to_report[future]
		try:
			r = future.result()
			tmp_results.append(r)
		except Exception as exc:
			print("Exception!", "{}".format('%r generated an exception: %s' % (result, traceback.format_exc())))
	tmp_results = sorted(tmp_results, key = lambda x: x[1])
	results = [result[0] for result in tmp_results]
	
	return results

def multiprocessed_workload(function, work):
	with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
		future_to_report = {executor.submit(function, *unit_of_work, order_of_execution): unit_of_work for order_of_execution,unit_of_work in enumerate(work)}

	tmp_results = []
	for future in concurrent.futures.as_completed(future_to_report):
		result = future_to_report[future]
		try:
			r = future.result()
			tmp_results.append(r)
		except Exception as exc:
			print("Exception!", "{}".format('%r generated an exception: %s' % (result, traceback.format_exc())))
	tmp_results = sorted(tmp_results, key = lambda x: x[1])
	results = [result[0] for result in tmp_results]
	
	return results