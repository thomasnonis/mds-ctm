import numpy as np
import cv2 as cv
from transforms import wavedec2d
import os
from scipy.signal import convolve2d
from math import sqrt

# ////VARIABLES START////
ALPHA = 23
BETA = 0.2
DETECTION_THRESHOLD = 12
MARK_SIZE = 32
ALPHA = 25
BETA = 0.05
DWT_LEVEL = 1
svd_keys = {}
# ////VARIABLES END////

TEAM_NAME = 'failedfouriertransform'

alpha = {}
alpha['lena'] = 0.00001

beta = {}
beta['lena'] = 0.00001

svd_key = {}
svd_key['lena'] = (np.ones((3, 3)), np.ones((3, 3)))

def similarity(img1,img2):
	#Computes the similarity measure between the original and the new watermarks.
	return np.sum(np.multiply(img1, img2)) / np.sqrt(np.sum(np.multiply(img2, img2)))

def csf(img):
	if not os.path.isfile('csf.csv'):  
		os.system('python -m wget "https://drive.google.com/uc?export=download&id=1w43k1BTfrWm6X0rqAOQhrbX6JDhIKTRW" -o csf.csv')
	if not os.path.isfile('csf.csv'):
		raise FileNotFoundError('csf.csv not found')

	csf = np.genfromtxt('csf.csv', delimiter=',')
	return convolve2d(img, np.rot90(csf,2), mode='same')

def wpsnr(img1: np.ndarray, img2: np.ndarray):
	img1 = np.float32(img1)/255.0
	img2 = np.float32(img2)/255.0

	difference = img1 - img2
	same = not np.any(difference)
	if same is True:
		return 9999999
	
	ew = csf(difference)
	decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2)))) # this is something that can be optimized by using numerical values instead of db
	return decibels

def local_variance(img, i, j, window_size):
	from math import floor
	# compute the local variance of the image within a square window of window_size centered at position (i,j)

	i_min = i-floor(window_size/2)
	i_max = i+floor(window_size/2) + 1
	j_min = j-floor(window_size/2)
	j_max = j+floor(window_size/2) + 1

	if i_min < 0:
		i_min = 0
	elif i_max > img.shape[0]:
		i_max = img.shape[0]

	if j_min < 0:	
		j_min = 0
	elif j_max > img.shape[1]:
		j_max = img.shape[1]
	mean = np.mean(img[i_min:i_max, j_min:j_max])

	variance = 0
	cnt = 0
	for x in range(i_min, i_max):
		for y in range(j_min, j_max):
			if x >= 0 and x < img.shape[0] and y >= 0 and y < img.shape[1]:
				cnt += 1
				variance += (img[x, y] - mean)**2
	
	variance = variance / cnt

	return variance

def nvf(img, D, window_size):
	max_variance = 0

	variance = np.zeros([img.shape[0], img.shape[1]])

	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			variance[i,j] = local_variance(img, i, j, window_size)
			if variance[i,j] > max_variance:
				max_variance = variance[i,j]

	theta = D/max_variance
	return 1/(1+theta*variance)

def extract_from_svd(original_img, watermarked_img, svd_key, alpha):  
	# Perform SVD decomposition of original_img
	_, svd_o_s, _ = np.linalg.svd(original_img)

	# Convert S from a 1D vector to a 2D diagonal matrix
	svd_o_s = np.diag(svd_o_s)

	# Perform SVD decomposition of watermarked_img
	_, svd_w_s, _ = np.linalg.svd(watermarked_img)

	# Convert S from a 1D vector to a 2D diagonal matrix
	svd_w_s = np.diag(svd_w_s)

	# Reconstruct S component using embedding key components
	s_ll_d = svd_key[0] @ svd_w_s @ svd_key[1]

	# Initialize the watermark matrix
	watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)
	
	# Extract the watermark
	for i in range(0,MARK_SIZE):
		for j in range(0,MARK_SIZE):
			watermark[i][j] = (s_ll_d[i][j] - svd_o_s[i][j]) / alpha

	return watermark

# Split function
def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def extract_watermark_tn(original_img: np.ndarray, img_name: str, watermarked_img: np.ndarray, attacked_img: np.ndarray) -> np.ndarray:
	"""Extracts the watermark from a watermarked image by appling the reversed embedding algorithm,
	provided that the proper configuration file and the original, unwatermarked, image are available.

	Args:
		original_img (np.ndarray): Original unwatermarked image
		img_name (str): Name of the image
		watermarked_img (np.ndarray): Image from which to extract the watermark

	Returns:
		np.ndarray: Extracted watermark
	"""
	original_coeffs = wavedec2d(original_img, DWT_LEVEL)
	original_h1 = original_coeffs[2][0]
	original_h2 = original_coeffs[1][0]
	original_v1 = original_coeffs[2][1]

	attacked_coeffs = wavedec2d(attacked_img, DWT_LEVEL)
	attacked_h1 = attacked_coeffs[2][0]
	attacked_h2 = attacked_coeffs[1][0]
	attacked_v1 = attacked_coeffs[2][1]

	attacked_watermarks = np.empty((1, MARK_SIZE, MARK_SIZE))

	# This is of size 128x128, while all others are 256x256!
	attacked_watermarks[0] = extract_from_svd(original_h2, attacked_h2, svd_key[img_name], alpha[img_name])

	original_h1_strength = nvf(csf(original_h1), 75, 3)
	original_v1_strength = nvf(csf(original_v1), 75, 3)

	attacked_watermark_h1 = (original_h1 - attacked_h1) / ((1-original_h1_strength) * BETA) 
	attacked_watermark_v1 = (original_v1 - attacked_v1) / ((1-original_v1_strength) * BETA)

	attacked_watermark_h1 = (attacked_watermark_h1 + 1) / 2
	attacked_watermark_v1 = (attacked_watermark_v1 + 1) / 2

	attacked_watermark_h1_mtx = split(attacked_watermark_h1, MARK_SIZE, MARK_SIZE)
	attacked_watermark_v1_mtx = split(attacked_watermark_v1, MARK_SIZE, MARK_SIZE)
	for i in range(0, attacked_watermark_h1_mtx.shape[0]):
		attacked_watermarks = np.append(attacked_watermarks, [attacked_watermark_h1_mtx[i]], axis=0)
		attacked_watermarks = np.append(attacked_watermarks, [attacked_watermark_v1_mtx[i]], axis=0)

	return np.mean(attacked_watermarks, axis=0)

def detection(original_path, watermarked_path, attacked_path):
	original_img = cv.imread(original_path, cv.IMREAD_GRAYSCALE)
	watermarked_img = cv.imread(watermarked_path, cv.IMREAD_GRAYSCALE)
	attacked_img = cv.imread(attacked_path, cv.IMREAD_GRAYSCALE)

	# Our watermarked images must be named: imageName_failedfouriertransform.bmp
	# Watermarked images by other groups will be named: groupB_imageName.bmp
	# Attacked images must be named: failedfouriertransform_groupB_imageName.bmp
	'''
	pixel
	ef26420c
	you_shall_not_mark
	blitz
	omega
	howimetyourmark
	weusedlsb
	thebavarians
	theyarethesamepicture
	dinkleberg
	failedfouriertransform
	'''
	if original_path.lower().split('/')[-1].split('_')[-1].split('.')[-1] == TEAM_NAME:
		img_name = original_path.lower().split('/')[-1].split('_')[0]
	else :
		img_name = original_path.lower().split('/')[-1].split('.')[-1].split('_')[-1]

	original_watermark = extract_watermark_tn(original_img, img_name, watermarked_img, watermarked_img)
	attacked_watermark = extract_watermark_tn(original_img, img_name, watermarked_img, attacked_img)

	if similarity(attacked_watermark, original_watermark) > DETECTION_THRESHOLD:
		has_watermark = True
	else:
		has_watermark = False

	return has_watermark, wpsnr(watermarked_img, attacked_img)