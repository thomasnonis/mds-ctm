import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
import matplotlib as mpl
from skimage.transform import rescale
import numpy as np
import cv2

# The only allowed attacks are:
# - Additive White Gaussian Noise
# - Blur
# - Sharpen
# - JPEG Compression
# - Resize
# - Median

def gaussian_blur(img, sigma):
	return gaussian_filter(img, sigma)

def average_blur(img, kernel):
	return cv2.blur(img, kernel)

def bilateral_blur():
	pass

def sharpen(img, sigma, alpha):
	blurred = gaussian_filter(img, sigma)
	return img + alpha * (img - blurred)

def median(img, kernel_size):
	return medfilt(img, kernel_size)

def resize(img, scale):
	x, y = img.shape
	attacked = rescale(img, scale)
	attacked = rescale(attacked, 1/scale)
	attacked = attacked[:x, :y]
	return attacked

def awgn(img, mean, std, seed):
	np.random.seed(seed)
	attacked = img + np.random.normal(mean, std, img.shape)
	attacked = np.clip(attacked, 0, 255)
	return attacked

def jpeg():
	pass