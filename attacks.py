from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt2d
from skimage.transform import rescale, resize as skimage_resize
import numpy as np
import cv2
from PIL import Image
import os
from random import randint, random
import uuid

from config import *

# The only allowed attacks are:
# - Additive White Gaussian Noise
# - Blur
# - Sharpen
# - JPEG Compression
# - Resize
# - Median

def gaussian_blur(img, sigma):
	return gaussian_filter(img, sigma)

def wrapper_gaussian_blur(sigma = -1, min_sigma = 1, max_sigma = 15):
	'''
	Gaussian Blur Attack wrapper, returns an attack to be used in do_attacks
	'''
	if sigma == -1:
		sigma = round(randint(min_sigma, max_sigma) * 2 / 10, 1) # 0.2, 0.4, ..., 3.0
	return {
		'function' : gaussian_blur,
		'arguments' : {
			"sigma" : sigma
		},
		'description' : 'Gaussian Blur ({})'.format(sigma)
	}

def average_blur(img, kernel):
	return cv2.blur(img, (kernel,kernel))

def wrapper_average_blur(avg_blur_kernel_size = -1, min_avg_blur_kernel_size = 1, max_avg_blur_kernel_size = 3):
	'''
	Average Blur Attack wrapper, returns an attack to be used in do_attacks
	'''
	if avg_blur_kernel_size == -1:
		avg_blur_kernel_size = (randint(min_avg_blur_kernel_size, max_avg_blur_kernel_size) * 2) + 1 # 3, 5, 7
	return {
		'function' : average_blur,
		'arguments' : {
			"kernel" : avg_blur_kernel_size
		},
		'description' : 'Average Blur ({})'.format(avg_blur_kernel_size)
	}

def sharpen(img, sigma, alpha):
	blurred = gaussian_filter(img, sigma)
	return img + alpha * (img - blurred)

def wrapper_sharpen(sigma = -1, alpha = -1, min_sigma = 0.2, max_sigma = 5, min_alpha = 0.1, max_alpha = 5):
	'''
	Sharpen Attack wrapper, returns an attack to be used in do_attacks
	'''
	# TODO: fix ranges
	if sigma == -1:
		sigma = round((random() * (max_sigma - min_sigma)) + min_sigma, 2)	
	if alpha == -1:
		alpha = round(random() * (max_alpha - min_alpha) + min_alpha, 2)
	return {
		'function' : sharpen,
		'arguments' : {
			"sigma" : sigma,
			"alpha": alpha
		},
		'description' : 'Sharpen ({}, {})'.format(sigma, alpha)
	}

def median(img, kernel_size):
	return medfilt2d(img, kernel_size)

def wrapper_median(kernel_size = -1, min_kernel_size = 1, max_kernel_size = 3):
	'''
	Median Attack wrapper, returns an attack to be used in do_attacks
	'''
	if kernel_size == -1:
		kernel_size = randint(min_kernel_size, max_kernel_size) * 2 + 1 # 3, 5, 7 (5 and 7 are not valid standalone, but may become valid after a sharpen)
	
	return {
		'function' : median,
		'arguments' : {
			"kernel_size" : kernel_size
		},
		'description' : 'Median ({})'.format(kernel_size)
	}

def resizing(img, scale):
  x, y = img.shape
  _x = int(x*scale)
  _y = int(y*scale)

  attacked = cv2.resize(img, (_x, _y))
  attacked = cv2.resize(attacked, (x, y))

  return attacked


def wrapper_resizing(resize_scale = -1, min_resize_scale = 1, max_resize_scale = 9):
	'''
	Resizing Attack wrapper, returns an attack to be used in do_attacks
	'''
	if resize_scale == -1:
		resize_scale = round(randint(min_resize_scale, max_resize_scale) / 10, 1) # 0.1, 0.2, ..., 0.9
	return {
		'function' : resizing,
		'arguments' : {
			"scale" : resize_scale
		},
		'description' : 'Resize ({})'.format(resize_scale)
	}

def awgn(img, std, seed):
	mean = 0.0
	np.random.seed(seed)
	attacked = img + np.random.normal(mean, std, img.shape)
	attacked = np.clip(attacked, 0, 255)
	return attacked


def wrapper_awgn(awgn_std_dev = -1, min_std_dev = 1, max_std_dev = 10):
	'''
	AWGN Attack wrapper, returns an attack to be used in do_attacks
	'''
	if awgn_std_dev == -1:
		awgn_std_dev = randint(min_std_dev, max_std_dev) * 5 # 5, 10, 15, 20, 25, 30, 35, 40, 45
	awgn_seed = randint(0, 1000)
	return {
		'function' : awgn,
		'arguments' : { 
			"std":  awgn_std_dev,
			"seed": awgn_seed
		},
		'description' : 'Additive White Gaussian Noise ({}, {})'.format(awgn_std_dev, awgn_seed)
	}


def jpeg_compression(img, QF):
	filename = str(uuid.uuid1()) + ".jpg"
	cv2.imwrite(filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
	attacked = img # Save the original image as attacked, in case of failure we'll return the original img
	if os.path.exists(filename):
		attacked = cv2.imread(filename, 0)
		try:
			os.remove(filename)
		except:
			print(f"Error while trying to remove {filename}") # We'll return the original img
	return attacked

def wrapper_jpeg_compression(quality_factor = -1, min_quality_factor = 1, max_quality_factor = 99):
	'''
	JPEG Compression Attack wrapper, returns an attack to be used in do_attacks
	'''
	if quality_factor == -1:
		quality_factor = randint(min_quality_factor, max_quality_factor)
	return {
		'function' : jpeg_compression,
		'arguments' : {
			"QF" : quality_factor
		},
		'description' : 'JPEG ({})'.format(quality_factor)
	}

def get_random_attacks(num_attacks):
	'''
	Returns a list containing num_attacks random Attacks with random parameters. 
	This list is to be used with do_attacks.
	'''
	attacks_list = []

	for _ in range(0, num_attacks):
		attack = randint(0, N_AVAILABLE_ATTACKS - 1)

		if attack == 0:
			attacks_list.append(wrapper_awgn())
		elif attack == 1:			
			attacks_list.append(wrapper_average_blur())
		elif attack == 2:
			attacks_list.append(wrapper_sharpen())
		elif attack == 3:
			attacks_list.append(wrapper_jpeg_compression())
		elif attack == 4:			
			attacks_list.append(wrapper_resizing())
		elif attack == 5:
			attacks_list.append(wrapper_median())
		elif attack == 6:			
			attacks_list.append(wrapper_gaussian_blur())
		else:
			exit('Invalid attack %d, check that N_AVAILABLE_ATTACKS is correct' % attack)
	return attacks_list

def describe_attacks(attacks_list):
	'''
	Returns a description for the attacks in an attacks list
	'''
	return ", ".join([attacks['description'] for attacks in attacks_list])

def do_attacks(img, attacks_list):
	'''
	Execute a list of attacks on an image one after the other sequentially
	'''
	for attack in attacks_list:
		attack['arguments']['img'] = img
		img = attack['function'](**attack['arguments'])
	return img, describe_attacks(attacks_list)