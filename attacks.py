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

# Naming convention for images attacked by us: failedfouriertransform_groupB_imageName.bmp

# Reasonable attack parameters 
# Orderd by most impactful on wpsnr to least impactful
attack_parameters = {
	"awgn" : [45, 40, 35, 30, 25, 20, 15, 10, 5],
	"average_blur" : [7, 5, 3],
	"sharpen" : [(0.95, 1.9), (0.9, 1.9), (0.85, 1.9), (0.8, 1.9), (0.75, 1.9), (0.7, 1.9), (0.65, 1.9), (0.95, 1.6), (0.6, 1.9), (0.9, 1.6), (0.85, 1.6), (0.8, 1.6), (0.75, 1.6), (0.55, 1.9), (0.7, 1.6), (0.65, 1.6), (0.6, 1.6), (0.5, 1.9), (0.95, 1.3), (0.9, 1.3), (0.85, 1.3), (0.8, 1.3), (0.55, 1.6), (0.75, 1.3), (0.7, 1.3), (0.65, 1.3), (0.5, 1.6), (0.45, 1.9), (0.6, 1.3), (0.95, 1.0), (0.55, 1.3), (0.9, 1.0), (0.85, 1.0), (0.8, 1.0), (0.75, 1.0), (0.45, 1.6), (0.7, 1.0), (0.5, 1.3), (0.65, 1.0), (0.6, 1.0), (0.55, 1.0), (0.45, 1.3), (0.4, 1.9), (0.95, 0.7), (0.9, 0.7), (0.5, 1.0), (0.85, 0.7), (0.8, 0.7), (0.75, 0.7), (0.7, 0.7), (0.4, 1.6), (0.65, 0.7), (0.6, 0.7), (0.45, 1.0), (0.55, 0.7), (0.4, 1.3), (0.5, 0.7), (0.95, 0.4), (0.45, 0.7), (0.9, 0.4), (0.85, 0.4), (0.4, 1.0), (0.8, 0.4), (0.75, 0.4), (0.7, 0.4), (0.65, 0.4), (0.6, 0.4), (0.55, 0.4), (0.5, 0.4), (0.4, 0.7), (0.35, 1.9), (0.45, 0.4), (0.35, 1.6), (0.35, 1.3), (0.4, 0.4), (0.35, 1.0), (0.95, 0.1), (0.9, 0.1), (0.85, 0.1), (0.8, 0.1), (0.75, 0.1), (0.35, 0.7), (0.7, 0.1), (0.65, 0.1), (0.6, 0.1), (0.55, 0.1), (0.5, 0.1), (0.45, 0.1), (0.35, 0.4), (0.4, 0.1), (0.3, 1.9), (0.3, 1.6), (0.2, 1.9), (0.25, 1.9), (0.3, 1.3), (0.2, 1.6), (0.25, 1.6), (0.3, 1.0), (0.35, 0.1), (0.2, 1.3), (0.25, 1.3), (0.2, 1.0), (0.25, 1.0), (0.3, 0.7), (0.2, 0.7), (0.25, 0.7), (0.3, 0.4), (0.2, 0.4), (0.25, 0.4), (0.3, 0.1), (0.2, 0.1), (0.25, 0.1)],
	"jpeg_compression" : [1, 2, 3, 4, 5, 7, 9, 11, 13, 16, 19, 22, 25, 30, 35, 40, 50, 60, 65, 70, 75, 80, 82, 84, 86, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
	"resizing" : [0.2, 0.25 ,0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
	"median" : [3, 2.8, 2.6, 2.4, 2.2, 2, 1.8, 1.6, 1.4, 1.2, 1, 0.8, 0.6, 0.4, 0.2, 0.13, 0.3],
	"gaussian_blur" : [7, 5, 3],
}

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

def wrapper_sharpen(sigma = -1, alpha = -1, min_sigma = 0.2, max_sigma = 1, min_alpha = 0.1, max_alpha = 2):
	'''
	Sharpen Attack wrapper, returns an attack to be used in do_attacks
	'''
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


def wrapper_resizing(resize_scale = -1, min_resize_scale = 2, max_resize_scale = 9):
	'''
	Resizing Attack wrapper, returns an attack to be used in do_attacks
	'''
	if resize_scale == -1:
		resize_scale = round(randint(min_resize_scale, max_resize_scale) / 10, 1) # 0.2, ..., 0.9
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

def wrapper_jpeg_compression(quality_factor = -1, min_quality_factor = 0, max_quality_factor = 36):
	'''
	JPEG Compression Attack wrapper, returns an attack to be used in do_attacks
	'''
	if quality_factor == -1:
		qf = [1, 2, 3, 4, 5, 7, 9, 11, 13, 16, 19, 22, 25, 30, 35, 40, 50, 60, 65, 70, 75, 80, 82, 84, 86, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100] # distribution optimized for wpsnr impact
		
		quality_factor = qf[randint(min_quality_factor, min(max_quality_factor,36))]
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