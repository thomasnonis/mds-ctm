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
			'sigma' : sigma
		}
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
			'kernel' : avg_blur_kernel_size
		}
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
			'sigma' : sigma,
			'alpha': alpha
		}
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
			'kernel_size' : kernel_size
		}
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
			'scale' : resize_scale
		}
	}

def awgn(img, std_dev, seed):
	mean = 0.0
	np.random.seed(seed)
	attacked = img + np.random.normal(mean, std_dev, img.shape)
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
			'std_dev':  awgn_std_dev,
			'seed': awgn_seed
		}
	}


def jpeg_compression(img, QF):
	if not os.path.exists(TMP_FOLDER_PATH):
		os.makedirs(TMP_FOLDER_PATH)
	filename = TMP_FOLDER_PATH + str(uuid.uuid1()) + ".jpg"
	cv2.imwrite(filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
	attacked = img # Save the original image as attacked, in case of failure we'll return the original img
	if os.path.exists(filename):
		attacked = cv2.imread(filename, 0)
		attempts = 10
		while(os.path.exists(filename)) and attempts > 0:
			attempts -= 1
			try:
				os.remove(filename)
			except:
				print(f"Error while trying to remove {filename}") # We'll return the original img
	return attacked

def wrapper_jpeg_compression(quality_factor = -1, min_quality_factor = 1, max_quality_factor = 100):
	'''
	JPEG Compression Attack wrapper, returns an attack to be used in do_attacks
	'''
	if quality_factor == -1:
		quality_factor = randint(min_quality_factor, max_quality_factor)
	return {
		'function' : jpeg_compression,
		'arguments' : {
			'QF' : quality_factor
		}
	}


def get_attack_description(attack):
	if attack['function'] == awgn:
		return 'Additive White Gaussian Noise ({}, {})'.format(attack['arguments']['std_dev'], attack['arguments']['seed'])
	elif attack['function'] == average_blur:
		return 'Average Blur ({})'.format(attack['arguments']['kernel'])
	elif attack['function'] == sharpen:
		return 'Sharpen ({}, {})'.format(attack['arguments']['sigma'], attack['arguments']['alpha'])
	elif attack['function'] == jpeg_compression:
		return 'JPEG ({})'.format(attack['arguments']['QF'])
	elif attack['function'] == median:
		return 'Median ({})'.format(attack['arguments']['kernel_size'])
	elif attack['function'] == gaussian_blur:
		return 'Gaussian Blur ({})'.format(attack['arguments']['sigma'])
	elif attack['function'] == resizing:
		return 'Resize ({})'.format(attack['arguments']['scale'])
	else:
		print(f"Whoops function {attack['function']} does not exist!")

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

# Reasonable attack parameters 
# Orderd by most impactful on wpsnr to least impactful
attack_parameters = {
	awgn : [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5],
	average_blur : [7, 5, 3],
	sharpen : [(0.65, 0.2), (0.35, 1.4), (0.45, 0.3), (0.6, 0.2), (0.35, 1.3), (0.55, 0.2), (0.4, 0.4), (0.35, 1.2), (0.5, 0.2), (0.35, 1.1), (0.35, 1.0), (0.45, 0.2), (0.4, 0.3), (0.35, 0.9), (0.35, 0.8), (0.65, 0.1), (0.35, 0.7), (0.6, 0.1), (0.55, 0.1), (0.4, 0.2), (0.35, 0.6), (0.5, 0.1), (0.35, 0.5), (0.45, 0.1), (0.35, 0.4), (0.4, 0.1), (0.35, 0.3), (0.3, 1.9), (0.35, 0.2), (0.3, 1.8), (0.3, 1.7), (0.3, 1.6), (0.3, 1.5), (0.15, 1.9), (0.2, 1.9), (0.25, 1.9), (0.3, 1.4), (0.15, 1.8), (0.2, 1.8), (0.25, 1.8), (0.3, 1.3), (0.15, 1.7), (0.2, 1.7), (0.25, 1.7), (0.3, 1.2), (0.15, 1.6), (0.2, 1.6), (0.25, 1.6), (0.15, 1.5), (0.2, 1.5), (0.25, 1.5), (0.3, 1.1), (0.15, 1.4), (0.2, 1.4), (0.25, 1.4), (0.3, 1.0), (0.15, 1.3), (0.2, 1.3), (0.25, 1.3), (0.35, 0.1), (0.3, 0.9), (0.15, 1.2), (0.2, 1.2), (0.25, 1.2), (0.15, 1.1), (0.2, 1.1), (0.25, 1.1), (0.3, 0.8), (0.15, 1.0), (0.2, 1.0), (0.25, 1.0), (0.3, 0.7), (0.15, 0.9), (0.2, 0.9), (0.25, 0.9), (0.3, 0.6), (0.15, 0.8), (0.2, 0.8), (0.25, 0.8), (0.15, 0.7), (0.2, 0.7), (0.25, 0.7), (0.3, 0.5), (0.15, 0.6), (0.2, 0.6), (0.25, 0.6), (0.3, 0.4), (0.15, 0.5), (0.2, 0.5), (0.25, 0.5), (0.3, 0.3), (0.15, 0.4), (0.2, 0.4), (0.25, 0.4), (0.15, 0.3), (0.2, 0.3), (0.25, 0.3), (0.3, 0.2), (0.15, 0.2), (0.2, 0.2), (0.25, 0.2), (0.3, 0.1), (0.15, 0.1), (0.2, 0.1), (0.25, 0.1)],
	jpeg_compression : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
	resizing : [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
	gaussian_blur : [2.99, 2.98, 2.97, 2.96, 2.95, 2.94, 2.93, 2.92, 2.91, 2.9, 2.89, 2.88, 2.87, 2.86, 2.85, 2.84, 2.83, 2.82, 2.81, 2.8, 2.79, 2.78, 2.77, 2.76, 2.75, 2.74, 2.73, 2.72, 2.71, 2.7, 2.69, 2.68, 2.67, 2.66, 2.65, 2.64, 2.63, 2.62, 2.61, 2.6, 2.59, 2.58, 2.57, 2.56, 2.55, 2.54, 2.53, 2.52, 2.51, 2.5, 2.49, 2.48, 2.47, 2.46, 2.45, 2.44, 22.24, 2.23, 2.22, 2.21, 2.2, 2.19, 2.18, 2.17, 2.16, 2.15, 2.14, 2.13, 2.12, 2.11, 2.1, 2.09, 2.08, 2.07, 2.06, 2.05, 2.04, 2.03, 2.02, 2.01, 2.0, 1.99, 1.98, 1.97, 1.96, 1.95, 1.94, 1.93, 1.92, 1.91, 1.9, 1.89, 1.88, 1.87, 1.86, 1.85, 1.84, 1.83, 1.82, 1.81, 1.8, 1.79, 1.78, 1.77, 1.76, 1.75, 1.74, 1.73, 1.72, 1.71, 1.7, 1.69, 1.68, 1.67, 1.66, 1.65, 1.64, 1.63, 1.62, 1.61, 1.6, 1.59, 1.58, 1.57, 1.56, 1.55, 1.54, 1.53, 1.52, 1.51, 1.5, 1.49, 1.48, 1.47, 1.46, 1.45, 1.44, 1.43, 1.42, 1.41, 1.4, 1.39, 1.38, 1.37, 1.36, 1.35, 1.34, 1.33, 1.32, 1.31, 1.3, 1.29, 1.28, 1.27, 1.26, 1.25, 1.24, 1.23, 1.22, 1.21, 1.2, 1.19, 1.18, 1.17, 1.16, 1.15, 1.14, 1.13, 1.12, 1.11, 1.1, 1.09, 1.08, 1.07, 1.06, 1.05, 1.04, 1.03, 1.02, 1.01, 1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.8, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0.7, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.6, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.45, 0.46, 0.44, 0.16, 0.31, 0.43, 0.13, 0.41, 0.42, 0.3, 0.4, 0.39, 0.14, 0.4, 0.19, 0.38, 0.37, 0.26, 0.36, 0.18, 0.35, 0.34, 0.33, 0.32, 0.15, 0.17, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.27, 0.28, 0.29],
	median : [7, 5, 3],
}

def get_indexed_random_attacks(num_attacks):
	'''
	Returns a list containing num_attacks random Attacks with random parameters. 
	This list is to be used with do_attacks.
	'''
	attacks_list = []
	index_list = []
	for _ in range(0, num_attacks):
		attack = randint(0, N_AVAILABLE_ATTACKS - 1)

		if attack == 0:
			idx = randint(0,len(attack_parameters[awgn])-1)
			std_dev = attack_parameters[awgn][idx]
			attacks_list.append(wrapper_awgn(std_dev))
			index_list.append(idx)
		elif attack == 1:		
			idx = randint(0,len(attack_parameters[average_blur])-1)
			blur_kernel_size = attack_parameters[average_blur][idx]
			attacks_list.append(wrapper_average_blur(blur_kernel_size))
			index_list.append(idx)
		elif attack == 2:
			idx = randint(0,len(attack_parameters[sharpen])-1)
			sigma,alpha = attack_parameters[sharpen][idx]
			attacks_list.append(wrapper_sharpen(sigma,alpha))
			index_list.append(idx)
		elif attack == 3:
			idx = randint(0,len(attack_parameters[jpeg_compression])-1)
			qf = attack_parameters[jpeg_compression][idx]
			attacks_list.append(wrapper_jpeg_compression(qf))
			index_list.append(idx)
		elif attack == 4:
			idx = randint(0,len(attack_parameters[resizing])-1)
			scale = attack_parameters[resizing][idx]
			d = wrapper_resizing(scale)
			attacks_list.append(d)
			index_list.append(idx)
		elif attack == 5:
			idx = randint(0,len(attack_parameters[median])-1)
			kernel_size = attack_parameters[median][idx]
			d = wrapper_median(kernel_size)
			attacks_list.append(d)
			index_list.append(idx)
		elif attack == 6:
			idx = randint(0,len(attack_parameters[gaussian_blur])-1)
			sigma = attack_parameters[gaussian_blur][idx]
			d = wrapper_gaussian_blur(sigma)
			attacks_list.append(d)
			index_list.append(idx)
		else:
			exit('Invalid attack %d, check that N_AVAILABLE_ATTACKS is correct' % attack)
	return attacks_list, index_list


def parse_parameters(function, params):
	if function == awgn:
		awgn_seed = randint(0, 1000)
		awgn_std_dev = params
		return {'std_dev':  awgn_std_dev, 'seed' : awgn_seed}
	elif function == average_blur:
		kernel = params
		return {'kernel' : kernel}
	elif function == sharpen:
		sigma, alpha = params
		return {"sigma" : sigma,"alpha": alpha}
	elif function == jpeg_compression:
		qf = params
		return {'QF' : qf}
	elif function == resizing:
		scale = params
		return {'scale' : scale}
	elif function == median:
		kernel = params
		return {'kernel_size' : kernel}
	elif function == gaussian_blur:
		sigma = params
		return {'sigma' : sigma}

def describe_attacks(attacks_list):
	'''
	Returns a description for the attacks in an attacks list
	'''
	return ", ".join([get_attack_description(attack) for attack in attacks_list])

def do_attacks(img, attacks_list):
	'''
	Execute a list of attacks on an image one after the other sequentially
	'''
	for attack in attacks_list:
		attack['arguments']['img'] = img
		img = attack['function'](**attack['arguments'])
		attack['arguments'].pop('img', None)
	return img, describe_attacks(attacks_list)