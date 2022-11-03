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

def average_blur(img, kernel):
	return cv2.blur(img, (kernel,kernel))

def sharpen(img, sigma, alpha):
	blurred = gaussian_filter(img, sigma)
	return img + alpha * (img - blurred)

def median(img, kernel_size):
	return medfilt2d(img, kernel_size)

def resizing(img, scale):
  x, y = img.shape
  _x = int(x*scale)
  _y = int(y*scale)

  attacked = cv2.resize(img, (_x, _y))
  attacked = cv2.resize(attacked, (x, y))

  return attacked

def awgn(img, std, seed):
	mean = 0.0
	np.random.seed(seed)
	attacked = img + np.random.normal(mean, std, img.shape)
	attacked = np.clip(attacked, 0, 255)
	return attacked


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

def get_random_attacks(num_attacks):
	attacks_list = []

	for _ in range(0, num_attacks):
		attack = randint(0, N_AVAILABLE_ATTACKS - 1)

		if attack == 0:
			awgn_std_dev = randint(1, 10) * 5 # 5, 10, 15, 20, 25, 30, 35, 40, 45
			awgn_seed = randint(0, 1000)
			attacks_list.append(
				{
					'function' : awgn,
					'arguments' : { 
						"std":  awgn_std_dev,
						"seed": awgn_seed
					},
					'description' : 'Additive White Gaussian Noise ({}, {})'.format(awgn_std_dev, awgn_seed)
				}
			)
		elif attack == 1:
			avg_blur_kernel_size = (randint(1, 3) * 2) + 1 # 3, 5, 7
			attacks_list.append(
				{
					'function' : average_blur,
					'arguments' : {
						"kernel" : avg_blur_kernel_size
					},
					'description' : 'Average Blur ({})'.format(avg_blur_kernel_size)
				}
			)
		elif attack == 2:
			# TODO: fix ranges
			sharpen_sigma = round((random() * (5 - 0.2)) + 0.2, 2)
			sharpen_alpha = round(random() * (5 - 0.1) + 0.1, 2)

			attacks_list.append(
				{
					'function' : sharpen,
					'arguments' : {
						"sigma" : sharpen_sigma,
						"alpha": sharpen_alpha
					},
					'description' : 'Sharpen ({}, {})'.format(sharpen_sigma, sharpen_alpha)
				}
			)
		elif attack == 3:
			qf = [1, 2, 3, 4, 5, 7, 9, 11, 13, 16, 19, 22, 25, 30, 35, 40, 50, 60, 65, 70, 75, 80, 82, 84, 86, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100] # distribution optimized for wpsnr impact
			jpeg_quality_factor = qf[randint(0, 36)] # len(qf) = 37
			attacks_list.append(
				{
					'function' : jpeg_compression,
					'arguments' : {
						"QF" : jpeg_quality_factor
					},
					'description' : 'JPEG ({})'.format(jpeg_quality_factor)
				}
			)
		elif attack == 4:
			resize_scale = round(randint(1, 9) / 10, 1) # 0.1, 0.2, ..., 0.9
			attacks_list.append(
				{
					'function' : resizing,
					'arguments' : {
						"scale" : resize_scale
					},
					'description' : 'Resize ({})'.format(resize_scale)
				}
			)
		elif attack == 5:
			median_kernel_size = randint(1, 3) * 2 + 1 # 3, 5, 7 (5 and 7 are not valid standalone, but may become valid after a sharpen)
			attacks_list.append(
				{
					'function' : median,
					'arguments' : {
						"kernel_size" : median_kernel_size
					},
					'description' : 'Median ({})'.format(median_kernel_size)
				}
			)
		elif attack == 6:
			sigma = round(randint(1, 15) * 2 / 10, 1) # 0.2, 0.4, ..., 3.0
			attacks_list.append(
				{
					'function' : gaussian_blur,
					'arguments' : {
						"sigma" : sigma
					},
					'description' : 'Gaussian Blur ({})'.format(sigma)
				}
			)
		else:
			exit('Invalid attack %d, check that N_AVAILABLE_ATTACKS is correct' % attack)
	return attacks_list

def describe_attacks(attacks_list):
	return ", ".join([attacks['description'] for attacks in attacks_list])

def do_attacks(img, attacks_list):
	for attack in attacks_list:
		attack['arguments']['img'] = img
		img = attack['function'](**attack['arguments'])
	return img, describe_attacks(attacks_list)

def get_attacks_list(attacks):
	attacks_list_out = []
	for attack in attacks:
		if attack == "gaus blur":
			sigma = [randint(1, 5), randint(1, 5)]
			attacks_list_out.append(
				{
					'function': gaussian_blur,
					'arguments': {
						"sigma": sigma
					},
					'description': 'Gaussian Blur ({})'.format(sigma)
				}
			)

		if attack == "avg blur":
			avg_blur_kernel_size = (randint(1, 3) * 2) + 1  # 3, 5, 7
			attacks_list_out.append(
				{
					'function': average_blur,
					'arguments': {
						"kernel": avg_blur_kernel_size
					},
					'description': 'Average Blur ({})'.format(avg_blur_kernel_size)
				}
			)

		elif attack == "jpeg":
			jpeg_quality_factor = randint(1, 10) * 10  # 10, 20, ..., 100
			attacks_list_out.append(
				{
					'function': jpeg_compression,
					'arguments': {
						"QF": jpeg_quality_factor
					},
					'description': 'JPEG ({})'.format(jpeg_quality_factor)
				}
			)

		elif attack == "sharpen":
			sharpen_sigma = (random() * (5 - 0.2)) + 0.2
			sharpen_alpha = random() * (5 - 0.1) + 0.1

			attacks_list_out.append(
				{
					'function': sharpen,
					'arguments': {
						"sigma": sharpen_sigma,
						"alpha": sharpen_alpha
					},
					'description': 'Sharpen ({}, {})'.format(sharpen_sigma, sharpen_alpha)
				}
			)

		elif attack == "awgn":
			awgn_mean = randint(-5, 5)
			awgn_std_dev = (random() * (5 - 0.2)) + 0.2
			awgn_seed = randint(0, 1000)
			attacks_list_out.append(
				{
					'function': awgn,
					'arguments': {
						"mean": awgn_mean,
						"std": awgn_std_dev,
						"seed": awgn_seed
					},
					'description': 'Additive White Gaussian Noise ({}, {}, {})'.format(awgn_mean, awgn_std_dev,
																					   awgn_seed)
				}
			)

		elif attack == "resize":
			resize_scale = randint(1, 9) / 10  # 0.1, 0.2, ..., 0.9
			attacks_list_out.append(
				{
					'function': resize,
					'arguments': {
						"scale": resize_scale
					},
					'description': 'Resize ({})'.format(resize_scale)
				}
			)

		elif attack == "median":
			median_kernel_size = (randint(1, 3) * 2) + 1  # 3, 5, 7
			attacks_list_out.append(
				{
					'function': median,
					'arguments': {
						"kernel_size": median_kernel_size
					},
					'description': 'Median ({})'.format(median_kernel_size)
				}
			)
	return attacks_list_out