from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt2d
import matplotlib as mpl
from skimage.transform import rescale, resize as skimage_resize
import numpy as np
import cv2
from PIL import Image
import os
from random import randint, random

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

def bilateral_blur():
	pass

def sharpen(img, sigma, alpha):
	blurred = gaussian_filter(img, sigma)
	return img + alpha * (img - blurred)

def median(img, kernel_size):
	return medfilt2d(img, kernel_size)

def resize(img, scale):
	x, y = img.shape
	attacked = rescale(img, scale)
	attacked = skimage_resize(attacked, (x,y))
	attacked = attacked[:x, :y]
	return attacked

def awgn(img, mean, std, seed):
	np.random.seed(seed)
	attacked = img + np.random.normal(mean, std, img.shape)
	attacked = np.clip(attacked, 0, 255)
	return attacked

def jpeg_compression(img, QF):
	img = Image.fromarray(img)
	img = img.convert('L')
	img.save('tmp.jpg',"JPEG", quality=QF)
	attacked = Image.open('tmp.jpg')
	attacked = np.asarray(attacked,dtype=np.uint8)
	os.remove('tmp.jpg')
	return attacked


def get_random_attacks(num_attacks):
	attacks_list = []

	# TODO: randomize parameters in a meaningful way
	for _ in range(0, num_attacks):
		attack = randint(0, N_AVAILABLE_ATTACKS - 1)
		if attack == 0:
			awgn_mean = randint(-5, 5)
			awgn_std_dev = (random() * (5 - 0.2)) + 0.2
			awgn_seed = randint(0, 1000)
			attacks_list.append(
				{
					'function' : awgn,
					'arguments' : {
						"mean" : awgn_mean, 
						"std":  awgn_std_dev,
						"seed": awgn_seed
					},
					'description' : 'Additive White Gaussian Noise ({}, {}, {})'.format(awgn_mean, awgn_std_dev, awgn_seed)
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
			sharpen_sigma = (random() * (5 - 0.2)) + 0.2
			sharpen_alpha = random() * (5 - 0.1) + 0.1

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
			jpeg_quality_factor = randint(1, 10) * 10 # 10, 20, ..., 100
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
			resize_scale = randint(1, 9) / 10 # 0.1, 0.2, ..., 0.9
			attacks_list.append(
				{
					'function' : resize,
					'arguments' : {
						"scale" : resize_scale
					},
					'description' : 'Resize ({})'.format(resize_scale)
				}
			)
		elif attack == 5:
			median_kernel_size = (randint(1, 3) * 2) + 1 # 3, 5, 7
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
			sigma = [randint(1, 5),randint(1, 5)]
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

def do_random_attacks(img,attacks_list):
	for attack in attacks_list:
		attack['arguments']['img'] = img
		img = attack['function'](**attack['arguments'])
	return img, describe_attacks(attacks_list)