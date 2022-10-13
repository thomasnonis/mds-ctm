import os
import sys
from time import time

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.fft import dct, idct
from scipy.signal import convolve2d
from math import sqrt
import pywt

sys.path.append("..")
from attacks import *
from measurements import *
from transforms import *
from embedding import *
from tools import *

start_time = time()
print('Starting...')

if not os.path.isdir('images/'):
	os.mkdir('images/')

if not os.path.isfile('images/lena.bmp'):  
	os.system('python -m wget "https://drive.google.com/uc?export=download&id=17MVOgbOEOqeOwZCzXsIJMDzW5Dw9-8zm" -o images/lena.bmp')

if not os.path.isfile('images/lena.bmp'):
	sys.exit('Failed downlading images')

# Read the image
img_path = 'images/lena.bmp'
original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Embed a watermark
level = 3
original_coeffs = wavedec2d(original_img,level)
original_LL = original_coeffs[0]

mark_size = 32
watermark = generate_watermark(1024).reshape((mark_size, mark_size)) # So that it is a square

alpha = 10

watermarked_LL = original_LL.copy()

for x in range(0,watermark.shape[0]):
	for y in range(0,watermark.shape[1]):
		watermarked_LL[x][y] += alpha * watermark[x][y]

original_coeffs[0] = watermarked_LL

watermarked_img = waverec2d(original_coeffs)

watermarked_img = jpeg_compression(watermarked_img, 50)
# watermarked_img = median(watermarked_img, 5)
# watermarked_img = resize(watermarked_img, 0.5)
# watermarked_img = gaussian_blur(watermarked_img, 1)

extracted_watermark = detection_dwt(original_img, watermarked_img, alpha, level, mark_size)

# TODO: Implement threshold learning with a set of images and a set of attacks
sim = similarity(watermark, extracted_watermark)

print('SIM: %.4f' % sim)

print('Total elapsed time: %.2f[s]' % (time() - start_time))

show_images([(original_img, "Original image"), (watermarked_img,"Watermarked image"), (watermarked_img - original_img, "Difference between images"), (watermark, "Watermark"), (extracted_watermark, "Extracted watermark"), (extracted_watermark - watermark, "Difference between watermarks")], 3, 3)