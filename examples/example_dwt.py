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
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Embed a watermark
level = 3
coeffs = wavedec2d(image,level)
LL3 = coeffs[0]

mark_size = 32
mark = generate_watermark(1024).reshape((mark_size, mark_size)) # So that it is a square

w_coeffs = wavedec2d(mark,level)
wLL3 = w_coeffs[0]

alpha = 10

newLL3 = LL3.copy()
print(newLL3.shape, wLL3.shape)
for x in range(0,wLL3.shape[0]):
	for y in range(0,wLL3.shape[1]):
		newLL3[x][y] += alpha * wLL3[x][y]

coeffs[0] = newLL3
watermarked = waverec2d(coeffs)

show_images([(watermarked,"Watermarked")], 1, 1)
print(mark)
extracted = detection_dwt(image, watermarked, alpha, level, mark_size)

print(extracted)

print(mark==extracted) # They should be equal!

print(mark.shape, extracted.shape)
print('Total elapsed time: %.2f[s]' % (time() - start_time))
