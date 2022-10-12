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

coeffs = pywt.dwt2(image, 'haar')
LL, (LH, HL, HH) = coeffs

coeffs2 = pywt.dwt2(LL, 'haar')
LL2, (LH2, HL2, HH2) = coeffs2

coeffs3 = pywt.dwt2(LL2, 'haar')
LL3, (LH3, HL3, HH3) = coeffs3

#u,s,vh = np.linalg.svd(LL3)

#print(s)
#print(len(s))

mark_size = 32
mark = generate_watermark(1024).reshape((mark_size, mark_size)) # So that it is a square
w_coeffs = pywt.dwt2(mark, 'haar')
wLL, (wLH, wHL, wHH) = w_coeffs

w_coeffs2 = pywt.dwt2(wLL, 'haar')
wLL2, (wLH2, wHL2, wHH2) = w_coeffs2

w_coeffs3 = pywt.dwt2(wLL2, 'haar')
wLL3, (wLH3, wHL3, wHH3) = w_coeffs3

#w_u,w_s,w_vh = np.linalg.svd(wLL3)


alpha = 10

newLL3 = LL3.copy()

for x in range(0,wLL3.shape[0]):
	for y in range(0,wLL3.shape[1]):
		newLL3[x][y] += alpha * wLL3[x][y]

#LL3_prime = u * s_prime * vh.transpose()
newLL2 = pywt.idwt2((newLL3, (LH3, HL3, HH3)), 'haar')
newLL = pywt.idwt2((newLL2, (LH2, HL2, HH2)), 'haar')
watermarked = pywt.idwt2((newLL, (LH, HL, HH)), 'haar')

show_images([(watermarked,"Watermarked")], 1, 1)

print(mark)
extracted = detection_dwt(image, watermarked, alpha, (mark_size,mark_size))

print(extracted)
print(mark==extracted) # They should be equal!

print('Total elapsed time: %.2f[s]' % (time() - start_time))