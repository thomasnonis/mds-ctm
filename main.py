# import attacks
# import measurements

import os
import sys
from time import time

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.fft import dct, idct
from scipy.signal import convolve2d
from math import sqrt

from attacks import *
from measurements import *
from transforms import *

start_time = time()
print('Starting...')

if not os.path.isdir('images/'):
	os.mkdir('images/')

if not os.path.isfile('images/lena.bmp'):  
	os.system('python -m wget "https://drive.google.com/uc?export=download&id=17MVOgbOEOqeOwZCzXsIJMDzW5Dw9-8zm" -o images/lena.bmp')

if not os.path.isfile('images/lena.bmp'):
	sys.exit('Failed downlading images')

img_path = 'images/lena.bmp'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

blurred = gaussian_blur(img, 2)

print('WPSNR: %.2f[dB]' % wpsnr(img, blurred))

plt.imshow(blurred, cmap='gray')
plt.show()

print('Total elapsed time: %.2f[s]' % (time() - start_time))
