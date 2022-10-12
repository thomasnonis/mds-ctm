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

mark_size = 1024
alpha = 24 #0.1
v = 'additive'
mark = generate_watermark(mark_size)

watermarked = embedding_dct(image, mark, alpha, v)

show_images([(image,"Original"),(watermarked,"Watermarked")], 1, 2)

# Do some attack
attacked = gaussian_blur(watermarked, 1)

print('WPSNR of Watermarked image: %.2f[dB]' % wpsnr(image, watermarked))
print('WPSNR of Attacked image: %.2f[dB]' % wpsnr(image, attacked))

show_images([(watermarked,"Watermarked"),(attacked,"Watermarked Attacked")], 1, 2)

# Extract the watermark
wat_ex = detection(image, watermarked, alpha, mark_size, v)

wat_att_ex = detection(image, attacked, alpha, mark_size, v)

find_mark(mark, wat_ex, mark_size)
find_mark(mark, wat_att_ex, mark_size)


print('Total elapsed time: %.2f[s]' % (time() - start_time))
