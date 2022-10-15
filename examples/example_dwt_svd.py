import os
import sys
from time import time
from cv2 import blur

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
level = 1
original_coeffs = wavedec2d(original_img,level)
original_LL = original_coeffs[0] 

mark_size = 32
watermark = generate_watermark(1024).reshape((mark_size, mark_size)) # So that it is a square
original_u_ll, original_s_ll, original_v_ll = np.linalg.svd(original_LL)

original_s_ll_d = np.diag(original_s_ll.copy())

alpha = 300
for x in range(0,watermark.shape[0]):
	for y in range(0,watermark.shape[1]):
		original_s_ll_d[x][y] += alpha * watermark[x][y]

# Step 4
original_s_ll_d_u, original_s_ll_d_s, original_s_ll_d_v = np.linalg.svd(original_s_ll_d)
original_s_ll_d_s_diag = np.diag(original_s_ll_d_s.copy())

LL_svd = np.matmul(np.matmul(original_u_ll,original_s_ll_d_s_diag),original_v_ll)
print(LL_svd.shape)
original_coeffs[0] = LL_svd

watermarked_img = waverec2d(original_coeffs)

extracted = detection_dwt_svd(original_img, watermarked_img, alpha,level,mark_size, original_s_ll_d_u, original_s_ll_d_v)

attacked = average_blur(watermarked_img,5)
attacked = sharpen(attacked,0.1,3)
attacked = jpeg_compression(attacked, 5)

extracted_watermark = detection_dwt_svd(original_img, attacked, alpha,level,mark_size, original_s_ll_d_u, original_s_ll_d_v)

print('WPSNR original_img-watermarked_img: %.2f[dB]' % wpsnr(original_img, watermarked_img))
print('WPSNR original_img-attacked: %.2f[dB]' % wpsnr(original_img, attacked))
print('WPSNR watermarked_img-attacked: %.2f[dB]' % wpsnr(watermarked_img, attacked))

sim = similarity(watermark, extracted_watermark)

print('SIM: %.4f' % sim)

#show_images([(original_img, "Original image"), (watermarked_img,"Watermarked image"), (watermarked_img - original_img, "Difference between images"), (watermark, "Watermark"), (extracted, "Extracted watermak"), (attacked, "Attacked"), (extracted_watermark, "Watermark attacked")], 3, 3)


show_images([(extracted, "Extracted watermak"), (attacked, "Attacked"), (extracted_watermark, "Watermark attacked")], 1, 3)