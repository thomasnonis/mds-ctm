import numpy as np
import os
from scipy.signal import convolve2d
from math import sqrt
import cv2

def wpsnr(img1, img2):
	if not os.path.isfile('csf.csv'):  
		os.system('python -m wget "https://drive.google.com/uc?export=download&id=1w43k1BTfrWm6X0rqAOQhrbX6JDhIKTRW" -o csf.csv')

	img1 = np.float32(img1)/255.0
	img2 = np.float32(img2)/255.0

	difference = img1 - img2
	same = not np.any(difference)
	if same is True:
		return 9999999
	
	csf = np.genfromtxt('csf.csv', delimiter=',')
	ew = convolve2d(difference, np.rot90(csf,2), mode='valid')
	decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2)))) # this is something that can be optimized by using numerical values instead of db
	return decibels

def psnr(img1, img2):
	return cv2.PSNR(img1, img2)

def similarity(img1,img2):
	#Computes the similarity measure between the original and the new watermarks.
	return np.sum(np.multiply(img1, img2)) / np.sqrt(np.sum(np.multiply(img2, img2)))

def compute_thr(sim, mark_size, w):
    SIM = np.zeros(1000)
    SIM[0] = abs(sim)
    for i in range(1, 1000):
        r = np.random.uniform(0.0, 1.0, mark_size)
        SIM[i] = abs(similarity(w, r))
    
    SIM.sort()
    t = SIM[-2]
    T = t + (0.1*t)
    return T

def find_mark(mark, extracted_watermark, mark_size):
	sim = similarity(mark, extracted_watermark)
	T = compute_thr(sim, mark_size, mark)
	if sim > T:
		print('Mark has been found. SIM = %f' % sim)
	else:
		print('Mark has been lost. SIM = %f' % sim)

def local_variance(img, i, j, window_size):
	from math import floor
	# compute the local variance of the image within a square window of window_size centered at position (i,j)

	i_min = i-floor(window_size/2)
	i_max = i+floor(window_size/2) + 1
	j_min = j-floor(window_size/2)
	j_max = j+floor(window_size/2) + 1

	if i_min < 0:
		i_min = 0
	elif i_max > img.shape[0]:
		i_max = img.shape[0]

	if j_min < 0:	
		j_min = 0
	elif j_max > img.shape[1]:
		j_max = img.shape[1]

	mean = np.mean(img[i_min:i_max, j_min:j_max])

	variance = 0
	for x in range(i-floor(window_size/2), i+floor(window_size/2)):
		for y in range(j-floor(window_size/2), j+floor(window_size/2)):
			if x >= 0 and x < img.shape[0] and y >= 0 and y < img.shape[1]:
				variance += (img[x, y] - mean)**2

	variance = variance / ((window_size*window_size) - 1)
	return variance

def nvf(img, D, window_size):
	max_variance = 0

	variance = np.zeros([img.shape[0], img.shape[1]])

	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			variance[i,j] = local_variance(img, i, j, window_size)
			if variance[i,j] > max_variance:
				max_variance = variance[i,j]

	theta = D/max_variance

	return 1/(1+theta*variance)