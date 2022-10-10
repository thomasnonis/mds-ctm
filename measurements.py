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