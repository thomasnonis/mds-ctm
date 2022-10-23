import numpy as np
import os
from scipy.signal import convolve2d
from math import sqrt
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from random import randint

from attacks import do_random_attacks, get_random_attacks
from tools import *
from config import *

def wpsnr(img1: np.ndarray, img2: np.ndarray):
	if not os.path.isfile('csf.csv'):
		os.system('python -m wget "https://drive.google.com/uc?export=download&id=1w43k1BTfrWm6X0rqAOQhrbX6JDhIKTRW" -o csf.csv')
		print("Ok")

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

# Check if correct
def compute_ROC(scores, labels, show: bool = True):
	# compute ROC
	fpr, tpr, thr = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
	# compute AUC
	roc_auc = auc(fpr, tpr)
	plt.figure()
	lw = 2

	plt.plot(fpr, tpr, color='darkorange',
			 lw=lw, label='AUC = %0.2f' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([-0.01, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC)')
	plt.legend(loc="lower right")
	idx_tpr = np.where((fpr - TARGET_FPR) == min(i for i in (fpr - TARGET_FPR) if i > 0))
	print('For a FPR approximately equals to %0.2f corresponds a TPR equal to %0.2f and a threshold equal to %0.4f with FPR equal to %0.2f' % (TARGET_FPR, tpr[idx_tpr[0][0]], thr[idx_tpr[0][0]], fpr[idx_tpr[0][0]]))
	if show is True:
		plt.show()
	return thr[idx_tpr[0][0]], tpr[idx_tpr[0][0]], fpr[idx_tpr[0][0]] # return thr

def compute_thr_multiple_images(images, original_watermark, alpha, level, subband, attacks, show: bool = True):
	scores = []
	labels = []
	n_images = len(images)
	i = 0
	m = 0
	attack_idx = 0
	n_computations = n_images * RUNS_PER_IMAGE * N_FALSE_WATERMARKS_GENERATIONS
	print('Total number of computations: %d' % n_computations)
	
	# step by step for clarity
	for original_img, watermarked_img, img_name in images:
		for j in range(attack_idx, attack_idx+RUNS_PER_IMAGE):
			attacked_img, attacks_list = do_random_attacks(watermarked_img, attacks[j])
			extracted_watermark = extract_watermark(original_img, img_name, attacked_img, alpha, level, subband)

			# true positive population
			scores.append(similarity(original_watermark, extracted_watermark))
			labels.append(1)			

			# perform multiple comparisons with random watermarks to better train the classifier against false positives
			# TODO: verify that this actually works
			for k in range(0, N_FALSE_WATERMARKS_GENERATIONS):
				print('{}/{} - Performed attack {}/{} on image {}/{} ({}) - false check {}/{} - attacks: {}'.format(m + 1, n_computations, j%RUNS_PER_IMAGE, RUNS_PER_IMAGE, i + 1, n_images, img_name, k + 1, N_FALSE_WATERMARKS_GENERATIONS, attacks_list))
				# true negative population
				scores.append(similarity(generate_watermark(MARK_SIZE), extracted_watermark))
				labels.append(0)
				m += 1
		i += 1
		attack_idx += RUNS_PER_IMAGE
	return scores,labels,compute_ROC(scores, labels, show)
