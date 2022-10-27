from difflib import diff_bytes
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from random import randint

from attacks import do_attacks
from tools import *
from config import *
from detection_failedfouriertransform import *

def psnr(img1, img2):
	return cv2.PSNR(img1, img2)

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

def compute_thr_multiple_images(extraction_function, images, original_watermark, params, attacks, show: bool = True):
	scores = []
	labels = []
	n_images = len(images)
	i = 0
	m = 0
	attack_idx = 0
	n_computations = n_images * RUNS_PER_IMAGE * N_FALSE_WATERMARKS_GENERATIONS
	print('Total number of computations: %d' % n_computations)

	# Continue training if a model with the same parameters already existed
	model_name = []
	for x in params:
		if type(x) == list:
			model_name.append('-'.join(x))
		else:
			model_name.append(str(x))

	model_name = '_'.join(model_name)
	if exists_model(model_name):
		(scores, labels, _, _, _, _) = read_model(model_name)
	
	# step by step for clarity
	for original_img, watermarked_img, img_name in images:
		for j in range(attack_idx, attack_idx+RUNS_PER_IMAGE):
			attacked_img, attacks_list = do_attacks(watermarked_img, attacks[j])
			extracted_watermark = None
			if extraction_function == extract_watermark:
				alpha = params[0]
				level = params[1]
				subband = params[2]
				extracted_watermark = extract_watermark(original_img, img_name, attacked_img, alpha, level, subband)
			elif extraction_function == extract_watermark_tn:
				alpha = params[0]
				beta = params[1]
				extracted_watermark = extract_watermark_tn(original_img, img_name, attacked_img, alpha, beta)
			else:
				print(f'Extraction function {extraction_function} does not exist!')

			# true positive population
			scores.append(similarity(original_watermark, extracted_watermark))
			labels.append(1)			

			# perform multiple comparisons with random watermarks to better train the classifier against false positives
			# TODO: verify that this actually works
			for k in range(0, N_FALSE_WATERMARKS_GENERATIONS):
				# true negative population
				scores.append(similarity(generate_watermark(MARK_SIZE), extracted_watermark))
				print('{}/{} - Performed attack {}/{} on image {}/{} ({}) - false check {}/{} - True sim: {} - False sim: {} - attacks: {}'.format(m + 1, n_computations, j + 1, RUNS_PER_IMAGE, i + 1, n_images, img_name, k + 1, N_FALSE_WATERMARKS_GENERATIONS, scores[-2], scores[-1], attacks_list))
				labels.append(0)
				m += 1
		i += 1
		attack_idx += RUNS_PER_IMAGE
	return scores,labels,compute_ROC(scores, labels, show)
