import pyswarms as ps
import multiprocessing
import cv2 as cv
from random import randint
import time
from datetime import datetime
import uuid
import os
import csv
import importlib
from others import import_others_detection

from attacks import gaussian_blur, average_blur, sharpen, median, resizing, awgn, jpeg_compression
from detection_failedfouriertransform import detection
from tools import show_images
from config import *

ATTACKED_TEAM_NAME = 'failedfouriertransform'
ATTACKED_IMG_NAME = 'lena'

mod = import_others_detection(ATTACKED_TEAM_NAME)

N_ITERATIONS = 10
N_PARTICLES = 30
N_SETS = 1
N_DIMENSIONS = 15 * N_SETS
N_PARALLEL_PROCESSES = 12
PENALTY = 1000
ENABLE_VERBOSE = True

ENABLE_GAUSSIAN_BLUR = 		[True, True, True]
ENABLE_AVERAGE_BLUR = 		[True, True, True]
ENABLE_SHARPEN = 			[True, True, True]
ENABLE_MEDIAN = 			[True, True, True]
ENABLE_RESIZING = 			[True, True, True]
ENABLE_AWGN = 				[True, True, True]
ENABLE_JPEG_COMPRESSION = 	[True, True, True]

MIN_BOUND_GAUSSIAN_BLUR_SIGMA = 0.1
MIN_BOUND_AVERAGE_BLUR_KERNEL_SIZE = 3
MIN_BOUND_SHARPEN_SIGMA = 0.1
MIN_BOUND_SHARPEN_ALPHA = 0.1
MIN_BOUND_MEDIAN_KERNEL_SIZE = 3
MIN_BOUND_RESIZING_SCALE = 0.2
MIN_BOUND_AWGN_STD_DEV = 0.1
MIN_BOUND_JPEG_QF = 0

MAX_BOUND_GAUSSIAN_BLUR_SIGMA = 5
MAX_BOUND_AVERAGE_BLUR_KERNEL_SIZE = 5
MAX_BOUND_SHARPEN_SIGMA = 10
MAX_BOUND_SHARPEN_ALPHA = 10
MAX_BOUND_MEDIAN_KERNEL_SIZE = 8
MAX_BOUND_RESIZING_SCALE = 1
MAX_BOUND_AWGN_STD_DEV = 50
MAX_BOUND_JPEG_QF = 101

def perform_attacks_args(attacked_img, args, penalty, i, set=0):
	offset = 15 * set
	# First set of attacks
	do_gaussian_blur =            args[0 + offset]
	do_average_blur =             args[1 + offset]
	do_sharpen =                  args[2 + offset]
	do_median =                   args[3 + offset]
	do_resizing =                 args[4 + offset]
	do_awgn =                     args[5 + offset]
	do_jpeg_compression =         args[6 + offset]

	gaussian_blur_sigma =         args[7 + offset]
	average_blur_kernel_size =    args[8 + offset]
	sharpen_sigma =               args[9 + offset]
	sharpen_alpha =               args[10 + offset]
	median_kernel_size =          args[11 + offset]
	resizing_scale =              args[12 + offset]
	awgn_std_dev =                args[13 + offset]
	jpeg_compression_qf =         args[14 + offset]
	
	if do_gaussian_blur[i] > 0.5:
		if ENABLE_GAUSSIAN_BLUR[set]:
			try:
				attacked_img = gaussian_blur(attacked_img, gaussian_blur_sigma[i])
			except Exception as e:
				penalty += PENALTY
				print('An error occurred during gaussian_blur({}) and a penalty has been applied: {}'.format(gaussian_blur_sigma[i], e))
		else:
			penalty += PENALTY

	if do_average_blur[i] > 0.5:
		if ENABLE_AVERAGE_BLUR[set]:
			if int(average_blur_kernel_size[i]) not in [3, 5, 7]:
				penalty += PENALTY
			else:
				attacked_img = average_blur(attacked_img, int(average_blur_kernel_size[i]))
		else:
			penalty += PENALTY

	if do_sharpen[i] > 0.5:
		if ENABLE_SHARPEN[set]:
			try:
				attacked_img = sharpen(attacked_img, sharpen_sigma[i], sharpen_alpha[i])
			except Exception as e:
				penalty += PENALTY
				print('An error occurred during sharpen({}, {}) and a penalty has been applied: {}'.format(sharpen_sigma[i], sharpen_alpha[i], e))
		else:
			penalty += PENALTY

	if do_median[i] > 0.5:
		if ENABLE_MEDIAN[set]:
			if int(median_kernel_size[i]) not in [3, 5, 7]:
				penalty += PENALTY
			else:
				attacked_img = median(attacked_img, int(median_kernel_size[i]))
		else:
			penalty += PENALTY

	if do_resizing[i] > 0.5:
		if ENABLE_RESIZING[set]:
			try:
				attacked_img = resizing(attacked_img, resizing_scale[i])
			except Exception as e:
				penalty += PENALTY
				print('An error occurred during resizing({}) and a penalty has been applied: {}'.format(resizing_scale[i], e))
		else:
			penalty += PENALTY

	if do_awgn[i] > 0.5:
		if ENABLE_AWGN[set]:
			try:
				seed = randint(0, 1000)
				attacked_img = awgn(attacked_img, awgn_std_dev[i], seed)
			except Exception as e:
				penalty += PENALTY
				print('An error occurred during awgn({}, {}) and a penalty has been applied: {}'.format(awgn_std_dev[i], seed, e))
		else:
			penalty += PENALTY

	if do_jpeg_compression[i] > 0.5:
		if ENABLE_JPEG_COMPRESSION[set]:
			try:
				attacked_img = jpeg_compression(attacked_img, int(jpeg_compression_qf[i]))
			except Exception as e:
				penalty += PENALTY
				print('An error occurred during jpeg_compression({}) and a penalty has been applied: {}'.format(int(jpeg_compression_qf[i]), e))
		else:
			penalty += PENALTY

	return attacked_img, penalty

def objective_function(args, **kwargs):
	args = args.T
	original_img_path = kwargs['original_image_path']
	watermarked_img_path = kwargs['watermarked_image_path']
	tmp_folder_path = kwargs['tmp_folder_path']

	watermarked_img = cv.imread(watermarked_img_path, cv.IMREAD_GRAYSCALE)

	ret = []
	for i in range(args.shape[1]):
		# print(args.T[i])
		penalty = 0
		attacked_img = watermarked_img.copy()

		for set in range(N_SETS):
			attacked_img, penalty = perform_attacks_args(attacked_img, args, penalty, i, set)

		tmp_attacked_img_path = tmp_folder_path + str(uuid.uuid4()) + '.bmp'
		cv.imwrite(tmp_attacked_img_path, attacked_img)

		# Let program wait up to 500ms in case write is not finished
		attempts = 5
		while not os.path.exists(tmp_attacked_img_path) and attempts > 0:
			print('Waiting for the file {}.bmp to be created...'.format(tmp_attacked_img_path))
			time.sleep(0.1)
			attempts -= 1
			
		# External detection function
		has_watermark, wpsnr = mod.detection(original_img_path, watermarked_img_path, tmp_attacked_img_path)
		# has_watermark, wpsnr = detection(original_img_path, watermarked_img_path, tmp_attacked_img_path)
		if has_watermark == 1:
			penalty += PENALTY
		
		if penalty == 0:
			ret.append(100 / abs(wpsnr))
		else:
			ret.append(penalty)

		attempts = 10
		while os.path.exists(tmp_attacked_img_path) and attempts > 0:
			attempts -= 1
			try:
				os.remove(tmp_attacked_img_path)
			except:
				print(f"Error while trying to remove {tmp_attacked_img_path}, attempts left: {attempts}")

	return ret

def print_results(cost, pos):
	print('========== RESULTS ==========')
	print('WPSNR: ', 100/cost)
	for i in range(N_SETS):
		offset = i * 15
		print('=== SET {} ==='.format(i + 1))
		print('DO GAUSSIAN BLUR {}: '.format(i + 1),		'True' if pos[0 + offset] > 0.5 else ('False' if ENABLE_GAUSSIAN_BLUR[i] else 'Disabled'))
		print('DO AVERAGE BLUR {}: '.format(i + 1),			'True' if pos[1 + offset] > 0.5 else ('False' if ENABLE_AVERAGE_BLUR[i] else 'Disabled'))
		print('DO SHARPEN {}: '.format(i + 1),				'True' if pos[2 + offset] > 0.5 else ('False' if ENABLE_SHARPEN[i] else 'Disabled'))
		print('DO MEDIAN {}: '.format(i + 1),				'True' if pos[3 + offset] > 0.5 else ('False' if ENABLE_MEDIAN[i] else 'Disabled'))
		print('DO RESIZING {}: '.format(i + 1),				'True' if pos[4 + offset] > 0.5 else ('False' if ENABLE_RESIZING[i] else 'Disabled'))
		print('DO AWGN {}: '.format(i + 1),					'True' if pos[5 + offset] > 0.5 else ('False' if ENABLE_AWGN[i] else 'Disabled'))
		print('DO JPEG COMPRESSION: {}: '.format(i + 1),	'True' if pos[6 + offset] > 0.5 else ('False' if ENABLE_JPEG_COMPRESSION[i] else 'Disabled'))
		if pos[0 + offset] > 0.5:
			print('GAUSSIAN BLUR {} SIGMA: '.format(i + 1), pos[7 + offset])

		if pos[1 + offset] > 0.5:
			print('AVERAGE BLUR {} KERNEL SIZE: '.format(i + 1), int(pos[8 + offset]))

		if pos[2 + offset] > 0.5:
			print('SHARPEN {} SIGMA: '.format(i + 1), pos[9 + offset])
			print('SHARPEN {} ALPHA: '.format(i + 1), pos[10 + offset])
		
		if pos[3 + offset] > 0.5:
			print('MEDIAN {} KERNEL SIZE: '.format(i + 1), int(pos[11 + offset]))

		if pos[4 + offset] > 0.5:
			print('RESIZING {} SCALE: '.format(i + 1), pos[12 + offset])
		
		if pos[5 + offset] > 0.5:
			print('AWGN {} STD DEV: '.format(i + 1), pos[13 + offset])

		if pos[6 + offset] > 0.5:
			print('JPEG {} COMPRESSION QF: '.format(i + 1), int(pos[14 + offset]))

def log_csv(filename, img_name, cost, pos):
	attacks_string = ''
	for i in range(N_SETS):
		offset = i * 15
		if pos[0 + offset] > 0.5:
			attacks_string += 'GAUSSIAN BLUR ({}), '.format(pos[7 + offset])

		if pos[1 + offset] > 0.5:
			attacks_string +=  'AVERAGE BLUR ({}), '.format(int(pos[8 + offset]))

		if pos[2 + offset] > 0.5:
			attacks_string += 'SHARPEN ({}, {}), '.format(pos[9 + offset], pos[10 + offset])
		
		if pos[3 + offset] > 0.5:
			attacks_string += 'MEDIAN ({}), '.format(int(pos[11 + offset]))

		if pos[4 + offset] > 0.5:
			attacks_string += 'RESIZING ({}), '.format(pos[12 + offset])
		
		if pos[5 + offset] > 0.5:
			attacks_string += 'AWGN ({}), '.format(pos[13 + offset])

		if pos[6 + offset] > 0.5:
			attacks_string += 'JPEG ({}), '.format(int(pos[14 + offset]))

	now = datetime.now()
	data = [now.strftime('%H:%M:%S'), img_name, 100/cost, attacks_string[:-2]]
	
	if os.path.exists(filename):
		with open(filename, 'a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(data)
	else:
		with open(filename, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['Time', 'Image', 'WPSNR', 'Attacks'])
			writer.writerow(data)

def run_best_attack(attacked_img, args):
	for i in range(N_SETS):
		offset = i * 15
		do_gaussian_blur = args[0 + offset]
		do_average_blur = args[1 + offset]
		do_sharpen = args[2 + offset]
		do_median = args[3 + offset]
		do_resizing = args[4 + offset]
		do_awgn = args[5 + offset]
		do_jpeg_compression = args[6 + offset]

		gaussian_blur_sigma = args[7 + offset]
		average_blur_kernel_size = args[8 + offset]
		sharpen_sigma = args[9 + offset]
		sharpen_alpha = args[10 + offset]
		median_kernel_size = args[11 + offset]
		resizing_scale = args[12 + offset]
		awgn_std_dev = args[13 + offset]
		jpeg_compression_qf = args[14 + offset]

		if do_gaussian_blur > 0.5:
			attacked_img = gaussian_blur(attacked_img, gaussian_blur_sigma)

		if do_average_blur > 0.5:
			attacked_img = average_blur(attacked_img, int(average_blur_kernel_size))

		if do_sharpen > 0.5:
			attacked_img = sharpen(attacked_img, sharpen_sigma, sharpen_alpha)

		if do_median > 0.5:
			attacked_img = median(attacked_img, int(median_kernel_size))

		if do_resizing > 0.5:
			attacked_img = resizing(attacked_img, resizing_scale)

		if do_awgn > 0.5:
			attacked_img = awgn(attacked_img, awgn_std_dev, randint(0, 1000))

		if do_jpeg_compression > 0.5:
			attacked_img = jpeg_compression(attacked_img, int(jpeg_compression_qf))

	return attacked_img

if __name__ == '__main__':
	start_time = time.time()
	original_img_path = 'images/' + ATTACKED_TEAM_NAME + '/original/' + ATTACKED_IMG_NAME + '.bmp'
	watermarked_img_path = 'images/' + ATTACKED_TEAM_NAME + '/watermarked/' + ATTACKED_TEAM_NAME + '_' + ATTACKED_IMG_NAME + '.bmp'
	attacked_img_path = 'images/' + ATTACKED_TEAM_NAME + '/attacked/' + TEAM_NAME + '_' + ATTACKED_TEAM_NAME + '_' + ATTACKED_IMG_NAME + '.bmp'

	# Set-up hyperparameters
	#'c1': 0.5, 'c2': 0.3, 'w':0.9
	c1 = 1.1
	c2 = 0.7
	w = 0.9
	assert (w > -1 and w < 1 and (c1 + c2) < ((24*(1 - (w * w)))/(7 - (5 * w)))), 'Invalid PSO options. The algorithm will not converge.'
	options = {'c1': c1, 'c2': c2, 'w':w}

	min_bounds = [0, 0, 0, 0, 0, 0, 0, MIN_BOUND_GAUSSIAN_BLUR_SIGMA, MIN_BOUND_AVERAGE_BLUR_KERNEL_SIZE, MIN_BOUND_SHARPEN_SIGMA, MIN_BOUND_SHARPEN_ALPHA, MIN_BOUND_MEDIAN_KERNEL_SIZE, MIN_BOUND_RESIZING_SCALE, MIN_BOUND_AWGN_STD_DEV, MIN_BOUND_JPEG_QF, 0, 0, 0, 0, 0, 0, 0, MIN_BOUND_GAUSSIAN_BLUR_SIGMA, MIN_BOUND_AVERAGE_BLUR_KERNEL_SIZE, MIN_BOUND_SHARPEN_SIGMA, MIN_BOUND_SHARPEN_ALPHA, MIN_BOUND_MEDIAN_KERNEL_SIZE, MIN_BOUND_RESIZING_SCALE, MIN_BOUND_AWGN_STD_DEV, MIN_BOUND_JPEG_QF, 0, 0, 0, 0, 0, 0, 0, MIN_BOUND_GAUSSIAN_BLUR_SIGMA, MIN_BOUND_AVERAGE_BLUR_KERNEL_SIZE, MIN_BOUND_SHARPEN_SIGMA, MIN_BOUND_SHARPEN_ALPHA, MIN_BOUND_MEDIAN_KERNEL_SIZE, MIN_BOUND_RESIZING_SCALE, MIN_BOUND_AWGN_STD_DEV, MIN_BOUND_JPEG_QF]
	max_bounds = [1, 1, 1, 1, 1, 1, 1, MAX_BOUND_GAUSSIAN_BLUR_SIGMA, MAX_BOUND_AVERAGE_BLUR_KERNEL_SIZE, MAX_BOUND_SHARPEN_SIGMA, MAX_BOUND_SHARPEN_ALPHA, MAX_BOUND_MEDIAN_KERNEL_SIZE, MAX_BOUND_RESIZING_SCALE, MAX_BOUND_AWGN_STD_DEV, MAX_BOUND_JPEG_QF, 1, 1, 1, 1, 1, 1, 1, MAX_BOUND_GAUSSIAN_BLUR_SIGMA, MAX_BOUND_AVERAGE_BLUR_KERNEL_SIZE, MAX_BOUND_SHARPEN_SIGMA, MAX_BOUND_SHARPEN_ALPHA, MAX_BOUND_MEDIAN_KERNEL_SIZE, MAX_BOUND_RESIZING_SCALE, MAX_BOUND_AWGN_STD_DEV, MAX_BOUND_JPEG_QF, 1, 1, 1, 1, 1, 1, 1, MAX_BOUND_GAUSSIAN_BLUR_SIGMA, MAX_BOUND_AVERAGE_BLUR_KERNEL_SIZE, MAX_BOUND_SHARPEN_SIGMA, MAX_BOUND_SHARPEN_ALPHA, MAX_BOUND_MEDIAN_KERNEL_SIZE, MAX_BOUND_RESIZING_SCALE, MAX_BOUND_AWGN_STD_DEV, MAX_BOUND_JPEG_QF]
	bounds = (min_bounds[:N_DIMENSIONS], max_bounds[:N_DIMENSIONS])

	if not os.path.exists(TMP_FOLDER_PATH):
		os.makedirs(TMP_FOLDER_PATH)

	print('Running optimization with {} iterations, {} particles and {} set{} of attacks'.format(N_ITERATIONS, N_PARTICLES, N_SETS, 's' if N_SETS > 1 else ''))
	# Call instance of PSO
	optimizer = ps.single.GlobalBestPSO(n_particles=N_PARTICLES, dimensions=N_DIMENSIONS, options=options, bounds=bounds)
	# Perform optimization
	cost, pos = optimizer.optimize(objective_function, iters=N_ITERATIONS, n_processes=min(multiprocessing.cpu_count(), N_PARALLEL_PROCESSES), verbose=ENABLE_VERBOSE, original_image_path=original_img_path, watermarked_image_path=watermarked_img_path, tmp_folder_path=TMP_FOLDER_PATH)


	print_results(cost, pos)
	log_csv('attacks_log.csv', original_img_path, cost, pos)

	end_time = time.time()
	
	print('========== RUNNING BEST ATTACK ==========')

	watermarked_img = cv.imread(watermarked_img_path, cv.IMREAD_GRAYSCALE)
	attacked_img = run_best_attack(watermarked_img, pos)
	cv.imwrite(attacked_img_path, attacked_img)

	# External detection function
	has_watermark, wpsnr = mod.detection(original_img_path, watermarked_img_path, attacked_img_path)
	# has_watermark, wpsnr = detection(original_img_path, watermarked_img_path, attacked_img_path)

	print('WPSNR: ', wpsnr)
	print('Has watermark: ', has_watermark)

	attempts = 10
	while(os.path.exists(TMP_FOLDER_PATH)) and attempts > 0:
		attempts -= 1
		try:
			os.rmdir(TMP_FOLDER_PATH)
		except:
			print(f"Error while trying to remove {TMP_FOLDER_PATH}, attempts left: {attempts}")
	
	exec_time = end_time - start_time
	if exec_time < 60:
		print('Execution time: {} seconds'.format(round(exec_time, 1)))
	else:
		mins = exec_time // 60
		print('Execution time: {}:{} minutes'.format(int(mins), int(exec_time - (mins * 60))))

	# show_images([(original_img, 'Original Image'), (watermarked_img, 'Watermarked Image'), (attacked_img, 'Attacked Image')], 1, 3)