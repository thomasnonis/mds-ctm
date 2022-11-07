from time import time
from datetime import datetime

from config import *
from attacks import *
from tools import *
from detection_failedfouriertransform import extract_watermark, similarity, wpsnr
from embedment_failedfouriertransform import embed_watermark
from roc_failedfouriertransform import read_model

def evaluate_model(params, order_of_execution):
	original_img, watermarked_image_name, watermark, attacks, watermarked_image, alpha, level, subband = params
	
	model_name = '_'.join(watermarked_image_name.split('_')[1:])
	print(model_name)
	_, _, threshold, tpr, _ = read_model(model_name)
		
	successful = 0
	wpsnr_tot = 0
	
	model_stats = {}
	
	print("Threshold: ", threshold)
	for attack in attacks:
		attacked_img, _ = do_attacks(watermarked_image,attack)
		extracted_watermark = None
		extracted_watermark = extract_watermark(original_img, attacked_img, alpha, level, subband)
		
		sim = similarity(watermark, extracted_watermark)
		_wpsnr = wpsnr(original_img, attacked_img)
		if sim < threshold and  _wpsnr > 35:
			successful += 1	
			wpsnr_tot += _wpsnr

			print("The following attack (%s) was succesfull with a similarity of %f and a wpsnr of %f" % (describe_attacks(attack), sim, _wpsnr))
	model_stats['Succesful Attacks'] =  successful
	model_stats['Unsuccesful Attacks'] =  RUNS_PER_IMAGE - successful
	model_stats['WPSNR Original-Watermarked'] = wpsnr(original_img, watermarked_image)
	
	score = 0
	if successful > 0:
		score += attacked_wpsnr_to_mark(wpsnr_tot / successful)
	else:
		score += 6 + 2

	score += wpsnr_to_mark(model_stats['WPSNR Original-Watermarked'])
	model_stats['Score'] = score * tpr

	return order_of_execution, model_name, model_stats

def multiproc_embed_watermark(params, order_of_execution):
	original_img, img_name, watermark, alpha, level, subband = params
	watermarked_img = embed_watermark(original_img, watermark, alpha, level, subband)
	
	return order_of_execution, original_img, img_name, alpha, level, subband, watermarked_img

def main():
	# Get N_IMAGES_LIMIT random images
	images = import_images('images/test/original/',N_IMAGES_LIMIT,True)

	# Read watermark
	watermark = np.load("failedfouriertransform.npy").reshape((MARK_SIZE, MARK_SIZE))
	
	attacks = []
	# Get list of attacks
	for _ in range(0, RUNS_PER_IMAGE):
		attacks.append(get_random_attacks(randint(1, MAX_N_ATTACKS)))
	
	watermarked_images = {}
	print('Welcome to multiprocessing city')
	print('Embedding...')
	work = []
	for image in images:
		original_img, img_name = image
		for alpha in range(18,25):
			for level in [2]:
				for subband in [["LL"]]:
						work.append((original_img, img_name, watermark, alpha, level, subband))
	
	results = multiprocessed_workload(multiproc_embed_watermark,work)

	for result in results:
		original_img, img_name, alpha, level, subband, watermarked_img = result
		params = '_'.join([str(alpha),str(level),'-'.join(subband)])
		watermarked_images[img_name + '_' + params] = (original_img, img_name, watermarked_img, alpha, level, subband)

	print("Let the Hunger Games begin!")
	
	work = []

	for watermarked_image_name in watermarked_images:
		original_img, img_name, watermarked_image, alpha, level, subband  = watermarked_images[watermarked_image_name]
		work.append((original_img, watermarked_image_name, watermark, attacks, watermarked_image, alpha, level, subband))
	
	results = multiprocessed_workload(evaluate_model,work)
	all_models = {}
	# Merge results from models on N_IMAGES_LIMIT different images
	# Let's say we've run model x on two different images, we want to know the average WPSNR that image had with certain embedding method
	# For example the average how many attacks the image with the watermark embedded a certain way did survive and so on
	for model in results:
		model_name, model = model
		if model_name in all_models:
			all_models[model_name]['AVG WPSNR Original-Watermarked'] += round((model['WPSNR Original-Watermarked'] / N_IMAGES_LIMIT),2)
			all_models[model_name]['AVG Score'] += round((model['Score'] / N_IMAGES_LIMIT),2)
			all_models[model_name]['AVG Succesful Attacks'] += round((model['Succesful Attacks'] / N_IMAGES_LIMIT),2)
			all_models[model_name]['AVG Unsuccesful Attacks'] += round((model['Unsuccesful Attacks'] / N_IMAGES_LIMIT),2)
		else:
			new_model = {}
			new_model['AVG WPSNR Original-Watermarked'] = round((model['WPSNR Original-Watermarked'] / N_IMAGES_LIMIT),2)
			new_model['AVG Score'] = round((model['Score'] / N_IMAGES_LIMIT),2)
			new_model['AVG Succesful Attacks'] = round((model['Succesful Attacks'] / N_IMAGES_LIMIT),2)
			new_model['AVG Unsuccesful Attacks'] = round((model['Unsuccesful Attacks'] / N_IMAGES_LIMIT),2)
			all_models[model_name] = new_model
	print("Models sorted by score:")
	lst_models = [(model_name, all_models[model_name]) for model_name in all_models]
	lst_models.sort(key=lambda x: x[1]['AVG Score'], reverse=True)
	for model_name, model in lst_models:
		print(model_name.ljust(10), model)
	
	best_model_name, _ = lst_models[0]
	_, _, threshold, _, _ = read_model(best_model_name)
	alpha, level, subband = best_model_name.split('_')
	subband = subband.split('-')
	update_parameters('detection_failedfouriertransform.py', ALPHA = alpha, DWT_LEVEL = level, SUBBANDS = subband, DETECTION_THRESHOLD = threshold, MARK_SIZE = MARK_SIZE)

	for image in images:
		original_img, img_name = image
		print(img_name, best_model_name)
		print(watermarked_images[img_name + '_' + best_model_name])
		_, _, watermarked_img, _, _, _ = watermarked_images[img_name + '_' + best_model_name]
		save_image(watermarked_img, img_name, "watermarked", TEAM_NAME)


if __name__ == '__main__':
	st = time()
	main()
	et = time()
	print(et-st)
