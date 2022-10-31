from time import time
from datetime import datetime

from config import *
from attacks import *
from tools import *
from detection_failedfouriertransform import extract_watermark, similarity, wpsnr
from embedment_failedfouriertransform import embed_watermark
from roc_failedfouriertransform import read_model

def evaluate_model(params, order_of_execution):
	original_img, img_name, watermarked_image_name, watermark, attacks, watermarked_image, alpha, level, subband = params
	
	model_name = '_'.join(watermarked_image_name.split('_')[1:])
	print(model_name)
	_, _, threshold, _, _ = read_model(model_name)
		
	successful = 0
	wpsnr_tot = 0
	sim_tot = 0
	model_stats = {}
	
	print("Threshold: ", threshold)
	for attack in attacks:
		attacked_img, _ = do_attacks(watermarked_image,attack)
		extracted_watermark = None
		extracted_watermark = extract_watermark(original_img, img_name, attacked_img, alpha, level, subband)
		
		sim = similarity(watermark, extracted_watermark)
		_wpsnr = wpsnr(original_img, attacked_img)
		if sim < threshold and  _wpsnr > 35:
			successful += 1	
			wpsnr_tot += _wpsnr
			sim_tot += sim
			print("The following attack (%s) was succesfull with a similarity of %f and a wpsnr of %f" % (describe_attacks(attack), sim, _wpsnr))
	model_stats['Succesful Attacks'] =  successful
	model_stats['Unsuccesful Attacks'] =  RUNS_PER_IMAGE - successful
	model_stats['WPSNR Original-Watermarked'] = wpsnr(original_img, watermarked_image)
	if successful > 0:
		model_stats['AVG WPSNR Succesful Attacks'] = wpsnr_tot / successful
		model_stats['AVG SIM Succesful Attacks'] = sim_tot / successful
	else:
		model_stats['AVG WPSNR Succesful Attacks'] = 0
		model_stats['AVG SIM Succesful Attacks'] = 0
	
	# TODO: Find a better way of calculating the score
	score = model_stats['WPSNR Original-Watermarked'] * 10 + model_stats['AVG WPSNR Succesful Attacks'] * model_stats['AVG SIM Succesful Attacks'] + 40 * (RUNS_PER_IMAGE - successful)
	if model_stats['Succesful Attacks'] == 0:
		score += 500
	model_stats['Score'] = score

	return order_of_execution, model_name, model_stats

def multiproc_embed_watermark(params, order_of_execution):
	original_img, img_name, watermark, alpha, level, subband = params
	watermarked_img = embed_watermark(original_img, img_name, watermark, alpha, level, subband)
	
	return order_of_execution, original_img, img_name, alpha, level, subband, watermarked_img

def main():
	# Get one random image
	N_IMAGES_LIMIT = 10
	images = import_images(IMG_FOLDER_PATH,N_IMAGES_LIMIT,True)

	# Generate watermark
	watermark = np.load("failedfouriertransform.npy").reshape((MARK_SIZE, MARK_SIZE))
	
	attacks = []
	# Get list of attacks
	for _ in range(0, RUNS_PER_IMAGE):
		attacks.append(get_random_attacks(randint(1, MAX_N_ATTACKS)))
	
	watermarked_images = {}
	print('Welcome to multiprocessing city')
	print('Embedding...')
	work = []
	alpha_range = [25, 50, 75, 100, 150, 250]
	for image in images:
		original_img, img_name = image
		for alpha in alpha_range:
			for level in [DWT_LEVEL - 3, DWT_LEVEL - 2, DWT_LEVEL - 1, DWT_LEVEL ]:
				for subband in [["LL"], ["HL", "LH"]]:
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
		work.append((original_img, img_name, watermarked_image_name, watermark, attacks, watermarked_image, alpha, level, subband))
	
	results = multiprocessed_workload(evaluate_model,work)
	print(results)
	max_score = 0
	best_technique = ''

	all_models = {}
	for model in results:
		model_name, model = model
		if model_name in all_models:
			all_models[model_name]['WPSNR Original-Watermarked'] += (model['WPSNR Original-Watermarked'] / N_IMAGES_LIMIT)
			all_models[model_name]['AVG WPSNR Succesful Attacks'] += (model['AVG WPSNR Succesful Attacks'] / N_IMAGES_LIMIT)
			all_models[model_name]['AVG SIM Succesful Attacks'] += (model['AVG SIM Succesful Attacks'] / N_IMAGES_LIMIT)
			all_models[model_name]['Score'] += model['Score']
			all_models[model_name]['Succesful Attacks'] += model['Succesful Attacks']
			all_models[model_name]['Unsuccesful Attacks'] += model['Unsuccesful Attacks']
		else:
			model['WPSNR Original-Watermarked'] /= N_IMAGES_LIMIT   # Average out the WPSNRs for the same model on different images
			model['AVG WPSNR Succesful Attacks'] /= N_IMAGES_LIMIT  # Average out the WPSNRs for the same model on different images
			model['AVG SIM Succesful Attacks'] /= N_IMAGES_LIMIT    # Average out the SIM for the same model on different images
			all_models[model_name] = model
	best_model = {}
	for model_name in all_models:
		model = all_models[model_name]
		if model['Score'] > max_score:
			best_model = model
			max_score = model['Score']
			best_technique = model_name
	print("="*10)
	print(all_models)
	
	
	print('Best technique was',best_technique, 'with a score of', best_model['Score'] )
	print(best_model)
	for image in images:
		original_img, img_name = image
		_, _, watermarked_img, _, _, _ = watermarked_images[img_name + '_' + best_technique]
		cv2.imwrite(img_name + '_' + best_technique + '.bmp', watermarked_img)

if __name__ == '__main__':
	st = time()
	main()
	et = time()
	print(et-st)
