from time import time
from datetime import datetime

from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *


def evaluate_model(params, order_of_execution):
	original_img = params[0]
	img_name = params[1]
	watermarked_image_name = params[2]
	watermark = params[3]
	attacks = params[4]
	watermarked_image = params[5]
	extraction_function = params[6]
	model_name = '_'.join(watermarked_image_name.split('_')[1:])
	print(model_name)
	scores, labels, threshold, tpr, fpr, params = read_model(model_name)
		
	successful = 0
	wpsnr_tot = 0
	sim_tot = 0
	model_stats = {}
	#watermarked_image, extraction_function = watermarked_images[watermarked_image_name]
	print("Threshold: ", threshold)
	for attack in attacks:
		attacked_img, _ = do_random_attacks(watermarked_image,attack)
		extracted_watermark = None
		if extraction_function == extract_watermark:
			alpha,level,subband = params
			extracted_watermark = extract_watermark(original_img, img_name, attacked_img, alpha, level, subband)
		elif extraction_function == extract_watermark_tn:
			alpha,beta = params
			extracted_watermark = extract_watermark_tn(original_img, img_name, attacked_img, alpha, beta)
		else:
			print(f'Extraction function {extraction_function} does not exist!')
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

def main():
	# Get one random image
	original_img, img_name = import_images(IMG_FOLDER_PATH,1,True)[0]

	# Generate watermark
	watermark = generate_watermark(MARK_SIZE)

	show_threshold = False
	attacks = []
	# Get list of attacks
	for _ in range(0, RUNS_PER_IMAGE):
		attacks.append(get_random_attacks(randint(1, MAX_N_ATTACKS)))
	
	watermarked_images = {}
	print('Embedding...')
	alpha_range = [25,50,75,100]
	for alpha in alpha_range:
		alpha = int(alpha)
		for level in [DWT_LEVEL-1,DWT_LEVEL,DWT_LEVEL+1]:
			for subband in [["LL"], ["HL","LH"]]:
				watermarked_img = embed_watermark(original_img, img_name, watermark, alpha, level, subband)
				params = "_".join([str(alpha),str(level),"-".join(subband)])
				watermarked_images[img_name + '_' + params] = (watermarked_img,extract_watermark)
	
	alpha_range = [10,20,40,60]
	beta_range = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6]

	for alpha in alpha_range:
		for beta in beta_range:
			watermarked_img = embed_watermark_tn(original_img, img_name, watermark, alpha, beta)
			params = "_".join([str(alpha),str(beta)])
			watermarked_images[img_name + '_' + params] = (watermarked_img,extract_watermark_tn)
	

	print("Let the Hunger Games begin!")
	
	work = []
	for watermarked_image_name in watermarked_images:
		watermarked_image,extraction_function  = watermarked_images[watermarked_image_name]
		work.append((original_img, img_name,watermarked_image_name,watermark, attacks,watermarked_image,extraction_function))
	
	result = multiprocessed_workload(evaluate_model,work)

	max_score = 0
	best_technique = ''
	best_model = {}
	for model in result:
		model_name, model = model
		if model['Score'] > max_score:
			best_model = model
			max_score = model['Score']
			best_technique = model_name
			best_watermarked_image = watermarked_images[img_name + '_' + model_name][0]

	print(result)
	
	
	print('Best technique was',best_technique, 'with a score of', best_model['Score'] )
	print(best_model)
	cv2.imwrite(img_name + '_' + best_technique + '.jpg', best_watermarked_image)

if __name__ == '__main__':
	st = time()
	main()
	et = time()
	print(et-st)
