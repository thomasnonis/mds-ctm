from time import time
from datetime import datetime

from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *
from detection_failedfouriertransform import *

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
	
	alpha_range = np.arange(0.1, 0.4, 0.1) * DEFAULT_ALPHA
	for alpha in alpha_range:
		alpha = int(alpha)
		for level in [DWT_LEVEL-1,DWT_LEVEL,DWT_LEVEL+1]:
			for subband in [["LL"], ["HL","LH"]]:
				watermarked_img = embed_watermark(original_img, img_name, watermark, alpha, level, subband)
				params = "_".join([str(alpha),str(level),"-".join(subband)])
				watermarked_images[img_name + '_' + params] = (watermarked_img,extract_watermark)
	
	alpha_range = np.arange(0.5, 0.8, 0.2) * ALPHA_TN
	beta_range = np.arange(0.01, BETA+0.1, 0.04)
	for alpha in alpha_range:
		alpha = round(alpha,2)
		for beta in beta_range:
			beta = round(beta,2)
			watermarked_img = embed_watermark_tn(original_img, img_name, watermark, alpha, beta)
			params = "_".join([str(round(alpha,2)),str(round(beta,2))])
			watermarked_images[img_name + '_' + params] = (watermarked_img,extract_watermark_tn)
			

	print("Let the Hunger Games begin!")
	model_stats = {}
	best_watermarked_image = {}
	max_score = 0
	for watermarked_image_name in watermarked_images:
		model_name = '_'.join(watermarked_image_name.split('_')[1:])
		model_stats[model_name] = {}
		print(model_name)
		scores, labels, threshold, tpr, fpr, params = read_model(model_name)
		
		successful = 0
		wpsnr_tot = 0
		sim_tot = 0
		watermarked_image, extraction_function = watermarked_images[watermarked_image_name]
		for attack in attacks:
			attacked_img, _ = do_random_attacks(watermarked_image,attack)
			extracted_watermark = None
			if extraction_function == extract_watermark:
				alpha,level,subband = params
				extracted_watermark = extract_watermark(original_img, img_name, attacked_img, alpha, level, subband)
			elif extraction_function == extract_watermark_tn:
				alpha,beta = params
				extracted_watermark = extract_watermark_tn(original_img, img_name, watermarked_image, alpha, beta)
			else:
				print(f'Extraction function {extraction_function} does not exist!')
			sim = similarity(watermark, extracted_watermark)
			_wpsnr = wpsnr(original_img, attacked_img)
			if sim < threshold and  _wpsnr > 35:
				successful += 1	
				wpsnr_tot += _wpsnr
				sim_tot += sim
				print("The following attack (%s) was succesfull with a similarity of %f and a wpsnr of %f" % (describe_attacks(attack), sim, _wpsnr))
		model_stats[model_name]['Succesful Attacks'] =  successful
		model_stats[model_name]['Unsuccesful Attacks'] =  RUNS_PER_IMAGE - successful
		model_stats[model_name]['WPSNR Original-Watermarked'] = wpsnr(original_img, watermarked_image)
		if successful > 0:
			model_stats[model_name]['AVG WPSNR Succesful Attacks'] = wpsnr_tot / successful
			model_stats[model_name]['AVG SIM Succesful Attacks'] = sim_tot / successful
		else:
			model_stats[model_name]['AVG WPSNR Succesful Attacks'] = 0
			model_stats[model_name]['AVG SIM Succesful Attacks'] = 0
		
		# TODO: Find a better way of calculating the score
		score = model_stats[model_name]['WPSNR Original-Watermarked'] * 10 + model_stats[model_name]['AVG WPSNR Succesful Attacks'] * model_stats[model_name]['AVG SIM Succesful Attacks'] - 75 * successful
		model_stats[model_name]['Score'] = score
		if score > max_score:
			max_score = score
			best_watermarked_image['Watermarked Image'] = watermarked_image
			best_watermarked_image['Technique'] = model_name

		print(model_stats[model_name])
		print('='*15)
	print(model_stats)
	print('Best technique was', best_watermarked_image['Technique'])
	print(model_stats[model_name])
	cv2.imwrite(img_name + '_' + best_watermarked_image['Technique'] + '.jpg', best_watermarked_image['Watermarked Image'])
if __name__ == '__main__':
	st = time()
	main()
	et = time()
	print(et-st)
