from time import time
from datetime import datetime

from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *


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

	alpha_range = np.arange(0.1, 1, 0.1) * DEFAULT_ALPHA
	for alpha in alpha_range:
		alpha = int(alpha)
		for level in [DWT_LEVEL-1,DWT_LEVEL,DWT_LEVEL+1]:
			for subband in [["LL"]]:
				watermarked_img = embed_watermark(original_img, img_name, watermark, alpha, level, subband)
				params = "_".join([str(alpha),str(level),"-".join(subband)])
				watermarked_images[img_name + '_' + params] = watermarked_img
	
	print("Let the Hunger Games begin!")
	model_stats = {}
	best_watermarked_image = {}
	max_score = 0
	for watermarked_image_name in watermarked_images:
		model_name = '_'.join(watermarked_image_name.split('_')[1:])
		model_stats[model_name] = {}
		print(model_name)
		scores, labels, threshold, tpr, fpr, alpha, level, subband = read_model(model_name)
		successful = 0
		wpsnr_tot = 0
		sim_tot = 0
		for attack in attacks:
			attacked_img, _ = do_random_attacks(watermarked_images[watermarked_image_name],attack)
			extracted_watermark = extract_watermark(original_img, img_name, attacked_img, alpha, level, subband)
			sim = similarity(watermark, extracted_watermark)
			_wpsnr = wpsnr(original_img, attacked_img)
			if sim < threshold and  _wpsnr > 35:
				successful += 1	
				wpsnr_tot += _wpsnr
				sim_tot += sim
				print("The following attack (%s) was succesfull with a similarity of %f and a wpsnr of %f" % (describe_attacks(attack), sim, _wpsnr))
		model_stats[model_name]['Succesful Attacks'] =  successful
		model_stats[model_name]['Unsuccesful Attacks'] =  RUNS_PER_IMAGE - successful
		model_stats[model_name]['WPSNR Original-Watermarked'] = wpsnr(original_img, watermarked_images[watermarked_image_name])
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
			best_watermarked_image['Watermarked Image'] = watermarked_images[watermarked_image_name]
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
