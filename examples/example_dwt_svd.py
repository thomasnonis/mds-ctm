import sys
from time import time
from collections import defaultdict

sys.path.append('..')
from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *

start_time = time()
print('Starting...')

# Load images
n_images = min(1,N_IMAGES_LIMIT)
images = import_images('../'+IMG_FOLDER_PATH, n_images, True)
watermark = generate_watermark(MARK_SIZE)




subbands = [['LL'],['HL','LH']]
levels = [DWT_LEVEL-1, DWT_LEVEL, DWT_LEVEL+1]
alpha_range = np.arange(0.1, 0.3, 0.1) * DEFAULT_ALPHA
watermarked_imgs = defaultdict(dict)
print("Total work: ", n_images * len(subbands) * len(alpha_range) * len(levels))
for original_img, img_name in images:
	watermarked_imgs[img_name]['original_img'] = original_img
	watermarked_imgs[img_name]['watermarked'] = {}
	for level in levels:
		watermarked_imgs[img_name]['watermarked'][level] = {}
		for subband in subbands:
			watermarked_imgs[img_name]['watermarked'][level]['-'.join(subband)] = {}
			d = {}
			for alpha in alpha_range:
				alpha = int(alpha)
				watermarked_img = embed_watermark(original_img, img_name, watermark, alpha, level, subband)
				d[alpha] = watermarked_img
			watermarked_imgs[img_name]['watermarked'][level]['-'.join(subband)] = d
plt.figure()
for img in watermarked_imgs:
	original_img = watermarked_imgs[img]['original_img']

	for level in watermarked_imgs[img]['watermarked']:
		for subband in watermarked_imgs[img]['watermarked'][level]:
			ys = []
			for alpha in watermarked_imgs[img]['watermarked'][level][subband]:
				watermarked_image = watermarked_imgs[img]['watermarked'][level][subband][alpha]
				ys.append(wpsnr(original_img,watermarked_image))
			plt.plot(alpha_range, ys, lw=2, label=img+'_'+subband+'_'+str(level))

plt.xlabel('Alpha')
plt.ylabel('WPSNR')
plt.title('WPSNR comparison of images with a watermark embedded and with different values of alpha and different subbands')
plt.legend(loc='upper right')
plt.show()

plt.figure()
attacks_list = get_random_attacks(1)

print(describe_attacks(attacks_list))
for img in watermarked_imgs:
	original_img = watermarked_imgs[img]['original_img']

	for level in watermarked_imgs[img]['watermarked']:
		for subband in subbands:
			xs = []
			ys = []
			for alpha in watermarked_imgs[img]['watermarked'][level]['-'.join(subband)]:
				watermarked_image = watermarked_imgs[img]['watermarked'][level]['-'.join(subband)][alpha]
				attacked_img, _ = do_random_attacks(watermarked_image,attacks_list)
				extracted_watermark = extract_watermark(original_img, img_name, attacked_img,level,subband)
				xs.append(wpsnr(original_img,attacked_img))
				ys.append(similarity(watermark, extracted_watermark)) # Sometimes we get RuntimeWarning: invalid value encountered in double_scalars for some unkown reason
			plt.plot(xs, ys, lw=2, label=img+'_'+'-'.join(subband)+'_'+str(level))

plt.xlabel('WPSNR')
plt.ylabel('Similarity')
plt.title('WPSNR-Similarity of attacked images at different levels ')
plt.legend(loc='upper right')
plt.show()

'''
print(results)
# Compute threshold
img_folder_path = '../sample-images/'
images = import_images(img_folder_path)
paths = os.listdir(img_folder_path)

w_images = []
for i in range(len(images)):
	#print(images[i][0])
	w_img = embed_watermark(images[i][0], paths[i], watermark, DEFAULT_ALPHA)
	w_images.append(images[i])

t = compute_thr_multiple_images(w_images, watermark, '../images/', True)
print("Threshold: ", t)
'''