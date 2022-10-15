import os
import numpy as np
import cv2
import pickle

def wpsnr_to_mark(wpsnr):
	if wpsnr >= 35 and wpsnr < 50:
		return 1
	if wpsnr >= 50 and wpsnr < 54:
		return 2
	if wpsnr >= 54 and wpsnr < 58:
		return 3
	if wpsnr >= 58 and wpsnr < 62:
		return 4
	if wpsnr >= 62 and wpsnr < 66:
		return 5
	if wpsnr >= 66:
		return 6
	return 0

def generate_watermark(mark_size):
	import numpy as np
	# Generate a watermark
	mark = np.random.uniform(0.0, 1.0, mark_size)
	mark = np.uint8(np.rint(mark))
	np.save('mark.npy', mark)
	return mark

# list_of_images: [(watermarked,"Watermarked"),(attacked,"Attacked"),...]
def show_images(list_of_images, rows, columns):
	import matplotlib.pyplot as plt
	
	plt.figure(figsize=(15, 6))
	for (i,(image,label)) in enumerate(list_of_images):

		plt.subplot(rows,columns,i+1)
		plt.title(list_of_images[i][1])
		plt.imshow(list_of_images[i][0], cmap='gray')
	plt.show()


def embed_into_svd(img, watermark, alpha):
	(svd_u, svd_s, svd_v) = np.linalg.svd(img)

	# Convert S from a 1D vector to a 2D diagonal matrix
	svd_s = np.diag(svd_s)

	# Embed the watermark in the SVD matrix
	for x in range(0, watermark.shape[0]):
		for y in range(0, watermark.shape[1]):
			svd_s[x][y] += alpha * watermark[x][y]

	(svd_s_u, svd_s_s, svd_s_v) = np.linalg.svd(svd_s)

	# Convert S from a 1D vector to a 2D diagonal matrix
	svd_s_s = np.diag(svd_s_s)

	# Recompose matrices from SVD decomposition
	watermarked = svd_u @ svd_s_s @ svd_v
	# key = svd_s_u @ svd_s @ svd_s_v

	return (watermarked, (svd_s_u, svd_s_v))

def save_parameters(img_name, alpha, svd_key):
	if not os.path.isdir('parameters/'):
		os.mkdir('parameters/')
	f = open('parameters/' + img_name + '_parameters.txt', 'wb')
	pickle.dump((img_name, alpha, svd_key), f, protocol=2)
	f.close()

def read_parameters(img_name):
	f = open('parameters/' + img_name + '_parameters.txt', 'rb')
	(img_name, alpha, svd_key) = pickle.load(f)
	f.close()
	return img_name, float(alpha), svd_key

def import_images(img_folder_path):
	if not os.path.isdir(img_folder_path):
		exit('Error: Images folder not found')
	
	images = []
	for img_filename in os.listdir(img_folder_path):
		# (img, name)
		images.append((cv2.imread(img_folder_path + img_filename, cv2.IMREAD_GRAYSCALE), img_filename.split('.')[-2]))

	n_images = len(images)
	print('Loaded', n_images, 'image' + ('s' if n_images > 1 else '') + ':')
	
	return images

def append_attack_to_list(attacks_list, attack_string):
	if attacks_list != '':
		attacks_list += ', '
	return attacks_list + attack_string