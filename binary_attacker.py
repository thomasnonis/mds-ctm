from config import *
from attacks import *
from others import *
from tools import multiprocessed_workload, show_images
from itertools import combinations

from random import randint, random
import os

ATTACKS_MUTATIONS = 30

def import_images(img_folder_path: str, num_images: int, shuffle: bool = False) -> list:
	"""Loads a list of all images contained in a folder and returns a list of (image, name) tuples
	Args:
		img_folder_path (str): Relative path to the folder containing the images (e.g. 'images/')
	Returns:
		list: List of (image, name) tuples
	"""
	if not os.path.isdir(img_folder_path):
		exit('Error: Images folder not found')
	num_images = min(num_images, len(os.listdir(img_folder_path)))
	images = []
	paths = os.listdir(img_folder_path)
	if shuffle:
		random.shuffle(paths)
	for img_filename in paths[:num_images]:
		# (image, name)
		images.append((cv2.imread(img_folder_path + img_filename, cv2.IMREAD_GRAYSCALE), img_folder_path + img_filename))

	print('Loaded', num_images, 'image' + ('s' if num_images > 1 else ''))

	return images


def reduce_attack_strength(idx, attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, old_wpsnr):
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	
	attack = attack_list[idx]
	min_index = index_list[idx]	
	max_index = len(attack_parameters[attack['function']]) - 1
	# Binary search best parameter
	while min_index != max_index:
		attack = attack_list[idx]
		min_index = index_list[idx]
		index = (min_index + max_index - 1) // 2
		
		new_params = attack_parameters[attack['function']][index]
		new_params_dict = parse_parameters(attack['function'], new_params)
		backup_attack = attack_list[idx]['arguments'].copy()
		index_list[idx] = index
		# Update the attack parameters with a weaker attack
		attack_list[idx]['arguments'] = new_params_dict
		
		attacked_img, attack_description = do_attacks(watermarked_img, attack_list)
		tmp_attacked_img_path = str(uuid.uuid4()) + ".bmp"
		cv2.imwrite(tmp_attacked_img_path, attacked_img)
		has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, tmp_attacked_img_path)
		# print("NEW Attack",attack_description, _wpsnr)
		if has_watermark == 0 and _wpsnr > old_wpsnr:
			min_index = index
			old_wpsnr = _wpsnr
			os.remove(attacked_img_path)
			os.rename(tmp_attacked_img_path, attacked_img_path)
		else:
			os.remove(tmp_attacked_img_path)
			index_list[idx] = min_index
			attack_list[idx]['arguments'] = backup_attack
			max_index = index
			

	return attack_list, index_list
def remove_attack(idx, attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, old_wpsnr):
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	
	index_list_copy = index_list.copy()
	attack_list_copy = attack_list.copy()
	# Remove an attack
	index_list_copy = index_list_copy[:idx] + index_list_copy[idx+1:]
	attack_list_copy = attack_list_copy[:idx] + attack_list_copy[idx+1:]
	#print(f"Substituting {get_attack_description(attack_list[idx])} with nothing")
	
	attacked_img, attack_description = do_attacks(watermarked_img, attack_list_copy)
	tmp_attacked_img_path = str(uuid.uuid1()) + ".bmp"
	cv2.imwrite(tmp_attacked_img_path, attacked_img)
	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, tmp_attacked_img_path)
	
	# print("NEW Attack",attack_description, _wpsnr)
	if has_watermark == 0 and _wpsnr > old_wpsnr:
		attack_list, index_list = attack_list_copy, index_list_copy
		os.remove(attacked_img_path)
		os.rename(tmp_attacked_img_path, attacked_img_path)
	else:
		os.remove(tmp_attacked_img_path)
		#print("Attempt to do substituion failed!")

	return attack_list, index_list
def swap_attack(idx, attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, old_wpsnr):
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	
	index_list_copy = index_list.copy()
	attack_list_copy = attack_list.copy()

	new_attack, new_index = get_indexed_random_attacks(1)
	index_list_copy[idx] = new_index[0]
	attack_list_copy[idx] = new_attack[0]
	#print(f"Substituting {get_attack_description(attack_list[idx])} with {get_attack_description(attack_list_copy[idx])}")
	#print(f'Trying to reduce {get_attack_description(attack_list_copy[idx])}')
	attack_list_copy, index_list_copy = reduce_attack_strength(idx,attack_list_copy, index_list_copy, detection_function, original_img_path, watermarked_img_path, attacked_img_path, old_wpsnr)
	
	
	attacked_img, attack_description = do_attacks(watermarked_img, attack_list_copy)
	tmp_attacked_img_path = str(uuid.uuid1()) + ".bmp"
	cv2.imwrite(tmp_attacked_img_path, attacked_img)
	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, tmp_attacked_img_path)
	#print("NEW Attack",attack_description, _wpsnr)
	os.remove(tmp_attacked_img_path)
	if has_watermark == 0 and _wpsnr > old_wpsnr:	
		attack_list, index_list = attack_list_copy, index_list_copy
	else:
		pass
		#print("Attempt to do substituion failed!")
	
	return attack_list, index_list

def mutate_attack(params, order_of_execution):
	original_img_path, watermarked_img_path, attacked_img_path,attack_list, index_list, _wpsnr, detection_function = params#valid_attacks[original_img_path]
	print("OLD Attack",describe_attacks(attack_list), _wpsnr)
	for _ in range(0,ATTACKS_MUTATIONS):
		strategy = randint(0,2)
		if strategy == 0:
			# Decrease the strength of an attack in the attack list
			idx = randint(0,len(attack_list)-1)
			#print(f'Trying to reduce {get_attack_description(attack_list[idx])}')
			attack_list, index_list = reduce_attack_strength(idx,attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, _wpsnr)
		elif strategy == 1:
			if len(attack_list) > 1:
				# Swap an attack from the attack list with another attack, or removes it
				idx = randint(0,len(attack_list)-1)
				#print(f'Trying to remove {get_attack_description(attack_list[idx])}')
				attack_list, index_list = remove_attack(idx,attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, _wpsnr)
		elif strategy == 2:
			# Swap an attack from the attack list with another attack
			idx = randint(0,len(attack_list)-1)
			#print(f'Trying to swap {get_attack_description(attack_list[idx])}')
			attack_list, index_list = swap_attack(idx,attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, _wpsnr)
	
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	attacked_img, attack_description = do_attacks(watermarked_img, attack_list)
	cv2.imwrite(attacked_img_path, attacked_img)
	print("Best attack was: ", describe_attacks(attack_list))
	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
	print("Has watermark?", has_watermark)
	print("WPSNR:", _wpsnr)

	
	
	locations = localize_attack(original_img_path, watermarked_img_path, attacked_img_path, detection_function)
	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
	

	result = {
		'Has watermark' : has_watermark,
		'Attack' : describe_attacks(attack_list),
		'WPSNR' : _wpsnr,
		'attacked_img_path': attacked_img_path,
		'locations' : locations # AAAAAAAAAAAAAAAAAAAAAAA Rimoovimiiii
	}
	return order_of_execution, result

def rect_localize_attack(original_img_path, watermarked_img_path, attacked_img_path, detection_function):
	attacked_img = cv2.imread(attacked_img_path, cv2.IMREAD_GRAYSCALE)
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	#show_images([(attacked_img, 'attacked'),(watermarked_img, 'watermarked'),(watermarked_img - attacked_img, 'diff')],1,3)
	min_x = 0
	max_x = watermarked_img.shape[0]
	min_y = 0
	max_y = watermarked_img.shape[1]

	
	print("Begin reducing x")
	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
	print(has_watermark, _wpsnr)
	idx_x = (min_x + max_x - 1) // 2
	while min_x != idx_x:
		attacked_img = cv2.imread(attacked_img_path, cv2.IMREAD_GRAYSCALE)
		watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
		attacked_img[min_x:idx_x] = watermarked_img[min_x:idx_x]
		tmp_attacked_img_path = str(uuid.uuid4()) + ".bmp"
		cv2.imwrite(tmp_attacked_img_path, attacked_img)
		has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, tmp_attacked_img_path)
		print(has_watermark, _wpsnr)
		#show_images([(attacked_img, 'attacked'),(watermarked_img, 'watermarked'),(watermarked_img - attacked_img, 'diff')],1,3)
		if not has_watermark:
			os.remove(attacked_img_path)
			os.rename(tmp_attacked_img_path, attacked_img_path)
			min_x = idx_x
		else:
			os.remove(tmp_attacked_img_path)
			max_x = idx_x
		idx_x = (min_x + max_x - 1) // 2
	
	attacked_img = cv2.imread(attacked_img_path, cv2.IMREAD_GRAYSCALE)
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	print("Begin reducing y")
	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
	print(has_watermark, _wpsnr)
	idx_y = (min_y + max_y - 1) // 2
	while min_y != idx_y:
		attacked_img = cv2.imread(attacked_img_path, cv2.IMREAD_GRAYSCALE)
		watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
		attacked_img[:,min_y: idx_y] = watermarked_img[:,min_y: idx_y]
		tmp_attacked_img_path = str(uuid.uuid4()) + ".bmp"
		cv2.imwrite(tmp_attacked_img_path, attacked_img)
		has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, tmp_attacked_img_path)
		print(has_watermark, _wpsnr)
		#show_images([(attacked_img, 'attacked'),(watermarked_img, 'watermarked'),(watermarked_img - attacked_img, 'diff')],1,3)
		if not has_watermark:
			os.remove(attacked_img_path)
			os.rename(tmp_attacked_img_path, attacked_img_path)
			min_y = idx_y
		else:
			os.remove(tmp_attacked_img_path)
			max_y = idx_y
		idx_y = (min_y + max_y - 1) // 2


def localize_attack(original_img_path, watermarked_img_path, attacked_img_path, detection_function):
	attacked_img = cv2.imread(attacked_img_path, cv2.IMREAD_GRAYSCALE)
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	show_images([(attacked_img, 'attacked'),(watermarked_img, 'watermarked'),(watermarked_img - attacked_img, 'diff')],1,3)
	min = 0
	max = watermarked_img.shape[0] * watermarked_img.shape[1]

	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
	print(has_watermark, _wpsnr)
	idx = (min + max - 1) // 2
	best_idx = 0
	while min != idx:
		attacked_img = cv2.imread(attacked_img_path, cv2.IMREAD_GRAYSCALE)
		watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
		attacked_img_flatten = attacked_img.flatten()
		watermarked_img_flatten = watermarked_img.flatten()
		attacked_img_flatten[min:idx] = watermarked_img_flatten[min:idx]
		tmp_attacked_img_path = str(uuid.uuid4()) + ".bmp"
		attacked_img = attacked_img_flatten.reshape((512, 512))
		cv2.imwrite(tmp_attacked_img_path, attacked_img)
		has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, tmp_attacked_img_path)
		print(has_watermark, _wpsnr)
		#show_images([(attacked_img, 'attacked'),(watermarked_img, 'watermarked'),(watermarked_img - attacked_img, 'diff')],1,3)
		if not has_watermark:
			os.remove(attacked_img_path)
			os.rename(tmp_attacked_img_path, attacked_img_path)
			min = idx
			best_idx = idx
		else:
			os.remove(tmp_attacked_img_path)
			max = idx
		idx = (min + max - 1) // 2

	locations = []
	for x in range(0,watermarked_img.shape[0]):
		for y in range(0, watermarked_img.shape[1]):
			locations.append((x,y))
	
	return locations[best_idx:]
	

def main():
	'''
	# Uncomment for competition
	watermarked = []
	for group in groups:
		watermarked_imgs = retrieve_others_images(group,IMG_FOLDER_PATH, 'watermarked')
		original_imgs = retrieve_others_images(group,IMG_FOLDER_PATH, 'original')
		lib_detection = import_others_detection(group)
		for original_img, watermarked_img in zip(original_imgs,watermarked_imgs):
			original_img, original_img_path = original_img
			watermarked_img, watermarked_img_path = watermarked_img
			watermarked.append((group, original_img_path, watermarked_img_path, lib_detection.detection))

	'''
	group = 'failedfouriertransform'
	watermarked_imgs = retrieve_others_images(group,IMG_FOLDER_PATH, 'watermarked')[:2]
	original_imgs = retrieve_others_images(group,IMG_FOLDER_PATH, 'original')[:2]
	lib_detection = import_others_detection(group)
	watermarked = []
	for original_img, watermarked_img in zip(original_imgs,watermarked_imgs):
		original_img, original_img_path = original_img
		watermarked_img, watermarked_img_path = watermarked_img
		watermarked.append((group, original_img_path, watermarked_img_path, lib_detection.detection))
	
	valid_attacks = {}
	# Find an attack that works
	for group, original_img_path, watermarked_img_path, detection_function in watermarked:
		attacked_img_path = watermarked_img_path
		img_name = original_img_path.split('/')[3]
		attacked_img_path = '/'.join(original_img_path.split('/')[:2]) + '/attacked/' + TEAM_NAME + '_' + group + '_' + img_name
		valid_attacks[original_img_path] = ''
		watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)

		while valid_attacks[original_img_path] == '':
			attack_list, index_list = get_indexed_random_attacks(randint(1,MAX_N_ATTACKS))
			attacked_img, attack_description = do_attacks(watermarked_img, attack_list)
			cv2.imwrite(attacked_img_path, attacked_img)
			has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
			if has_watermark == 0 and _wpsnr > 35:
				print(attack_description,has_watermark, _wpsnr)
				valid_attacks[original_img_path] = (original_img_path, watermarked_img_path, attacked_img_path,attack_list, index_list, _wpsnr, detection_function)
	
	
	results = {}
	work = []
	for original_img_path in valid_attacks:
		print(f"Attacking: {original_img_path}")
		original_img_path, watermarked_img_path, attacked_img_path,attack_list, index_list, _wpsnr, detection_function = valid_attacks[original_img_path]
		work.append((original_img_path, watermarked_img_path, attacked_img_path,attack_list, index_list, _wpsnr, detection_function))

	results = multiprocessed_workload(mutate_attack,work)
	f = open("output.txt", "a")
	for result in results:
		print("Wrote" + result[0]['attacked_img_path'])
		f.write(str(result[0]) + "\n")
	f.close()
	
if __name__ == '__main__':
	main()

