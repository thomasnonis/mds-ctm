from config import *
from attacks import *
from others import *
from tools import multiprocessed_workload, show_images, localize_attack
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
		print("Mutated attack",attack_description, has_watermark, _wpsnr)
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
	
	print("Mutated attack",attack_description, has_watermark, _wpsnr)
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
	print("Mutated attack",attack_description, has_watermark, _wpsnr)
	os.remove(tmp_attacked_img_path)
	if has_watermark == 0 and _wpsnr > old_wpsnr:	
		attack_list, index_list = attack_list_copy, index_list_copy
	else:
		pass
		#print("Attempt to do substituion failed!")
	
	return attack_list, index_list

def find_attack(params, order_of_execution):
	original_img_path, watermarked_img_path, attacked_img_path, detection_function = params
	
	attack_found = False
	index_list = []
	attack_list = []
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	while not attack_found:
		attack_list, index_list = get_indexed_random_attacks(randint(1,MAX_N_ATTACKS))
		attacked_img, attack_description = do_attacks(watermarked_img, attack_list)
		cv2.imwrite(attacked_img_path, attacked_img)
		has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
		if has_watermark == 0: #and _wpsnr > 35:
			print(attack_description,has_watermark, _wpsnr)
			attack_found = True

	
	print("OLD Attack",describe_attacks(attack_list), _wpsnr)
	for i in range(0,ATTACKS_MUTATIONS):
		print(f"Performing mutation {i} on image {watermarked_img_path}")
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
	print("="*10)
	print("Watermarked Path", watermarked_img_path)
	print("Best attack was: ", describe_attacks(attack_list))
	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
	print("Has watermark?", has_watermark)
	print("WPSNR:", _wpsnr)

	
	
	location, _wpsnr = localize_attack(original_img_path, watermarked_img_path, attacked_img_path, detection_function)
	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
	

	result = {
		'Has watermark' : has_watermark,
		'Attack' : describe_attacks(attack_list),
		'WPSNR' : _wpsnr,
		'attacked_img_path': attacked_img_path,
		'location' : location
	}
	return order_of_execution, result

	

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
	group = 'thebavarians'
	watermarked_imgs = retrieve_others_images(group,IMG_FOLDER_PATH, 'watermarked')
	original_imgs = retrieve_others_images(group,IMG_FOLDER_PATH, 'original')
	lib_detection = import_others_detection(group)
	watermarked = []
	for original_img, watermarked_img in zip(original_imgs,watermarked_imgs):
		original_img, original_img_path = original_img
		watermarked_img, watermarked_img_path = watermarked_img
		watermarked.append((group, original_img_path, watermarked_img_path, lib_detection.detection))
	results = {}
	work = []
	for group, original_img_path, watermarked_img_path, detection_function in watermarked:
		print(f"Attacking: {original_img_path}")
		attacked_img_path = watermarked_img_path
		img_name = original_img_path.split('/')[3]
		attacked_img_path = '/'.join(original_img_path.split('/')[:2]) + '/attacked/' + TEAM_NAME + '_' + group + '_' + img_name	
		work.append((original_img_path, watermarked_img_path, attacked_img_path, detection_function))

	results = multiprocessed_workload(find_attack,work)
	f = open("output.txt", "a")
	for result in results:
		print("Wrote " + result[0]['attacked_img_path'])
		f.write(str(result[0]) + "\n")
	f.close()
	
if __name__ == '__main__':
	main()
	'''
	group = 'thebavarians'
	watermarked_imgs = retrieve_others_images(group,IMG_FOLDER_PATH, 'watermarked')
	original_imgs = retrieve_others_images(group,IMG_FOLDER_PATH, 'original')
	attacked_imgs = retrieve_others_images(group,IMG_FOLDER_PATH, 'attacked')
	lib_detection = import_others_detection(group)
	f = open("output.txt", "a")
	
	for original_img, watermarked_img, attacked_img in zip(original_imgs, watermarked_imgs, attacked_imgs):
		result = {}
		original_img, original_img_path = original_img
		watermarked_img, watermarked_img_path = watermarked_img
		attacked_img, attacked_img_path = attacked_img

		has_watermark, _wpsnr = lib_detection.detection(original_img_path, watermarked_img_path, attacked_img_path)
		print(watermarked_img_path, has_watermark, _wpsnr)
		location, wpsnr = localize_attack(original_img_path, watermarked_img_path, attacked_img_path, lib_detection.detection)
		result[watermarked_img_path] = {
			'WPSNR' : wpsnr,
			'starting_location' : location
		}
		f.write(str(result) + "\n")
	f.close()
	'''

		

