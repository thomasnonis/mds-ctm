from config import *
from detection_failedfouriertransform import ALPHA, DWT_LEVEL, SUBBANDS, detection, wpsnr
from embedment_failedfouriertransform import embed_watermark
from attacks import *
import numpy as np
from random import randint, random
import os

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
		tmp_attacked_img_path = str(uuid.uuid1()) + ".bmp"
		cv2.imwrite(tmp_attacked_img_path, attacked_img)
		has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, tmp_attacked_img_path)
		print("NEW Attack",attack_description, _wpsnr)
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

def old_reduce_attack_strength(idx, attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, old_wpsnr):
	can_still_reduce = True
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	while can_still_reduce:
		index = index_list[idx]
		attack = attack_list[idx]
		can_still_reduce = False
		# If this condition is true means we can reduce the stenght of this attack
		if index + 1 != len(attack_parameters[attack['function']]):
			new_params = attack_parameters[attack['function']][index+1]
			new_params_dict = parse_parameters(attack['function'], new_params)
			backup_attack = attack_list[idx]['arguments'].copy()
			backup_index = index
			index_list[idx] = index + 1
			# Update the attack parameters with a weaker attack
			attack_list[idx]['arguments'] = new_params_dict
			
			attacked_img, attack_description = do_attacks(watermarked_img, attack_list)
			tmp_attacked_img_path = str(uuid.uuid1()) + ".bmp"
			cv2.imwrite(tmp_attacked_img_path, attacked_img)
			has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, tmp_attacked_img_path)
			print("NEW Attack",attack_description, _wpsnr)
			if has_watermark == 0 and _wpsnr > old_wpsnr:
				can_still_reduce = True
				os.remove(attacked_img_path)
				os.rename(tmp_attacked_img_path, attacked_img_path)
			else:
				print("Tried to reduce strenght of attack but the watermark is present or got a lower wpsnr")
				attack_list[idx]['arguments'] = backup_attack
				index_list[idx] = backup_index
				os.remove(tmp_attacked_img_path)
		else:
			print("Reached maximum reduction!")
	return attack_list, index_list

def remove_attack(idx, attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, old_wpsnr):
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	
	index_list_copy = index_list.copy()
	attack_list_copy = attack_list.copy()
	# Remove an attack
	index_list_copy = index_list_copy[:idx] + index_list_copy[idx+1:]
	attack_list_copy = attack_list_copy[:idx] + attack_list_copy[idx+1:]
	print(f"Substituting {get_attack_description(attack_list[idx])} with nothing")
	
	attacked_img, attack_description = do_attacks(watermarked_img, attack_list_copy)
	tmp_attacked_img_path = str(uuid.uuid1()) + ".bmp"
	cv2.imwrite(tmp_attacked_img_path, attacked_img)
	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, tmp_attacked_img_path)
	
	print("NEW Attack",attack_description, _wpsnr)
	if has_watermark == 0 and _wpsnr > old_wpsnr:
		attack_list, index_list = attack_list_copy, index_list_copy
		os.remove(attacked_img_path)
		os.rename(tmp_attacked_img_path, attacked_img_path)
	else:
		os.remove(tmp_attacked_img_path)
		print("Attempt to do substituion failed!")

	return attack_list, index_list
def swap_attack(idx, attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, old_wpsnr):
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	
	index_list_copy = index_list.copy()
	attack_list_copy = attack_list.copy()

	new_attack, new_index = get_indexed_random_attacks(1)
	index_list_copy[idx] = new_index[0]
	attack_list_copy[idx] = new_attack[0]
	print(f"Substituting {get_attack_description(attack_list[idx])} with {get_attack_description(attack_list_copy[idx])}")
	print(f'Trying to reduce {get_attack_description(attack_list_copy[idx])}')
	attack_list_copy, index_list_copy = reduce_attack_strength(idx,attack_list_copy, index_list_copy, detection_function, original_img_path, watermarked_img_path, attacked_img_path, old_wpsnr)
	
	
	attacked_img, attack_description = do_attacks(watermarked_img, attack_list_copy)
	tmp_attacked_img_path = str(uuid.uuid1()) + ".bmp"
	cv2.imwrite(tmp_attacked_img_path, attacked_img)
	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, tmp_attacked_img_path)
	print("NEW Attack",attack_description, _wpsnr)
	os.remove(tmp_attacked_img_path)
	if has_watermark == 0 and _wpsnr > old_wpsnr:	
		attack_list, index_list = attack_list_copy, index_list_copy
	else:
		print("Attempt to do substituion failed!")
	
	return attack_list, index_list

def main():
	# Get N_IMAGES_LIMIT random images
	images_to_watermark = import_images(IMG_TO_WATERMARK_FOLDER_PATH,N_IMAGES_LIMIT,False)
	# Read watermark
	watermark = np.load("failedfouriertransform.npy").reshape((MARK_SIZE, MARK_SIZE))
	watermarked = []
	for original_img, img_path in images_to_watermark:
		watermarked_img = embed_watermark(original_img, watermark, ALPHA, DWT_LEVEL, SUBBANDS)
		img_name = img_path.split('/')[2]
		watermarked_path = IMG_WATERMARKED_FOLDER_PATH + img_name
		cv2.imwrite(watermarked_path, watermarked_img)
		watermarked.append((img_path, watermarked_path, detection))

	valid_attacks = {}
	# Find an attack that works
	for original_img_path, watermarked_img_path, detection_function in watermarked:
		attacked_img_path = watermarked_img_path
		img_name = original_img_path.split('/')[2]
		valid_attacks[img_name] = ''
		watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)

		while valid_attacks[img_name] == '':
			attack_list, index_list = get_indexed_random_attacks(randint(1,MAX_N_ATTACKS))
			attacked_img, attack_description = do_attacks(watermarked_img, attack_list)
			attacked_img_path = IMG_ATTACKED_FOLDER_PATH + img_name
			cv2.imwrite(attacked_img_path, attacked_img)
			has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
			if has_watermark == 0 and _wpsnr > 35:
				print(attack_description,has_watermark, _wpsnr)
				valid_attacks[img_name] = (original_img_path, watermarked_img_path, attacked_img_path,attack_list, index_list, _wpsnr)
	

	for img_name in valid_attacks:
		original_img_path, watermarked_img_path, attacked_img_path,attack_list, index_list, _wpsnr = valid_attacks[img_name]
		print("OLD Attack",describe_attacks(attack_list), _wpsnr)
		for i in range(0,20):
			strategy = randint(0,2)
			if strategy == 0:
				# Decrease the strength of an attack in the attack list
				idx = randint(0,len(attack_list)-1)
				print(f'Trying to reduce {get_attack_description(attack_list[idx])}')
				attack_list, index_list = reduce_attack_strength(idx,attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, _wpsnr)
			elif strategy == 1:
				if len(attack_list) > 1:
					# Swap an attack from the attack list with another attack, or removes it
					idx = randint(0,len(attack_list)-1)
					print(f'Trying to remove {get_attack_description(attack_list[idx])}')
					attack_list, index_list = remove_attack(idx,attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, _wpsnr)
			elif strategy == 2:
				# Swap an attack from the attack list with another attack
				idx = randint(0,len(attack_list)-1)
				print(f'Trying to swap {get_attack_description(attack_list[idx])}')
				attack_list, index_list = swap_attack(idx,attack_list, index_list, detection_function, original_img_path, watermarked_img_path, attacked_img_path, _wpsnr)
		
		watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
		attacked_img, attack_description = do_attacks(watermarked_img, attack_list)
		print("Best attack was: ", describe_attacks(attack_list))
		has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
		print("Has watermark?", has_watermark)
		print("WPSNR:", _wpsnr)
		
	
if __name__ == '__main__':
	main()

