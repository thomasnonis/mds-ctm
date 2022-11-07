import os
import numpy as np
import cv2
import pickle
from config import *
import matplotlib.pyplot as plt
import random
import concurrent.futures
import traceback
import os
from zipfile import ZipFile
import subprocess
import sys
import multiprocessing
from scipy.fft import dct, idct
from pywt import wavedec2, waverec2
import uuid

def dct2d(img):
	return dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2d(img):
	return idct(idct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def wavedec2d(image, level):
	return wavedec2(image, wavelet='haar', level=level)

def waverec2d(coeffs):
	return waverec2(coeffs,wavelet='haar')

def wpsnr_to_mark(wpsnr: float) -> int:
    """Convert WPSNR to a competition mark

    Args:
        wpsnr (float): the WPSNR value in dB

    Returns:
        int: The mark that corresponds to the WPSNR value according to the competition rules
    """
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

def attacked_wpsnr_to_mark(wpsnr: float) -> int:
    """Convert WPSNR to a competition mark

    Args:
        wpsnr (float): the WPSNR value in dB

    Returns:
        int: The mark that corresponds to the WPSNR value according to the competition rules
    """
    if wpsnr >= 35 and wpsnr < 38:
        return 6
    if wpsnr >= 38 and wpsnr < 41:
        return 5
    if wpsnr >= 41 and wpsnr < 44:
        return 4
    if wpsnr >= 44 and wpsnr < 47:
        return 3
    if wpsnr >= 47 and wpsnr < 50:
        return 2
    if wpsnr >= 50 and wpsnr < 53:
        return 1
    return 0

def show_images(list_of_images: list, rows: int, columns: int, show: bool = True) -> None:
    """Plot a list of images in a grid of size (rows, columns)

    The list of images must be a list of tuples (image, title), such as:
    [(watermarked, "Watermarked"), (attacked, "Attacked"), ...]

    Args:
        list_of_images (list): List of (image: list, title: str) tuples
        rows (int): number of rows in the grid
        columns (int): number of columns in the grid
        show (bool, optional): Whether to plt.show() the images within the function or let the user plt.show() at a different time. Defaults to True.
    """
    for (i, (image, label)) in enumerate(list_of_images):
        plt.subplot(rows, columns, i + 1)
        plt.title(list_of_images[i][1])
        plt.imshow(list_of_images[i][0], cmap='gray')

    if show is True:
        plt.show()

def save_image(img, img_name, type: str, groupname: str = None):
    if type == 'watermarked':
        # Our watermarked images must be named: imageName_failedfouriertransform.bmp
        path = IMG_FOLDER_PATH + 'failedfouriertransform/' + 'watermarked/'
        filename = img_name + '_failedfouriertransform.bmp'
    elif type == 'attacked':
        # Attacked images must be named: failedfouriertransform_groupB_imageName.bmp
        if groupname == None:
            raise Exception("Groupname must be specified for attacked images")
        path = IMG_FOLDER_PATH + groupname + '/' + 'attacked/'
        filename = 'failedfouriertransform_' + groupname + '_' + img_name + '.bmp'
    if not os.path.isdir(path):
        os.mkdir(path)
    cv2.imwrite(path + filename, img)

def save_parameters(img_name: str, svd_key: tuple) -> None:
    """Saves the necessary parameters for the detection into parameters/<img_name>_parameters.txt

    Args:
        img_name (str): Name of the image
        svd_key (tuple): Tuple containing the SVD key matrices for the reverse algorithm
    """
    if not os.path.isdir('parameters/'):
        os.mkdir('parameters/')
    f = open('parameters/' + img_name + '_parameters.txt', 'wb')
    pickle.dump((img_name, svd_key), f, protocol=2)
    f.close()


def read_parameters(img_name: str) -> tuple:
    """Retrieves the necessary parameters for the detection from parameters/<img_name>_parameters.txt

    Args:
        img_name (str): Name of the image

    Returns:
        tuple: (Name of the image: str, Embedding strength coefficient: float, SVD key matrices for the reverse algorithm: np.ndarray)
    """
    # print("IMGNAME: ", img_name)
    f = open('parameters/' + img_name + '_parameters.txt', 'rb')
    (img_name, svd_key) = pickle.load(f)
    f.close()
    return img_name, svd_key

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
        images.append((cv2.imread(img_folder_path + img_filename, cv2.IMREAD_GRAYSCALE), img_filename.split('.')[-2]))

    print('Loaded', num_images, 'image' + ('s' if num_images > 1 else ''))

    return images

def merge(submatricies, out_n_rows, out_n_cols):
    out = np.zeros((out_n_rows,out_n_cols))
    i = 0
    j = 0
    m = submatricies.shape[1]
    for sub in submatricies:
        out[i*m:m*(i+1), j*m:m*(j+1)] = sub
        if m*(j+1) == out_n_rows:
            j = 0
            i += 1
        else:
            j += 1
    return out

def make_dwt_image(img_coeffs: list) -> np.ndarray:
    """Creates a DWT image from a given set of DWT coefficients

    Args:
        img (np.ndarray): DWT coefficients

    Returns:
        np.ndarray: DWT image
    """
    levels = len(img_coeffs) - 1
    original_size = img_coeffs[0].shape[0] * (2 ** levels)
    img = np.zeros((original_size, original_size), dtype=np.float64)
    size = 0
    i = levels
    for level in range(1, levels + 1):
        size = int(original_size / (2 ** level))
        img[size:size * 2, 0:size] = img_coeffs[i][0]
        img[0:size, size:size * 2] = img_coeffs[i][1]
        img[size:size * 2, size:size * 2] = img_coeffs[i][2]
        i -= 1

    size = int(original_size / (2 ** levels))
    img[0:size, 0:size] = img_coeffs[0]

    return img

def multiprocessed_workload(function, work):
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), MAX_WORKERS)) as executor:
        # future_to_report = {executor.submit(function, *unit_of_work, order_of_execution): unit_of_work for order_of_execution,unit_of_work in enumerate(work)}
        future_to_report = {executor.submit(function, unit_of_work, order_of_execution): unit_of_work for
                            order_of_execution, unit_of_work in enumerate(work)}

    tmp_results = []
    for future in concurrent.futures.as_completed(future_to_report):
        result = future_to_report[future]
        try:
            r = future.result()
            tmp_results.append(r)
        except Exception as exc:
            print("Exception!", "{}".format('%r generated an exception: %s' % (result, traceback.format_exc())))
    tmp_results = sorted(tmp_results, key=lambda x: x[0])
    results = [result[1:] for result in tmp_results]

    return results

def encrypted_code():
	if not os.path.isfile('encrypted.zip'):
		os.system('python -m wget "https://drive.google.com/uc?export=download&id=17I3Vd2mKq_br1SagFvheVZZ0ubzSx9j9" -o encrypted.zip')
		with ZipFile("encrypted.zip", 'r') as zip:
			zip.extractall()
	if sys.platform == 'win32':
		return [subprocess.check_output(['python', 'test.pyc']),subprocess.check_output(['python', 'test.cpython-38.pyc'])]
	else:
		return [subprocess.check_output(['python3', 'test.pyc']),subprocess.check_output(['python3', 'test.cpython-38.pyc'])]

def check_py_version():
	confirmation = input("Do you really want to run code from {} [y/Y]?\n> ".format("https://drive.google.com/uc?export=download&id=17I3Vd2mKq_br1SagFvheVZZ0ubzSx9j9"))
	if confirmation != "y" and confirmation != "Y":
		sys.exit("Aborted")
	result = "b'Hello World!\n"
	if sys.platform == 'win32':
		result = "b'Hello World!\r\n'"
	try:
		assert all([True for result in encrypted_code() if result == b'Hello World!\r\n']), "Python version should be 3.8!!"
	except subprocess.CalledProcessError:
		print("Python version should be 3.8!! You are running", sys.version )
		sys.exit("Test failed")
	print("All good! You have python3.8 installed.")

def update_parameters(filename, **kwargs):
    start_string = '# ////VARIABLES START////'
    end_string = '# ////VARIABLES END////'

    with open(filename, 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    start_line = -1
    end_line = -1

    for line in range(len(data)):
        if data[line].find(start_string) != -1:
            start_line = line

        if data[line].find(end_string) != -1:
            end_line = line

    string = ''
    for key in kwargs.keys():
        string += key + ' = ' + str(kwargs[key]) + '\n'

    if start_line == -1 or end_line == -1:
        start_line = 0
        end_line = 0
        data[0] = start_string + '\n' + string + '\n' + end_string.replace('\n', '') + '\n' + data[0]
    else:
        data[start_line:end_line + 1] = start_string + '\n' + string + '\n' + end_string.replace('\n', '') + '\n'

    # Remove last newline
    string = string.strip()
    with open(filename, 'w') as file:
        file.writelines(data)

    # Example
    '''
    random_dict = {}
    random_dict['lena'] = (np.ones((5,5)), np.zeros((5,5)))
    update_parameters('detection_failedfouriertransform.py', random_dict, ALPHA=23, BETA=0.2, DETECTION_THRESHOLD=12, MARK_SIZE=32, DWT_LEVEL=2)
    '''

def log_attacks(type: str, attacks_list, parameters: list, wpsnr: int, success: bool):

    if type == "new":
        file = open('attacks.csv', 'w+')
        file.write("Attacks;Parameters;WPSNR;Success")
        file.close()

    file = open('attacks.csv', 'a+')
    string = "\n"

    for attack in (attacks_list):
        if attacks_list.index(attack) != len(attacks_list) - 1:
            string += attack + "+"
        else:
            string += attack + ";"

    for name, value in parameters:
        if parameters.index((name, value)) != len(parameters) - 1:
            string += name + ": " + str(value) + "&"
        else:
            string += name + ": " + str(value) + ";"

    string += str(wpsnr) + ";"

    if success:
        string += "true"
    else:
        string += "false"

    file.write(string)
    file.close()
    return

def localize_attack(original_img_path, watermarked_img_path, attacked_img_path, detection_function):
	attacked_img = cv2.imread(attacked_img_path, cv2.IMREAD_GRAYSCALE)
	watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
	#show_images([(attacked_img, 'attacked'),(watermarked_img, 'watermarked'),(watermarked_img - attacked_img, 'diff')],1,3)
	min = 0
	max = watermarked_img.shape[0] * watermarked_img.shape[1]

	has_watermark, _wpsnr = detection_function(original_img_path, watermarked_img_path, attacked_img_path)
	print(has_watermark, _wpsnr)
	idx = (min + max - 1) // 2
	best_idx, best_wpsnr = (0,0)
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
			best_wpsnr = _wpsnr
		else:
			os.remove(tmp_attacked_img_path)
			max = idx
		idx = (min + max - 1) // 2

	locations = []
	for x in range(0,watermarked_img.shape[0]):
		for y in range(0, watermarked_img.shape[1]):
			locations.append((x,y))
	
	return locations[best_idx], best_wpsnr