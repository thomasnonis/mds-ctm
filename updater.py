import sys
from time import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *

start_time = time()

start_string = '# ////VARIABLES START////'
end_string = '# ////VARIABLES END////'


keys = {}

def update_parameters(filename, svd_keys, **kwargs):
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
        if type(kwargs[key]) != dict:
            string += key + ' = ' + str(kwargs[key]) + '\n'

    # keys must be a dictionary with the following structure: key['lena'] = [(svd_u, svd_u)]
    string += 'svd_keys = {}'
    for keys_key in svd_keys.keys():
        string += '\nsvd_keys[\'{}\']'.format(keys_key) + ' = (' + np.array2string(svd_keys[keys_key][0], separator=',',
                                                                                   suppress_small=False,
                                                                                   threshold=9999999,
                                                                                   max_line_width=9999999).replace('\n',
                                                                                                                   '') + '), (' + np.array2string(
            svd_keys[keys_key][1], separator=',', suppress_small=False, threshold=9999999,
            max_line_width=9999999).replace('\n', '') + ')'

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


def embed_watermark_dct(original_img: np.ndarray, img_name: str, watermark: np.ndarray, alpha: float, level,
                        subbands: list) -> np.ndarray:
    """Embeds a watermark into the S component of the SVD decomposition of an image's LL DWT subband

    Args:
        original_img (np.ndarray): Image in which to embed the watermark
        img_name (str): Name of the image
        watermark (np.ndarray): Watermark to embed
        alpha (float): Watermark embedding strength coefficient
        subbands (list): List of subbands where to embed the watermark

    Returns:
        np.ndarray: Watermarked image
    """
    coeffs = wavedec2d(original_img, level)

    for subband in subbands:
        band = None
        if subband == "LL":
            band = coeffs[0]
        elif subband == "HL":
            band = coeffs[1][0]
        elif subband == "LH":
            band = coeffs[1][1]
        elif subband == "HH":
            band = coeffs[1][2]
        else:
            raise Exception(f"Subband {subband} does not exist")

        band = dct(dct(band, axis=0, norm='ortho'), axis=1, norm='ortho')

        # keys must be a dictionary with the following structure: key['lena'] = [(svd_u, svd_v)]
        band_svd, svd_key = embed_into_svd(band, watermark, alpha)

        band_svd = idct(idct(band_svd, axis=1, norm='ortho'), axis=0, norm='ortho')

        if subband == "LL":
            coeffs[0] = band_svd
        elif subband == "HL":
            coeffs[1] = (band_svd, coeffs[1][1], coeffs[1][2])
        elif subband == "LH":
            coeffs[1] = (coeffs[1][0], band_svd, coeffs[1][2])
        elif subband == "HH":
            coeffs[1] = (coeffs[1][0], coeffs[1][1], band_svd)
        else:
            raise Exception(f"Subband {subband} does not exist")

    return (waverec2d(coeffs), svd_key)


subbands = ['LL'] # ['HL', 'LH']
images = import_images(IMG_FOLDER_PATH, min(len(os.listdir(IMG_FOLDER_PATH)), N_IMAGES_LIMIT), False)
watermark = np.load("failedfouriertransform.npy").reshape((MARK_SIZE, MARK_SIZE))

for i in images:
    print("Embedding image " + i[1])
    print(i[0].shape)
    w, key = embed_watermark_dct(i[0], i[1], watermark, ALPHA_GAS, DWT_LEVEL_GAS, subbands)
    keys[i[1].lower()] = key

print("Updating parameters")
update_parameters("detection_failedfouriertrasform.py", keys, ALPHA = ALPHA_GAS, DWT_LEVEL = DWT_LEVEL_GAS, subbands = subbands, threshold = 11.97636616610219)
