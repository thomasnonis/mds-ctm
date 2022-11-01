import numpy as np
import cv2 as cv
import os
from scipy.signal import convolve2d
from math import sqrt
from scipy.fft import dct, idct
from pywt import dwt2, wavedec2
import os
from tools import split

# ////VARIABLES START////
ALPHA = 55
DWT_LEVEL = 4
SUBBANDS = ['LL']
DETECTION_THRESHOLD = 15.496826790280725
MARK_SIZE = 32

# ////VARIABLES END////

TEAM_NAME = 'failedfouriertransform'

def dct2d(img):
	return dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2d(img):
	return idct(idct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def dwt2d(img):
    return dwt2(img, 'haar')

def wavedec2d(image, level):
    return wavedec2(image, wavelet='haar', level=level)

def similarity(img1, img2):
    # Computes the similarity measure between the original and the new watermarks.
    return np.sum(np.multiply(img1, img2)) / np.sqrt(np.sum(np.multiply(img2, img2)))

def csf(img):
    if not os.path.isfile('csf.csv'):
        os.system(
            'python -m wget "https://drive.google.com/uc?export=download&id=1w43k1BTfrWm6X0rqAOQhrbX6JDhIKTRW" -o csf.csv')
    if not os.path.isfile('csf.csv'):
        raise FileNotFoundError('csf.csv not found')

    csf = np.genfromtxt('csf.csv', delimiter=',')
    return convolve2d(img, np.rot90(csf, 2), mode='same')

def wpsnr(img1: np.ndarray, img2: np.ndarray):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0

    difference = img1 - img2
    same = not np.any(difference)
    if same is True:
        return 9999999

    ew = csf(difference)
    decibels = 20.0 * np.log10(1.0 / sqrt(
        np.mean(np.mean(ew ** 2))))  # this is something that can be optimized by using numerical values instead of db
    return decibels

def extract_from_dct(original_img, watermarked_img, alpha):
    
    original_blocks = split(original_img, 4, 4)
    watermarked_blocks = split(watermarked_img, 4, 4)
    watermark = np.zeros(MARK_SIZE*MARK_SIZE)
    for idx,(original_block, watermarked_block) in enumerate(zip(original_blocks, watermarked_blocks)):
        original_dct_block = dct2d(original_block)
        watermarked_dct_block = dct2d(watermarked_block)
        dc_diff = (watermarked_dct_block[0][0] - original_dct_block[0][0]) / alpha
        watermark[idx] =  dc_diff

    return watermark.reshape((MARK_SIZE, MARK_SIZE))

def extract_watermark(original_img: np.ndarray, img_name: str, watermarked_img: np.ndarray, alpha: int, level: int, subbands: list) -> np.ndarray:
    """Extracts the watermark from a watermarked image by appling the reversed embedding algorithm,
    provided that the proper configuration file and the original, unwatermarked, image are available.

    Args:
        original_img (np.ndarray): Original unwatermarked image
        img_name (str): Name of the image
        watermarked_img (np.ndarray): Image from which to extract the watermark
        subbands (list): List of subbands where to extract the watermark

    Returns:
        np.ndarray: Extracted watermark
    """
    original_coeffs = wavedec2d(original_img, level)
    watermarked_coeffs = wavedec2d(watermarked_img, level)
    watermarks = []
    for subband in subbands:
        original_band = None
        watermarked_band = None
        if subband == "LL":
            original_band = original_coeffs[0]
            watermarked_band = watermarked_coeffs[0]
        elif subband == "HL":
            original_band = original_coeffs[1][0]
            watermarked_band = watermarked_coeffs[1][0]
        elif subband == "LH":
            original_band = original_coeffs[1][1]
            watermarked_band = watermarked_coeffs[1][1]
        elif subband == "HH":
            original_band = original_coeffs[1][2]
            watermarked_band = watermarked_coeffs[1][2]
        else:
            raise Exception(f"Subband {subband} does not exist")

        watermark = extract_from_dct(original_band, watermarked_band, alpha)
        watermarks.append(watermark)

    final_watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)

    for watermark in watermarks:
        final_watermark += watermark
    final_watermark = final_watermark / len(subbands)

    return final_watermark




def detection(original_path, watermarked_path, attacked_path):
    original_img = cv.imread(original_path, cv.IMREAD_GRAYSCALE)
    watermarked_img = cv.imread(watermarked_path, cv.IMREAD_GRAYSCALE)
    attacked_img = cv.imread(attacked_path, cv.IMREAD_GRAYSCALE)

    # Our watermarked images must be named: imageName_failedfouriertransform.bmp
    # Watermarked images by other groups will be named: groupB_imageName.bmp
    # Attacked images must be named: failedfouriertransform_groupB_imageName.bmp
    '''
    pixel
    ef26420c
    you_shall_not_mark
    blitz
    omega
    howimetyourmark
    weusedlsb
    thebavarians
    theyarethesamepicture
    dinkleberg
    failedfouriertransform
    '''

    if watermarked_path.lower().split('/')[-1].split('_')[-1].split('.')[0] == TEAM_NAME:
        img_name = watermarked_path.lower().split('/')[-1].split('_')[0]
    else:
        img_name = watermarked_path.lower().split('/')[-1].split('.')[-2].split('_')[-1]
    original_watermark = extract_watermark(original_img, img_name, watermarked_img, ALPHA, DWT_LEVEL, SUBBANDS)
    attacked_watermark = extract_watermark(original_img, img_name, attacked_img, ALPHA, DWT_LEVEL, SUBBANDS)
    
    if similarity(original_watermark, attacked_watermark) > DETECTION_THRESHOLD:
        has_watermark = True
    else:
        has_watermark = False

    return has_watermark, wpsnr(original_img, attacked_img)
