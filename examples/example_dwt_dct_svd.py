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
print('Starting DWT_DCT_SVD...')

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

        # print(band)

        band = dct(dct(band, axis=0, norm='ortho'), axis=1, norm='ortho')

        band_svd, svd_key = embed_into_svd(band, watermark, alpha)
        save_parameters(img_name + '_' + subband + str(level), svd_key)

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

    """watermark = waverec2d(coeffs)
    print(wpsnr(watermark, original_img))
    plt.figure()
    plt.subplot(121)
    plt.title("Original")
    plt.imshow(original_img, cmap='gray')
    plt.subplot(122)
    plt.title("Watermarked")
    plt.imshow(watermark, cmap='gray')
    plt.show()"""
    return waverec2d(coeffs)


def extract_watermark_dct(original_img: np.ndarray, img_name: str, watermarked_img: np.ndarray, alpha, level: int,
                      subbands: list) -> np.ndarray:
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
            watermarked_band = dct(dct(watermarked_coeffs[0], axis=0, norm='ortho'), axis=1, norm='ortho')
        elif subband == "HL":
            original_band = original_coeffs[1][0]
            watermarked_band = dct(dct(watermarked_coeffs[1][0], axis=0, norm='ortho'), axis=1, norm='ortho')
        elif subband == "LH":
            original_band = original_coeffs[1][1]
            watermarked_band = dct(dct(watermarked_coeffs[1][1], axis=0, norm='ortho'), axis=1, norm='ortho')
        elif subband == "HH":
            original_band = original_coeffs[1][2]
            watermarked_band = dct(dct(watermarked_coeffs[1][2], axis=0, norm='ortho'), axis=1, norm='ortho')
        else:
            raise Exception(f"Subband {subband} does not exist")

        original_band_u, original_band_s, original_band_v = np.linalg.svd(original_band)
        original_band_s = np.diag(original_band_s)

        watermarked_band_u, watermarked_band_s, watermarked_band_v = np.linalg.svd(watermarked_band)
        watermarked_band_s = np.diag(watermarked_band_s)

        (_, svd_key) = read_parameters(img_name + '_' + subband + str(level))
        # original_s_ll_d_u, original_s_ll_d_s, original_s_ll_d_v
        s_band_d = svd_key[0] @ watermarked_band_s @ svd_key[1]

        # Initialize the watermark matrix
        watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)

        # Extract the watermark

        for i in range(0, MARK_SIZE):
            for j in range(0, MARK_SIZE):
                watermark[i][j] = (s_band_d[i][j] - original_band_s[i][j]) / alpha
        watermarks.append((watermark, subband + " " + img_name))

    final_watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)

    for watermark in watermarks:
        final_watermark += watermark[0]
    final_watermark = final_watermark / len(subbands)

    # show_images(watermarks + [(final_watermark, "Final")], 1, 3)
    """plt.figure()
    plt.subplot(121)
    plt.title("Final Watermark")
    plt.imshow(final_watermark, cmap='gray')
    plt.show()"""
    return final_watermark


subbands = ['HL', 'LH']
images = import_images("../" + IMG_FOLDER_PATH, N_IMAGES_LIMIT, True)
watermark = np.load("../failedfouriertransform.npy").reshape((MARK_SIZE, MARK_SIZE))
w_images = []
w_extracted = []
wpsnr_sum = 0

for i in images:
    # w = embed_watermark_tn(i[0], i[1], watermark, ALPHA_TN, BETA)
    w = embed_watermark(i[0], i[1], watermark, ALPHA_GAS, DWT_LEVEL, subbands)
    # w = embed_watermark_dct(i[0], i[1], watermark, ALPHA_GAS, DWT_LEVEL_GAS, subbands)
    w_images.append((w, "Watermarked " + i[1]))

    quality = wpsnr(w, i[0])
    print("WPSNR ", i[1], ": ", quality)
    wpsnr_sum += quality
    attacks_list = get_random_attacks(1)
    attacked = do_attacks(w, attacks_list)
    print("WPSNR attacked with ", attacked[1], i[1], ": ", wpsnr(attacked[0], i[0]))

    # e = extract_watermark_tn(i[0], i[1], attacked[0], ALPHA_TN, BETA)
    e = extract_watermark(i[0], i[1], attacked[0], ALPHA_GAS, DWT_LEVEL, subbands)
    # e = extract_watermark_dct(i[0], i[1], attacked[0], ALPHA_GAS, DWT_LEVEL_GAS, subbands)

    # Embedding quality
    # e = extract_watermark_dct(i[0], i[1], w, ALPHA_TN, DWT_LEVEL, subbands)
    w_extracted.append((e, "Extracted " + i[1]))

    print("SIM extraction ", i[1], ": ", similarity(watermark, e))

print("Alpha:", ALPHA_GAS, "Level:", DWT_LEVEL_GAS, "WPSNR mean:", wpsnr_sum / N_IMAGES_LIMIT, "Mark:", wpsnr_to_mark(wpsnr_sum / N_IMAGES_LIMIT))
"""show_images(images + w_images, 2, len(images))
show_images([(watermark, "Original")] + w_extracted, 2, 2)"""
"""
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
"""
