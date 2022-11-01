import numpy as np
from scipy.fft import dct, idct
from pywt import wavedec2, waverec2
from tools import import_images
from detection_failedfouriertransform import wpsnr
MARK_SIZE = 32

def dct2d(img):
	return dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2d(img):
	return idct(idct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def wavedec2d(image, level):
	return wavedec2(image, wavelet='haar', level=level)

def waverec2d(coeffs):
	return waverec2(coeffs,wavelet='haar')

def extract_from_dct(original_img, watermarked_img, alpha):
    # Initialize the watermark matrix
    #watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)

    ori_dct = abs(original_img)
    wat_dct = abs(watermarked_img)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = original_img.shape[0]
    locations = [(val//rows, val%rows) for val in locations][SKIP_LOCATIONS:] # locations as (x,y) coordinates

    # Generate a watermark
    mark_size = MARK_SIZE * MARK_SIZE
    w_ex = np.zeros((mark_size), dtype=np.float64)

    # Embed the watermark
    print(len(locations))
    for idx, loc in enumerate(locations[:mark_size]):
        w_ex[idx] =  (wat_dct[loc] - ori_dct[loc]) /alpha
            
    return w_ex.reshape((MARK_SIZE, MARK_SIZE))

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

        original_band = dct2d(original_band)
        watermarked_band = dct2d(watermarked_band)

        watermark = extract_from_dct(original_band, watermarked_band, alpha)
        watermarks.append(watermark)

    final_watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)

    for watermark in watermarks:
        final_watermark += watermark
    final_watermark = final_watermark / len(subbands)

    return final_watermark


def embed_into_dct(img: np.ndarray, watermark: list, alpha: float) -> tuple:
    """Embeds the watermark into the DCT tranfom of a subband

    Args:
        img (np.ndarray): Image in which to embed the watermark
        watermark (list): Watermark to embed
        alpha (float): Embedding strength coefficient

    Returns:
        tuple: Watermarked image: np.ndarray
    """
    sign = np.sign(img)
    ori_dct = abs(img)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = img.shape[0]
    print([(val//rows, val%rows) for val in locations][:10])
    locations = [(val//rows, val%rows) for val in locations][SKIP_LOCATIONS:] # locations as (x,y) coordinates
    
    watermarked_dct = ori_dct
    for idx, (loc,mark_val) in enumerate(zip(locations, watermark.flatten())):
        watermarked_dct[loc] += (alpha * mark_val)

    # Restore sign and o back to spatial domain
    watermarked_dct *= sign
    return watermarked_dct

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def embed_watermark(original_img: np.ndarray, img_name: str, watermark: np.ndarray, alpha: float, level, subbands: list) -> np.ndarray:
    """Embeds a watermark into the DWT subband after calculating its DCT tranform

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

        band = dct2d(band)

        band_dct = embed_into_dct(band, watermark, alpha)

        band_dct = idct2d(band_dct)

        if subband == "LL":
            coeffs[0] = band_dct
        elif subband == "HL":
            coeffs[1] = (band_dct, coeffs[1][1], coeffs[1][2])
        elif subband == "LH":
            coeffs[1] = (coeffs[1][0], band_dct, coeffs[1][2])
        elif subband == "HH":
            coeffs[1] = (coeffs[1][0], coeffs[1][1], band_dct)
        else:
            raise Exception(f"Subband {subband} does not exist")

    return waverec2d(coeffs)

from config import *
from tools import show_images
images = import_images(IMG_FOLDER_PATH, 1, False)
watermark = np.load("failedfouriertransform.npy").reshape((MARK_SIZE, MARK_SIZE))

original_img, img_name = images[0]

alpha = 50
level = 2
subbands = ["LL"]
watermarked = embed_watermark(original_img, img_name, watermark, alpha, level, subbands)
extracted_watermark = extract_watermark(original_img, img_name, watermarked, alpha, level, subbands)
show_images([(original_img, "original_img"), (watermarked, "watermarked")],1,2)
print(wpsnr(original_img, watermarked))
show_images([(extracted_watermark, "extracted_watermark"), (watermark, "watermark")],1,2)