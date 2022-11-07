import numpy as np
import cv2 as cv
from tools import wavedec2d, waverec2d, dct2d, idct2d, merge
from detection_failedfouriertransform import split, ALPHA, DWT_LEVEL, SUBBANDS, MARK_SIZE


def embed_into_dct(img: np.ndarray, watermark: list, alpha: float) -> tuple:
    """Embeds the watermark into the DCT tranfom of a subband

    Args:
        img (np.ndarray): Image in which to embed the watermark
        watermark (list): Watermark to embed
        alpha (float): Embedding strength coefficient

    Returns:
        tuple: Watermarked image: np.ndarray
    """
    # img (128,128)
    blocks = split(img,4,4) # (1024,4,4)
    output = np.zeros(blocks.shape) # (1024,4,4)
    watermark = watermark.flatten() # 1024,1
    for idx,block in enumerate(blocks):
        dct_block = dct2d(block)
        dct_block[0][0] += alpha * watermark[idx] # Embed in DC
        block = idct2d(dct_block)
        output[idx] = block  

    watermarked = merge(output, img.shape[0], img.shape[1])
    return watermarked

# embedded images naming convention: imageName_groupA.bmp
def embed_watermark(original_img: np.ndarray, watermark: np.ndarray, alpha: float, level, subbands: list) -> np.ndarray:
    """Embeds a watermark into the DWT subband after calculating its DCT tranform

    Args:
        original_img (np.ndarray): Image in which to embed the watermark
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

        band = embed_into_dct(band, watermark, alpha)

        if subband == "LL":
            coeffs[0] = band
        elif subband == "HL":
            coeffs[1] = (band, coeffs[1][1], coeffs[1][2])
        elif subband == "LH":
            coeffs[1] = (coeffs[1][0], band, coeffs[1][2])
        elif subband == "HH":
            coeffs[1] = (coeffs[1][0], coeffs[1][1], band)
        else:
            raise Exception(f"Subband {subband} does not exist")

    return waverec2d(coeffs)

from tools import import_images
from config import TEAM_NAME

if __name__ == "__main__":
    watermark = np.load('failedfouriertransform.npy')
    images = import_images('images/' + TEAM_NAME + '/original/', 10, False)
    for img, img_name in images:
        watermarked = embed_watermark(img, watermark, ALPHA, DWT_LEVEL, SUBBANDS)
        cv.imwrite('images/' + TEAM_NAME + '/watermarked/' + img_name + '_' + TEAM_NAME + '.bmp', watermarked)
    
    print('Watermarking done')