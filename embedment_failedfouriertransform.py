import numpy as np
from tools import wavedec2d, waverec2d, dct2d, idct2d


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
    locations = [(val//rows, val%rows) for val in locations][1:] # locations as (x,y) coordinates, skip DC
    
    watermarked_dct = ori_dct
    for idx, (loc,mark_val) in enumerate(zip(locations, watermark.flatten())):
        watermarked_dct[loc] += (alpha * mark_val)

    # Restore sign and o back to spatial domain
    watermarked_dct *= sign
    return watermarked_dct

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

        # print(band)

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