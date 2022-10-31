import numpy as np
from tools import wavedec2d, waverec2d, dct2d, idct2d


def embed_into_svd(img: np.ndarray, watermark: list, alpha: float) -> tuple:
    """Embeds the watermark into the S component of the SVD decomposition of the image

    Args:
        img (np.ndarray): Image in which to embed the watermark
        watermark (list): Watermark to embed
        alpha (float): Embedding strength coefficient

    Returns:
        tuple: (Watermarked image: np.ndarray, SVD key matrices: tuple)
    """
    (svd_u, svd_s, svd_v) = np.linalg.svd(img)

    # Convert S from a 1D vector to a 2D diagonal matrix
    svd_s = np.diag(svd_s)

    # Embed the watermark in the SVD matrix
    for x in range(0, watermark.shape[0]):
        for y in range(0, watermark.shape[1]):
            svd_s[x][y] += alpha * watermark[x][y]

    (svd_s_u, svd_s_s, svd_s_v) = np.linalg.svd(svd_s)

    # Convert S from a 1D vector to a 2D diagonal matrix
    svd_s_s = np.diag(svd_s_s)

    # Recompose matrices from SVD decomposition
    watermarked = svd_u @ svd_s_s @ svd_v
    # key = svd_s_u @ svd_s @ svd_s_v

    return (watermarked, (svd_s_u, svd_s_v))

def embed_watermark(original_img: np.ndarray, img_name: str, watermark: np.ndarray, alpha: float, level, subbands: list) -> np.ndarray:
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

        band = dct2d(band)

        band_svd, svd_key = embed_into_svd(band, watermark, alpha)

        band_svd = idct2d(band_svd)

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

    return waverec2d(coeffs), svd_key