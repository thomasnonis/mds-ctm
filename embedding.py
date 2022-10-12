from scipy.fft import dct, idct
import numpy as np
from math import sqrt,log,e
from transforms import dct2d,idct2d
import pywt

def embedding_dct(image, mark, alpha, v='multiplicative'):
    # Get the DCT transform of the image
    ori_dct = dct2d(image) 

    # Get the locations of the most perceptually significant components
    sign = np.sign(ori_dct)
    ori_dct = abs(ori_dct)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val//rows, val%rows) for val in locations] # locations as (x,y) coordinates

    # Embed the watermark
    watermarked_dct = ori_dct.copy()
    for idx, (loc,mark_val) in enumerate(zip(locations[1:], mark)):
        if v == 'additive':
            watermarked_dct[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_dct[loc] *= 1 + ( alpha * mark_val)
        elif v == 'exponential':
            watermarked_dct[loc] *= e**(alpha*mark_val)

    # Restore sign and o back to spatial domain
    watermarked_dct *= sign
    watermarked = np.uint8(idct2d(watermarked_dct))

    return watermarked

def detection_dct(image, watermarked, alpha, mark_size, v='multiplicative'):
    ori_dct = dct2d(image)
    wat_dct = dct2d(watermarked)

    # Get the locations of the most perceptually significant components
    ori_dct = abs(ori_dct)
    wat_dct = abs(wat_dct)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val//rows, val%rows) for val in locations] # locations as (x,y) coordinates

    # Generate a watermark
    w_ex = np.zeros(mark_size, dtype=np.float64)

    # Embed the watermark
    for idx, loc in enumerate(locations[1:mark_size+1]):
        if v=='additive':
            w_ex[idx] =  (wat_dct[loc] - ori_dct[loc]) /alpha
        elif v=='multiplicative':
            w_ex[idx] =  (wat_dct[loc] - ori_dct[loc]) / (alpha*ori_dct[loc])
        elif v=='exponential':
            w_ex[idx] =  log(wat_dct[loc] / ori_dct[loc]) / alpha
            
    return w_ex

def detection_dwt(image, watermarked, alpha, mark_size):
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    coeffs2 = pywt.dwt2(LL, 'haar')
    LL2, (LH2, HL2, HH2) = coeffs2

    coeffs3 = pywt.dwt2(LL2, 'haar')
    LL3, (LH3, HL3, HH3) = coeffs3


    n_coeffs = pywt.dwt2(watermarked, 'haar')
    nLL, (nLH, nHL, nHH) = n_coeffs

    n_coeffs2 = pywt.dwt2(nLL, 'haar')
    nLL2, (nLH2,nHL2, nHH2) = n_coeffs2

    n_coeffs3 = pywt.dwt2(nLL2, 'haar')
    nLL3, (nLH3, nHL3, nHH3) = n_coeffs3

    # Generate a watermark
    w_ex = np.zeros([mark_size[0],mark_size[1]], dtype=np.float64)

    # Embed the watermark
    for i in range(0,mark_size[0]):
        for j in range(0,mark_size[1]):
            w_ex[i][j] = (nLL3[i][j]  - LL3[i][j]) / alpha
                
    return w_ex