from scipy.fft import dct, idct
import numpy as np
from math import sqrt,log,e
from transforms import dct2d,idct2d,dwt2d,idwt2d, wavedec2d, waverec2d
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

def detection_dwt(image, watermarked, alpha, level, mark_size):
    coeffs = wavedec2d(image,level)
    LL = coeffs[0]

    ncoeffs = wavedec2d(watermarked,level)
    nLL = ncoeffs[0]

    # Generate a watermark
    size = int(mark_size / (2**level))
    wLL = np.zeros([size,size], dtype=np.float64)

    # Embed the watermark
    for i in range(0,size):
        for j in range(0,size):
            wLL[i][j] = (nLL[i][j]  - LL[i][j]) / alpha
    
    coeffs = [wLL]

    for i in range(0,level):
        tmp = np.zeros([max(size,size*(2*i)),max(size,size*(2*i))])
        coeffs.append((tmp,tmp,tmp))

    w_ex = waverec2d(coeffs)
    return w_ex