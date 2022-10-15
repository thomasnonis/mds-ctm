from scipy.fft import dct, idct
import numpy as np
from math import sqrt,log,e
from transforms import dct2d,idct2d,dwt2d,idwt2d, wavedec2d, waverec2d
import pywt

def embedding_dct(original_img, mark, alpha, v='multiplicative'):
    # Get the DCT transform of the original_img
    ori_dct = dct2d(original_img) 

    # Get the locations of the most perceptually significant components
    sign = np.sign(ori_dct)
    ori_dct = abs(ori_dct)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = original_img.shape[0]
    locations = [(val//rows, val%rows) for val in locations] # locations as (x,y) coordinates

    # Embed the watermark
    watermarked_img_dct = ori_dct.copy()
    for idx, (loc,mark_val) in enumerate(zip(locations[1:], mark)):
        if v == 'additive':
            watermarked_img_dct[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_img_dct[loc] *= 1 + ( alpha * mark_val)
        elif v == 'exponential':
            watermarked_img_dct[loc] *= e**(alpha*mark_val)

    # Restore sign and o back to spatial domain
    watermarked_img_dct *= sign
    watermarked_img = np.uint8(idct2d(watermarked_img_dct))

    return watermarked_img

def detection_dct(original_img, watermarked_img, alpha, mark_size, v='multiplicative'):
    ori_dct = dct2d(original_img)
    wat_dct = dct2d(watermarked_img)

    # Get the locations of the most perceptually significant components
    ori_dct = abs(ori_dct)
    wat_dct = abs(wat_dct)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = original_img.shape[0]
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

def detection_dwt(original_img, watermarked_img, alpha, level, mark_size):
    original_coeffs = wavedec2d(original_img, level)
    original_LL = original_coeffs[0]

    watermarked_coeffs = wavedec2d(watermarked_img, level)
    watermarked_LL = watermarked_coeffs[0]

    # Initialize the watermark matrix
    watermark = np.zeros([mark_size, mark_size], dtype=np.float64)

    # Extract the watermark
    for i in range(0,mark_size):
        for j in range(0,mark_size):
            watermark[i][j] = (watermarked_LL[i][j] - original_LL[i][j]) / alpha

    return watermark

def detection_dwt_svd(original_img, watermarked_img, alpha, level, mark_size, original_s_ll_d_u, original_s_ll_d_v):
    original_coeffs = wavedec2d(original_img, level)
    original_LL = original_coeffs[0]

    original_ll_u, original_ll_s, original_ll_v = np.linalg.svd(original_LL)
    
    original_ll_s = np.diag(original_ll_s)
    watermarked_coeffs = wavedec2d(watermarked_img, level)
    watermarked_LL = watermarked_coeffs[0]

    watermarked_ll_u, watermarked_ll_s, watermarked_ll_v = np.linalg.svd(watermarked_LL)
    watermarked_ll_s = np.diag(watermarked_ll_s)
    
    # original_s_ll_d_u, original_s_ll_d_s, original_s_ll_d_v 
    s_ll_d = np.matmul(np.matmul(original_s_ll_d_u,watermarked_ll_s),original_s_ll_d_v)

    # Initialize the watermark matrix
    watermark = np.zeros([mark_size, mark_size], dtype=np.float64)
    
    # Extract the watermark
    for i in range(0,mark_size):
        for j in range(0,mark_size):
            watermark[i][j] = (s_ll_d[i][j] - original_ll_s[i][j]) / alpha

    return watermark