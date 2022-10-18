from time import time
from collections import defaultdict

from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *

import numpy as np
import math
from scipy import signal

def normxcorr2D(image, template):
    """
    Normalized cross-correlation for 2D PIL images
    Inputs:
    ----------------
    template    The template. A PIL image.  Elements cannot all be equal.
    image       The PIL image.
    Output:
    ----------------
    nxcorr      Array of cross-correlation coefficients, in the range
                -1.0 to 1.0.
                Wherever the search space has zero variance under the template,
                normalized cross-correlation is undefined.
    Implemented for CPSC 425 Assignment 3
    Bob Woodham
    January, 2013
    """

    # (one-time) normalization of template
    t = np.asarray(template, dtype=np.float64)
    t = t - np.mean(t)
    norm = math.sqrt(np.sum(np.square(t)))
    t = t / norm

    # create filter to sum values under template
    sum_filter = np.ones(np.shape(t))

    # get image
    a = np.asarray(image, dtype=np.float64)
    #also want squared values
    aa = np.square(a)

    # compute sums of values and sums of values squared under template
    a_sum = signal.correlate2d(a, sum_filter, 'same')
    aa_sum = signal.correlate2d(aa, sum_filter, 'same')
    # Note:  The above two lines could be made more efficient by
    #        exploiting the fact that sum_filter is separable.
    #        Even better would be to take advantage of integral images

    # compute correlation, 't' is normalized, 'a' is not (yet)
    numer = signal.correlate2d(a, t, 'same')
    # (each time) normalization of the window under the template
    denom = np.sqrt(aa_sum - np.square(a_sum)/np.size(t))

    # wherever the denominator is near zero, this must be because the image
    # window is near constant (and therefore the normalized cross correlation
    # is undefined). Set nxcorr to zero in these regions
    tol = np.sqrt(np.finfo(denom.dtype).eps)
    nxcorr = np.where(denom < tol, 0, numer/denom)

    # if any of the coefficients are outside the range [-1 1], they will be
    # unstable to small variance in a or t, so set them to zero to reflect
    # the undefined 0/0 condition
    nxcorr = np.where(np.abs(nxcorr-1.) > np.sqrt(np.finfo(nxcorr.dtype).eps),nxcorr,0)

    return nxcorr

results = defaultdict(dict)

# Note sharpen is not equal to what described in the paper
attacks_list = [
    {
        'function' : jpeg_compression,
        'arguments' : {
            "QF" : 95
        },
        'description' : 'JPEG ({})'.format(95)
    },
    {
        'function': gaussian_blur, 
        'arguments': {
            'sigma': [3,3]
        }, 
        'description': 'Gaussian Blur (3,3)'
    },
    {
        'function': resize, 
        'arguments': {
            'scale': 0.5
        }, 
        'description': 'Resize (0.5)'
    },    
    {
        'function' : sharpen,
        'arguments' : {
            "sigma" : 0.2,
            "alpha": 0.2
        },
        'description' : 'Sharpen ({}, {})'.format(0.2, 0.2)
    }
]

img_name = "lena.bmp"
original_img = cv2.imread(IMG_FOLDER_PATH + img_name, cv2.IMREAD_GRAYSCALE)
watermark = generate_watermark(MARK_SIZE)

results["original_img"] = original_img



alpha_range = np.arange(0.1, 0.5, 0.1) * DEFAULT_ALPHA
for alpha in alpha_range:
    alpha = int(alpha)
    watermarked_img = embed_watermark(original_img, img_name+"_"+str(alpha), watermark, alpha)
    results["watermarked"][str(alpha)] = watermarked_img
print("Finished watermarking")

for attack in attacks_list:
    att_lst = [(attack)]
    results["attacked"][describe_attacks(att_lst)] = {}
    for alpha in alpha_range:
        alpha = int(alpha)
        attacked_img, _ = do_random_attacks(results["watermarked"][str(alpha)],att_lst)
        results["attacked"][attack["description"]][alpha] = attacked_img
print("Finished attacking")


plt.figure()
lw = 2

'''
ys = []
for watermarked in results["watermarked"].values():
    ys.append(wpsnr(results['original_img'], watermarked)) 
plt.plot(alpha_range, ys, lw=lw, label="Signed")
'''

for attack in attacks_list:
    ys = []
    print(attack["description"])
    for attacked_image in results["attacked"][attack["description"]].values():
        #ys.append(wpsnr(results['original_img'], attacked_image)) 
        ys.append(psnr(results['original_img'], np.uint8(attacked_image)))
    plt.plot(alpha_range, ys, lw=lw, label=attack["description"])

plt.xlabel('Alpha')
plt.ylabel('WPSNR')
plt.title('WPSNR comparison of attacked images with a watermark embedded and with different levels of alpha')
plt.legend(loc="upper right")
plt.show()


plt.figure()
for attack in attacks_list:
    ys = []
    print(attack["description"])
    for attacked_image,alpha in zip(results["attacked"][attack["description"]].values(), alpha_range):
        extracted_watermark = extract_watermark(results['original_img'],img_name+"_"+str(int(alpha)),attacked_image)
        
        ys.append(normxcorr2D(watermark, extracted_watermark).mean())
    
    print(len(ys))
    plt.plot(alpha_range, ys, lw=lw, label=attack["description"])

plt.xlabel('Alpha')
plt.ylabel('NCC')
plt.title('WPSNR comparison of attacked images with a watermark embedded and with different levels of alpha')
plt.legend(loc="upper right")
plt.show()

#attacked_img, attacks_list = do_random_attacks(watermarked_img,attacks_list)
#extracted_watermark = extract_watermark(original_img, img_name, attacked_img)
