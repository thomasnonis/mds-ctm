import pyswarms as ps
import multiprocessing
import cv2 as cv
from random import randint
import time

from attacks import gaussian_blur, average_blur, sharpen, median, resizing, awgn, jpeg_compression
from detection_failedfouriertransform import wpsnr, extract_watermark, similarity, detection, ALPHA, DWT_LEVEL, DETECTION_THRESHOLD
from tools import show_images
from config import *

N_ITERATIONS = 33
N_PARTICLES = 30
N_DIMENSIONS = 15
N_PARALLEL_PROCESSES = 12
PENALTY = 1000
ENABLE_VERBOSE = False

ENABLE_GAUSSIAN_BLUR = True
ENABLE_AVERAGE_BLUR = True
ENABLE_SHARPEN = True
ENABLE_MEDIAN = True
ENABLE_RESIZING = True
ENABLE_AWGN = True
ENABLE_JPEG_COMPRESSION = True

MIN_BOUND_GAUSSIAN_BLUR_SIGMA = 0.1
MIN_BOUND_AVERAGE_BLUR_KERNEL_SIZE = 3
MIN_BOUND_SHARPEN_SIGMA = 0.1
MIN_BOUND_SHARPEN_ALPHA = 0.1
MIN_BOUND_MEDIAN_KERNEL_SIZE = 3
MIN_BOUND_RESIZING_SCALE = 0.2
MIN_BOUND_AWGN_STD_DEV = 0.1
MIN_BOUND_JPEG_QF = 0

MAX_BOUND_GAUSSIAN_BLUR_SIGMA = 5
MAX_BOUND_AVERAGE_BLUR_KERNEL_SIZE = 5
MAX_BOUND_SHARPEN_SIGMA = 10
MAX_BOUND_SHARPEN_ALPHA = 10
MAX_BOUND_MEDIAN_KERNEL_SIZE = 8
MAX_BOUND_RESIZING_SCALE = 1
MAX_BOUND_AWGN_STD_DEV = 50
MAX_BOUND_JPEG_QF = 101

def detection_nopath(original_img, watermarked_img, attacked_img):
    original_watermark = extract_watermark(original_img, watermarked_img, ALPHA, DWT_LEVEL, SUBBANDS)
    attacked_watermark = extract_watermark(original_img, attacked_img, ALPHA, DWT_LEVEL, SUBBANDS)

    if similarity(original_watermark, attacked_watermark) > DETECTION_THRESHOLD:
        has_watermark = 1
    else:
        has_watermark = 0

    return has_watermark, wpsnr(watermarked_img, attacked_img)

def objective_function(args, **kwargs):
    args = args.T
    original_img = kwargs['original_image']
    watermarked_img = kwargs['watermarked_image']
    do_gaussian_blur = args[0]
    do_average_blur = args[1]
    do_sharpen = args[2]
    do_median = args[3]
    do_resizing = args[4]
    do_awgn = args[5]
    do_jpeg_compression = args[6]

    gaussian_blur_sigma = args[7]
    average_blur_kernel_size = args[8]
    sharpen_sigma = args[9]
    sharpen_alpha = args[10]
    median_kernel_size = args[11]
    resizing_scale = args[12]
    awgn_std_dev = args[13]
    jpeg_compression_qf = args[14]


    ret = []
    for i in range(args.shape[1]):
        # print(args.T[i])
        penalty = 0
        attacked_img = watermarked_img.copy()

        if do_gaussian_blur[i] > 0.5:
            if ENABLE_GAUSSIAN_BLUR:
                try:
                    attacked_img = gaussian_blur(attacked_img, gaussian_blur_sigma[i])
                except Exception as e:
                    penalty += PENALTY
                    print('An error occurred during gaussian_blur({}) and a penalty has been applied: {}'.format(gaussian_blur_sigma[i], e))
            else:
                penalty += PENALTY

        if do_average_blur[i] > 0.5:
            if ENABLE_AVERAGE_BLUR:
                if int(average_blur_kernel_size[i]) not in [3, 5, 7]:
                    penalty += PENALTY
                else:
                    attacked_img = average_blur(attacked_img, int(average_blur_kernel_size[i]))
            else:
                penalty += PENALTY

        if do_sharpen[i] > 0.5:
            if ENABLE_SHARPEN:
                try:
                    attacked_img = sharpen(attacked_img, sharpen_sigma[i], sharpen_alpha[i])
                except Exception as e:
                    penalty += PENALTY
                    print('An error occurred during sharpen({}, {}) and a penalty has been applied: {}'.format(sharpen_sigma[i], sharpen_alpha[i], e))
            else:
                penalty += PENALTY

        if do_median[i] > 0.5:
            if ENABLE_MEDIAN:
                if int(median_kernel_size[i]) not in [3, 5, 7]:
                    penalty += PENALTY
                else:
                    attacked_img = median(attacked_img, int(median_kernel_size[i]))
            else:
                penalty += PENALTY

        if do_resizing[i] > 0.5:
            if ENABLE_RESIZING:
                try:
                    attacked_img = resizing(attacked_img, resizing_scale[i])
                except Exception as e:
                    penalty += PENALTY
                    print('An error occurred during resizing({}) and a penalty has been applied: {}'.format(resizing_scale[i], e))
            else:
                penalty += PENALTY

        if do_awgn[i] > 0.5:
            if ENABLE_AWGN:
                try:
                    seed = randint(0, 1000)
                    attacked_img = awgn(attacked_img, awgn_std_dev[i], seed)
                except Exception as e:
                    penalty += PENALTY
                    print('An error occurred during awgn({}, {}) and a penalty has been applied: {}'.format(awgn_std_dev[i], seed, e))
            else:
                penalty += PENALTY

        if do_jpeg_compression[i] > 0.5:
            if ENABLE_JPEG_COMPRESSION:
                try:
                    attacked_img = jpeg_compression(attacked_img, int(jpeg_compression_qf[i]))
                except Exception as e:
                    penalty += PENALTY
                    print('An error occurred during jpeg_compression({}) and a penalty has been applied: {}'.format(int(jpeg_compression_qf[i]), e))
            else:
                penalty += PENALTY
            

        has_watermark, wpsnr = detection_nopath(original_img, watermarked_img, attacked_img)
        if has_watermark == 1:
            penalty += PENALTY
        
        if penalty == 0:
            ret.append(100 / abs(wpsnr))
        else:
            ret.append(penalty)


    return ret

def print_results(cost, pos):
    print('========== RESULTS ==========')
    print('WPSNR: ', 100/cost)
    print('DO GAUSSIAN BLUR: ', 'True' if pos[0] > 0.5 else 'False')
    print('DO AVERAGE BLUR: ', 'True' if pos[1] > 0.5 else 'False')
    print('DO SHARPEN: ', 'True' if pos[2] > 0.5 else 'False')
    print('DO MEDIAN: ', 'True' if pos[3] > 0.5 else 'False')
    print('DO RESIZING: ', 'True' if pos[4] > 0.5 else 'False')
    print('DO AWGN: ', 'True' if pos[5] > 0.5 else 'False')
    print('DO JPEG COMPRESSION: ', 'True' if pos[6] > 0.5 else 'False')
    if pos[0] > 0.5:
        print('GAUSSIAN BLUR SIGMA: ', pos[7])

    if pos[1] > 0.5:
        print('AVERAGE BLUR KERNEL SIZE: ', int(pos[8]))

    if pos[2] > 0.5:
        print('SHARPEN SIGMA: ', pos[9])
        print('SHARPEN ALPHA: ', pos[10])
    
    if pos[3] > 0.5:
        print('MEDIAN KERNEL SIZE: ', int(pos[11]))

    if pos[4] > 0.5:
        print('RESIZING SCALE: ', pos[12])
    
    if pos[5] > 0.5:
        print('AWGN STD DEV: ', pos[13])

    if pos[6] > 0.5:
        print('JPEG COMPRESSION QF: ', int(pos[14]))

def run_best_attack(attacked_img, args):
    do_gaussian_blur = args[0]
    do_average_blur = args[1]
    do_sharpen = args[2]
    do_median = args[3]
    do_resizing = args[4]
    do_awgn = args[5]
    do_jpeg_compression = args[6]

    gaussian_blur_sigma = args[7]
    average_blur_kernel_size = args[8]
    sharpen_sigma = args[9]
    sharpen_alpha = args[10]
    median_kernel_size = args[11]
    resizing_scale = args[12]
    awgn_std_dev = args[13]
    jpeg_compression_qf = args[14]

    if do_gaussian_blur > 0.5:
        attacked_img = gaussian_blur(attacked_img, gaussian_blur_sigma)

    if do_average_blur > 0.5:
        attacked_img = average_blur(attacked_img, int(average_blur_kernel_size))

    if do_sharpen > 0.5:
        attacked_img = sharpen(attacked_img, sharpen_sigma, sharpen_alpha)

    if do_median > 0.5:
        attacked_img = median(attacked_img, int(median_kernel_size))

    if do_resizing > 0.5:
        attacked_img = resizing(attacked_img, resizing_scale)

    if do_awgn > 0.5:
        attacked_img = awgn(attacked_img, awgn_std_dev, randint(0, 1000))

    if do_jpeg_compression > 0.5:
        attacked_img = jpeg_compression(attacked_img, int(jpeg_compression_qf))

    return attacked_img

if __name__ == '__main__':
    start_time = time.time()
    original_img = cv.imread('images/to-watermark/Lena.bmp', cv.IMREAD_GRAYSCALE)
    watermarked_img = cv.imread('images/watermarked/lena.bmp', cv.IMREAD_GRAYSCALE)
    # print(img)
    # Set-up hyperparameters
    #'c1': 0.5, 'c2': 0.3, 'w':0.9
    c1 = 1.1
    c2 = 0.7
    w = 0.9
    assert (w > -1 and w < 1 and (c1 + c2) < ((24*(1 - (w * w)))/(7 - (5 * w)))), 'Invalid PSO options. The algorithm will not converge.'
    options = {'c1': c1, 'c2': c2, 'w':w}

    min_bounds = [0, 0, 0, 0, 0, 0, 0, MIN_BOUND_GAUSSIAN_BLUR_SIGMA, MIN_BOUND_AVERAGE_BLUR_KERNEL_SIZE, MIN_BOUND_SHARPEN_SIGMA, MIN_BOUND_SHARPEN_ALPHA, MIN_BOUND_MEDIAN_KERNEL_SIZE, MIN_BOUND_RESIZING_SCALE, MIN_BOUND_AWGN_STD_DEV, MIN_BOUND_JPEG_QF]
    max_bounds = [1, 1, 1, 1, 1, 1, 1, MAX_BOUND_GAUSSIAN_BLUR_SIGMA, MAX_BOUND_AVERAGE_BLUR_KERNEL_SIZE, MAX_BOUND_SHARPEN_SIGMA, MAX_BOUND_SHARPEN_ALPHA, MAX_BOUND_MEDIAN_KERNEL_SIZE, MAX_BOUND_RESIZING_SCALE, MAX_BOUND_AWGN_STD_DEV, MAX_BOUND_JPEG_QF]
    bounds = (min_bounds, max_bounds)

    print('Running Optimization with {} iterations and {} particles'.format(N_ITERATIONS, N_PARTICLES))
    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=N_PARTICLES, dimensions=N_DIMENSIONS, options=options, bounds=bounds)
    # Perform optimization
    cost, pos = optimizer.optimize(objective_function, iters=N_ITERATIONS, n_processes=min(multiprocessing.cpu_count(), N_PARALLEL_PROCESSES), verbose=ENABLE_VERBOSE, original_image=original_img, watermarked_image=watermarked_img)

    print_results(cost, pos)

    end_time = time.time()

    
    # print('========== RUNNING BEST ATTACK ==========')

    # attacked_img = run_best_attack(watermarked_img, pos)

    # has_watermark, wpsnr = detection_nopath(original_img, watermarked_img, attacked_img)

    # print('WPSNR: ', wpsnr)
    # print('Has watermark: ', has_watermark)
    
    print('Execution time: {}s'.format(round((end_time - start_time), 1)))

    # show_images([(original_img, 'Original Image'), (watermarked_img, 'Watermarked Image'), (attacked_img, 'Attacked Image')], 1, 3)