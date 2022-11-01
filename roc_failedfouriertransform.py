import os
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

from config import *
from attacks import *
from tools import import_images, multiprocessed_workload, update_parameters

from detection_failedfouriertransform import similarity, extract_watermark, wpsnr
from embedment_failedfouriertransform import embed_watermark

def compute_ROC(scores, labels, alpha, show: bool = True):
    # compute ROC
    fpr, tpr, thr = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    # compute AUC
    roc_auc = auc(fpr, tpr)
    if show is True:
        plt.figure()
        lw = 2

        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) with alpha {str(alpha)}')
        plt.legend(loc="lower right")
        plt.show()
    idx_tpr = np.where((fpr - TARGET_FPR) == min(i for i in (fpr - TARGET_FPR) if i > 0))
    print('For a FPR approximately equals to %0.2f corresponds a TPR equal to %0.2f and a threshold equal to %0.4f with FPR equal to %0.2f' % (TARGET_FPR, tpr[idx_tpr[0][0]], thr[idx_tpr[0][0]], fpr[idx_tpr[0][0]]))

    return thr[idx_tpr[0][0]], tpr[idx_tpr[0][0]], fpr[idx_tpr[0][0]] # return thr

def generate_watermark(size_h: int, size_v: int = 0, save: bool = False) -> np.ndarray:
    """Generates a random watermark of size (size_h, size_v)

    Generates a random watermark of size (size_h, size_v) if size_v is specified,
    otherwise generates a square watermark of size (size_h, size_h)

    Args:
        size_h (int): Horizontal size
        size_v (int, optional): Vertical size. Defaults to size_h.

    Returns:
        np.ndarray: Random watermark of the desired size
    """
    if size_v == 0:
        size_v = size_h

    # Generate a watermark
    mark = np.random.uniform(0.0, 1.0, size_v * size_h)
    mark = np.uint8(np.rint(mark))
    if save is True:
        np.save('mark.npy', mark)
    return mark.reshape((size_v, size_h))


def compute_thr_multiple_images(images, original_watermark, attacks, alpha, level, subband, show: bool = True):
    scores = []
    labels = []

    n_images = len(images)
    i = 0
    m = 0
    attack_idx = 0
    model_name = '_'.join([str(alpha),str(level),'-'.join(subband)])
    if exists_model(model_name):
        scores, labels, _, _, _ = read_model(model_name)

    n_computations = n_images * RUNS_PER_IMAGE * N_FALSE_WATERMARKS_GENERATIONS
    print('Total number of computations: %d' % n_computations)
    
    # step by step for clarity
    for original_img, watermarked_img, img_name in images:
        for j in range(attack_idx, attack_idx+RUNS_PER_IMAGE):
            attacked_img, attacks_list = do_attacks(watermarked_img, attacks[j])
            
            while wpsnr(original_img, attacked_img) < 35:
                print("Retry", attacks_list, "was too powerful")
                attack = get_random_attacks(randint(MIN_N_ATTACKS, MAX_N_ATTACKS))
                attacked_img, attacks_list = do_attacks(watermarked_img, attack)
            
            # 3. Extract the watermark with your planned technique Wextracted
            extracted_watermark = extract_watermark(original_img, img_name, attacked_img, alpha, level, subband)
            
            # 4. Compute sim(Woriginal,Wextracted) and append it in the scores array and the value 1 in the labels array. These values will correspond to the true positive hypothesis.
            # true positive population
            scores.append(similarity(original_watermark, extracted_watermark))
            labels.append(1)

            # perform multiple comparisons with random watermarks to better train the classifier against false positives
            # 5. Generate a random watermark Wrandom and compute sim(Wrandom,Wextracted) to append it in the scores array and the value 0 in the labels array. These values will correspond to the true negative hypothesis.
            for k in range(0, N_FALSE_WATERMARKS_GENERATIONS):
                print('{}/{} - Performed attack {}/{} on image {}/{} ({}) - false check {}/{} - attacks: {}'.format(m + 1, n_computations, j%RUNS_PER_IMAGE, RUNS_PER_IMAGE, i + 1, n_images, img_name, k + 1, N_FALSE_WATERMARKS_GENERATIONS, attacks_list))
                # true negative population
                fake_mark = generate_watermark(32)
                scores.append(similarity(fake_mark, extracted_watermark))
                labels.append(0)

                m += 1
        i += 1
        attack_idx += RUNS_PER_IMAGE
    # 6. with scores and labels, generate the ROC and choose the best threshold τ corresponding to a False Positive Rate FPR ∈ [0, 0.1].
    return scores,labels,compute_ROC(scores, labels, alpha, show)

def save_model(scores: list, labels: list, threshold: float, tpr: float, fpr: float, alpha: int, level: int, subband: list) -> None:
    """Saves the model trained models/model_<alpha>_<level>_<subband>.txt
    The scores and label are saved too in case we want to continue training

    Args:
        scores (list): Scores list
        labels (list): Labels list
        threshold (float): The threshold
        tpr (float): The true positive rate
        fpr (float): The false positive rate
        alpha (float): The alpha used for embedding
        level (int): The level used for embedding
        subband (list): The subband(s) used for embedding
    """
    directory = 'models/'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    
    params = '_'.join([str(alpha),str(level),'-'.join(subband)])
    f = open(directory + 'model_' + params, 'wb')
    pickle.dump((scores, labels, threshold, tpr, fpr), f, protocol=2)
    f.close()


def read_model(name: str) -> None:
    """Loads a model from a file

    Args:
        name (str): Name of the model to be loaded
    """
    f = open('models/model_' + name, 'rb')
    values = list(pickle.load(f))

    (scores, labels, threshold, tpr, fpr) = (values[0], values[1], values[2], values[3], values[4])
    f.close()
    return scores, labels, threshold, tpr, fpr


def exists_model(name: str) -> None:
    """Checks if a model exists

    Args:
        name (str): Name of the model to be checked
    """
    return os.path.exists('models/model_' + name)

def create_model(params, order_of_execution):
    from roc_failedfouriertransform import compute_thr_multiple_images
    from embedment_failedfouriertransform import embed_watermark
    watermarked_images = []
    images, watermark, alpha, level, subband, attacks, show_threshold = params
    for original_img, img_name in images:
        watermarked_img = embed_watermark(original_img, img_name, watermark, alpha, level, subband)
        watermarked_images.append((original_img, watermarked_img, img_name))

    scores, labels, (threshold, tpr, fpr) = compute_thr_multiple_images(watermarked_images, watermark, attacks, alpha, level, subband, show_threshold)
    
    # We should do this only once, when we know what the good parameters are, or remove it entirely
    update_parameters('detection_failedfouriertransform.py', ALPHA = alpha, DWT_LEVEL = level, SUBBANDS = subband, DETECTION_THRESHOLD = threshold, MARK_SIZE = MARK_SIZE)
    
    save_model(scores, labels, threshold, tpr, fpr, alpha, level, subband)
    return order_of_execution, threshold, tpr, fpr, alpha,level,subband

def print_models():
    # Sometimes this crashes because it can not find the file. Don't know why
    for alpha in range(10,30,2):
        for level in [2]:
            for subband in [["LL"] , ["HL", "LH"]]:
                alpha = str(int(alpha))
                level = str(level)
                subband = "-".join(subband)
                (scores, labels, threshold, tpr, fpr) = read_model(alpha + '_' + level + '_' + subband)
                tpr = round(tpr,2)
                fpr = round(fpr,2)
                threshold = round(threshold,2)

                print((alpha + '_' + level + '_' + subband).ljust(10), tpr, fpr, threshold)

def threshold_computation():
    images = import_images(IMG_FOLDER_PATH, N_IMAGES_LIMIT, True)
    watermark = np.load("failedfouriertransform.npy").reshape((MARK_SIZE, MARK_SIZE))
    attacks = []
    # 2. In a loop, attack one by one these images (with random attacks or the strategy you prefer)
    for _ in images:
        for _ in range(0, RUNS_PER_IMAGE):
            attacks.append(get_random_attacks(randint(MIN_N_ATTACKS, MAX_N_ATTACKS)))
    work = []
    show_threshold = False
    for alpha in range(10,30,2):
        for level in [2]:
            for subband in [["LL"], ["HL", "LH"]]:
                work.append((images, watermark, alpha, level, subband, attacks, show_threshold))
    result = multiprocessed_workload(create_model, work)
    print(result)

if __name__ == '__main__':
    threshold_computation()
    print_models()