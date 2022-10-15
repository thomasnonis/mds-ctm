import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import measurements
from embedding import detection_dwt_svd
from config import *
from tools import *
from attacks import *

def compute_thr_multiple_images(w_images, img_name, watermark, alpha, mark_size):
    ext_watermarks = []
    scores = []
    labels = []

    (_, alpha, svd_key) = read_parameters(img_name)
    # step by step for clarity
    for w_image in w_images:
        a_image = attack(w_image[0])
        w_ext = detection_dwt_svd(w_image[0], a_image, DEFAULT_ALPHA, DWT_LEVEL, mark_size, svd_key)

        # store extracted watermarks
        ext_watermarks.append(w_ext)

    # true positive population
    for i in range(len(ext_watermarks)):
        scores.append(measurements.similarity(watermark, ext_watermarks[i]))
        labels.append(1)

    # true negative population
    for i in range(len(ext_watermarks)):
        scores.append(measurements.similarity(generate_watermark(mark_size), ext_watermarks[i]))
        labels.append(0)

    return compute_ROC(scores, labels)

def attack(w_image):
    return w_image #jpeg_compression(w_image, 50)

# Check if correct
def compute_ROC(scores, labels):
    # compute ROC
    fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    # compute AUC
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    idx_tpr = np.where((fpr - 0.05) == min(i for i in (fpr - 0.05) if i > 0))
    print('For a FPR approximately equals to 0.05 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.05 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])
    return tau[idx_tpr[0][0]] # return thr