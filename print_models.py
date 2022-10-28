import numpy as np
import pickle
from tools import read_model

alpha_range = np.arange(0.5, 1, 0.2) * 23
beta_range = np.arange(0.01, 0.2+0.1, 0.04)

for alpha in alpha_range:
    for beta in beta_range:
        alpha = round(alpha,2)
        beta = round(beta,2)
        (scores, labels, threshold, tpr, fpr, params) = read_model(str(alpha) + '_' + str(beta))
        tpr = round(tpr,2)
        fpr = round(fpr,2)
        threshold = round(threshold,2)
        print(str(alpha) + '_' + str(beta), tpr, fpr, threshold)

print("="*10)
alpha_range = np.arange(0.1, 0.4, 0.1) * 250
for alpha in alpha_range:
    for level in [2-1,2,2+1]:
        for subband in [["LL"], ["HL","LH"]]:
            alpha = str(int(alpha))
            level = str(level)
            subband = "-".join(subband)
            (scores, labels, threshold, tpr, fpr, params) = read_model(alpha + '_' + level + '_' + subband)
            tpr = round(tpr,2)
            fpr = round(fpr,2)
            threshold = round(threshold,2)

            print((alpha + '_' + level + '_' + subband).ljust(12), tpr, fpr, threshold)

