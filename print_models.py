import numpy as np
import pickle
from tools import read_model
from config import DWT_LEVEL
alpha_range = [10,20,40,60]
beta_range = [0.005,0.01,0.1,0.2,0.3,0.4,0.5,0.6] 

for alpha in alpha_range:
    for beta in beta_range:
        (scores, labels, threshold, tpr, fpr, params) = read_model(str(alpha) + '_' + str(beta))
        tpr = round(tpr,2)
        fpr = round(fpr,2)
        threshold = round(threshold,2)
        print(str(alpha) + '_' + str(beta), tpr, fpr, threshold)

print("="*10)
alpha_range = [25,50,75,100]
for alpha in alpha_range:
    for level in [DWT_LEVEL-1,2,DWT_LEVEL+1]:
        for subband in [["LL"], ["HL","LH"]]:
            alpha = str(int(alpha))
            level = str(level)
            subband = "-".join(subband)
            (scores, labels, threshold, tpr, fpr, params) = read_model(alpha + '_' + level + '_' + subband)
            tpr = round(tpr,2)
            fpr = round(fpr,2)
            threshold = round(threshold,2)

            print((alpha + '_' + level + '_' + subband).ljust(12), tpr, fpr, threshold)

print("="*10)

alpha_range = [25, 50, 60, 75, 250] 
for alpha in alpha_range:
    for level in [DWT_LEVEL - 1, DWT_LEVEL, DWT_LEVEL + 1]:
        for subband in [["LL"], ["HL", "LH"]]:
            alpha = str(int(alpha))
            level = str(level)
            subband = "-".join(subband)
            (scores, labels, threshold, tpr, fpr, params) = read_model('dct_' + alpha + '_' + level + '_' + subband)
            tpr = round(tpr,2)
            fpr = round(fpr,2)
            threshold = round(threshold,2)

            print(('dct' + '_' + alpha + '_' + level + '_' + subband).ljust(15), tpr, fpr, threshold)


