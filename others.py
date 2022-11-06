"""
1. funzioni che leggono le immagini con watermark degli altri gruppi
2. funzione che carichi le funzioni di detection degli altri gruppi e le renda utilizzabili nel nostro codice

pixel
ef26420c
you_shall_not_mark
blitz
omega
howimetyourmark
weusedlsb
thebavarians
theyarethesamepicture
dinkleberg
"""

import os
from config import *
from tools import save_image
import importlib
import cv2

groups = ['pixel', 'ef26420c', 'youshallnotmark', 'blitz', 'omega', 'howimetyourmark', 'weusedlsb', 'thebavarians', 'theyarethesamepicture', 'dinkleberg']

# watermarked images (downloaded from the website) naming convention: groupB_imageName.bmp
def retrieve_others_images(groupName: str, img_folder: str, typ: str):
    path = img_folder + groupName + '/' + typ +  '/'

    if not os.path.isdir(path):
        exit('Error: Images folder not found')

    images_name = os.listdir(path)
    images = []
    for img_name in images_name:
        images.append((cv2.imread(path + img_name, cv2.IMREAD_GRAYSCALE), path + img_name))

    return images




"""
    Simple import of the detection function of other groups.
    To use it: 
        mod = import_others_detection(groupname)
        tr, w = mod.detection(original, watermarked, attacked)
"""
def import_others_detection(groupName: str):
    toImport = "detection_" + groupName
    mod = importlib.import_module(toImport, toImport)

    return mod


"""
# TEST
img = retrieve_watermarked_images('blitz', IMG_FOLDER_PATH)
save_image(img[0][0], img[0][1], 'attacked', 'weusedlsb')

mod = import_others_detection('failedfouriertransform')
print(mod.detection('images/failedfouriertransform/watermarked/' + img[0][1] + '.bmp', 'images/failedfouriertransform/watermarked/' + img[0][1] + '.bmp', 'images/failedfouriertransform/watermarked/' + img[0][1] + '.bmp'))
"""
