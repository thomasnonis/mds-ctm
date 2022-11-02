from detection_failedfouriertransform import detection
from embedment_failedfouriertransform import embed_watermark
from tools import import_images, show_images # Remove show images before commtting
from config import *
import numpy as np
import cv2

if __name__ == '__main__':
    images = import_images(IMG_FOLDER_PATH,N_IMAGES_LIMIT,True)
    watermark = np.load("failedfouriertransform.npy").reshape((MARK_SIZE, MARK_SIZE))
    
    TEAM_NAME = 'failedfouriertransform'
    alpha = 16
    level = 2
    for image in images:
        original_img, img_name = image
        
        watermarked_img = embed_watermark(original_img, img_name, watermark, alpha, level, SUBBANDS)

        original_img_path = IMG_FOLDER_PATH + img_name + '.bmp'
        watermarked_img_path = 'images/watermarked/'+img_name + '_' + TEAM_NAME + '.bmp'
        
        # Save the original image as "watermarked" to test if we find the watermark in a non watermarked image
        cv2.imwrite(watermarked_img_path, watermarked_img)
        has_watermark, wpsnr =  detection(original_img_path, watermarked_img_path, watermarked_img_path)
        assert has_watermark == True, 'Did not find watermark in watermarked image'

        has_watermark, wpsnr =  detection(original_img_path, watermarked_img_path, original_img_path)
        assert has_watermark == False, 'Did find watermark in non watermarked image'

        
        
