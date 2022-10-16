from collections import defaultdict
from time import time
from datetime import datetime

from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *
import numpy as np
import random

start_time = time()
print('Starting...')

# Load images
images = import_images(IMG_FOLDER_PATH)
n_images = min(len(images), N_IMAGES_LIMIT) # set to a lower number to limit the number of images to process
watermark = generate_watermark(MARK_SIZE)

random.shuffle(images) # Shuffle list of images
show_WPSNR = False
show_similarity = False
show_WPSNR_attacked_images = False

watermarked_images = defaultdict(dict)

alpha_range = np.arange(0.1, 2, 0.1) * DEFAULT_ALPHA

for original_img, img_name in images[:n_images]:  # n_images
    watermarked_images[img_name]["original_img"] = original_img
    watermarked_images[img_name]["watermarked_images"] = {}
    print('Loading image %s' % img_name)
    for alpha in alpha_range: 
        watermarked_img = embed_watermark(original_img, img_name, watermark, int(alpha))
        watermarked_images[img_name]["watermarked_images"][img_name+"_"+str(int(alpha))] = watermarked_img


wpsnr_plot = plt.figure()
lw = 2

for image_name in watermarked_images:
    ys = []
    for watermarked_image_name in watermarked_images[image_name]["watermarked_images"]:
        watermarked_img = watermarked_images[image_name]["watermarked_images"][watermarked_image_name]
        ys.append(wpsnr(watermarked_images[image_name]["original_img"], watermarked_img))
    
    
    plt.plot(alpha_range, ys, lw=lw, label=image_name)

end_time = time()
print('Time WPSNR: ',end_time-start_time)
start_time = time()
plt.xlabel('Alpha')
plt.ylabel('WPSNR')
plt.title('WPSNR comparison of images with a watermark embedded and with different levels of alpha')
plt.legend(loc="upper right")
plt.savefig('WPSNR-Alpha.png')
if show_WPSNR:
    plt.show()

plt.figure()
lw = 2

attacks_list = get_random_attacks(randint(1, MAX_N_ATTACKS))
print('The following attack will be done to images: ',describe_attacks(attacks_list))
for image_name in watermarked_images:
    ys = []
    for watermarked_image_name in watermarked_images[image_name]["watermarked_images"]:
        watermarked_img = watermarked_images[image_name]["watermarked_images"][watermarked_image_name]
        attacked_img, _ = do_random_attacks(watermarked_img,attacks_list)
        extracted_watermark = extract_watermark(original_img, img_name, attacked_img)
        ys.append(similarity(watermark, extracted_watermark))

        watermarked_images[image_name]["watermarked_images"][watermarked_image_name] = attacked_img # Overwrite the watermarked image with the attacked one for later
    plt.plot(alpha_range, ys, lw=lw, label=image_name)

end_time = time()
print('Time to attack: ',end_time-start_time)

plt.xlabel('Alpha')
plt.ylabel('Similarity')
plt.title('Similarity of extracted watermark and original watermark of attacked images')
plt.legend(loc="upper right")
plt.savefig('Similarity-Alpha.png')
if show_similarity:
    plt.show()


plt.figure()
lw = 2

for image_name in watermarked_images:
    ys = []
    for watermarked_image_name in watermarked_images[image_name]["watermarked_images"]:
        watermarked_img = watermarked_images[image_name]["watermarked_images"][watermarked_image_name]
        # Should this be (watermarked_image,attacked_watermarked_image) ? 
        ys.append(wpsnr(watermarked_images[image_name]["original_img"], watermarked_img)) # watermarked image is in reality the attacked watermarked image
    
    
    plt.plot(alpha_range, ys, lw=lw, label=image_name)

end_time = time()
print('Time WPSNR Attacked: ',end_time-start_time)
start_time = time()
plt.xlabel('Alpha')
plt.ylabel('WPSNR Attacked')
plt.title('WPSNR comparison of attacked images with different levels of alpha')
plt.legend(loc="upper right")
plt.savefig('Attacked-WPSNR-Alpha.png')
if show_WPSNR_attacked_images:
    plt.show()
