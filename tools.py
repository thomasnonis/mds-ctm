import os
import cv2


def wpsnr_to_mark(wpsnr):
	if wpsnr >= 35 and wpsnr < 50:
		return 1
	if wpsnr >= 50 and wpsnr < 54:
		return 2
	if wpsnr >= 54 and wpsnr < 58:
		return 3
	if wpsnr >= 58 and wpsnr < 62:
		return 4
	if wpsnr >= 62 and wpsnr < 66:
		return 5
	if wpsnr >= 66:
		return 6
	return 0

def generate_watermark(mark_size):
	import numpy as np
	# Generate a watermark
	mark = np.random.uniform(0.0, 1.0, mark_size)
	mark = np.uint8(np.rint(mark))
	np.save('mark.npy', mark)
	return mark

# list_of_images: [(watermarked,"Watermarked"),(attacked,"Attacked"),...]
def show_images(list_of_images, rows, columns):
	import matplotlib.pyplot as plt
	
	plt.figure(figsize=(15, 6))
	for (i,(image,label)) in enumerate(list_of_images):

		plt.subplot(rows,columns,i+1)
		plt.title(list_of_images[i][1])
		plt.imshow(list_of_images[i][0], cmap='gray')
	plt.show()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

