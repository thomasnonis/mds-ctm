from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *

def main():
	# Get one random image
	original_img, img_name = import_images(IMG_FOLDER_PATH,1,True)[0]

	# Generate watermark
	watermark = generate_watermark(MARK_SIZE)

	alpha = DEFAULT_ALPHA
	subband = DEFAULT_SUBBAND
	level = DWT_LEVEL
	watermarked_img = embed_watermark(original_img, img_name, watermark, alpha, level, subband)

	attack = get_random_attacks(randint(1, MAX_N_ATTACKS))
	attacked_img, _ = do_random_attacks(watermarked_img,attack)
	extracted_watermark = extract_watermark(original_img, img_name, attacked_img, alpha, level, subband)
	sim = similarity(watermark, extracted_watermark)
	_wpsnr = wpsnr(original_img, attacked_img)
	print(_wpsnr, sim)

	coeffs = wavedec2d(original_img, level)
	
	attacked_subband, _ = do_random_attacks(coeffs[0],attack)

	coeffs[0] = attacked_subband

	lmao = waverec2d(coeffs)
	extracted_watermark = extract_watermark(original_img, img_name, lmao, alpha, level, subband)
	sim = similarity(watermark, extracted_watermark)
	_wpsnr = wpsnr(original_img, lmao)
	print(_wpsnr, sim)
	show_images([(attacked_img,"Attacked"),(lmao, "LMAO")],1,2)


if __name__ == '__main__':
	main()