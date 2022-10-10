from scipy.fft import dct, idct, fft2, ifft2
from pywt import dwt2, idwt2

def dct2d(img):
	return dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2d(img):
	return idct(idct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def fft2d(img):
	return fft2(img)

def ifft2d(img):
	return ifft2(img)

def dwt2d(img):
	return dwt2(img, 'haar')

def idwt2d(img):
	return idwt2(img, 'haar')