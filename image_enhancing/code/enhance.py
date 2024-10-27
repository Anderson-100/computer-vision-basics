import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from utils import *

def histogram(im):
    pdf = np.zeros(256)
    cdf = np.zeros(256)

    for i in range(len(im)):
        for j in range(len(im[0])):
            pixel_value = im[i, j]
            pdf[pixel_value] += 1

    curSum = 0
    for i in range(len(pdf)):
        curSum += pdf[i]
        cdf[i] = curSum

    return pdf, cdf

def contrast_stretch(img):
    print(type(img))
    pdf, cdf = histogram(img)

    min_pixel_value = 0
    max_pixel_value = 255

    # Find the min pixel value
    for i in range(len(pdf)):
        if pdf[i] > 0:
            min_pixel_value = i
            break

    # Find the max pixel value
    for j in range(len(pdf)-1, -1, -1):
        if pdf[j] > 0:
            max_pixel_value = j
            break

    # Calculate new pixel values
    h = lambda v : (v - min_pixel_value) / (max_pixel_value - min_pixel_value) * 255
    con_str = np.round(h(img)).astype(int)
    return con_str

def adjust_gamma(image, gamma=1.0):
    h = lambda v : (v / 255)**(1 / gamma) * 255
    img_gamma = np.round(h(image)).astype(int)
    return img_gamma

def histogram_eqalization(img):
    pdf, cdf = histogram(img)

    # Find the min pixel value in cdf
    min_pixel_value = 0
    for i in range(len(cdf)):
        if cdf[i] > 0:
            min_pixel_value = i
            break

    height, width = img.shape
    h = lambda v : (cdf[v] - min_pixel_value) / (height * width - min_pixel_value) * 255
    equ = np.round(h(img)).astype(int)
    return equ

def plot_hist(img, name):
	pdf, cdf = histogram(img)
	plt.figure(1, figsize=(10, 4))

	plt.subplot(121)
	plt.bar(range(256), pdf)
	plt.xlabel("Intensity value")
	plt.ylabel("Number of pixels")
	plt.title("Intensity PDF")
	plt.tight_layout()
                   
	plt.subplot(122)
	plt.plot(cdf, color = 'b')
	plt.xlabel("Intensity value")
	plt.ylabel("Number of pixels")
	plt.title("Intensity CDF")
	plt.tight_layout()
	plt.savefig(name)
	plt.show()
	plt.close()

if __name__=='__main__':
	#Path to your data directory
	data_dir = os.path.join('..', 'data', 'contrast')

	#Path to your output directory
	out_dir = os.path.join('..', 'output', 'contrast')
	mkdir(out_dir)
	img = cv2.imread(os.path.join(data_dir, 'forest.png'), 0)
	plt.imshow(img, vmin=0, vmax=255, cmap='gray')
	plt.show()
	# Plotting histogram
	plot_hist(img, os.path.join(out_dir, 'forest-hist-loop.pdf'))

	# Contrast stretching
	con_str = contrast_stretch(img)
	cv2.imwrite(os.path.join(out_dir, 'con_str.png'), con_str)
	plt.imshow(con_str, vmin=0, vmax=255, cmap='gray')
	plt.show()
	plot_hist(con_str, os.path.join(out_dir, 'con_str_hist.pdf'))

	# Gamma adjustment
	img_gamma = adjust_gamma(img, gamma=0.5)
	cv2.imwrite(os.path.join(out_dir, 'gamma_05.png'), img_gamma)
	plt.imshow(img_gamma, vmin=0, vmax=255, cmap='gray')
	plt.show()
	plot_hist(img_gamma, os.path.join(out_dir, 'gamma_05_hist.pdf'))

	img_gamma = adjust_gamma(img, gamma=2)
	cv2.imwrite(os.path.join(out_dir, 'gamma_2.png'), img_gamma)
	plt.imshow(img_gamma, vmin=0, vmax=255, cmap='gray')
	plt.show()
	plot_hist(img_gamma, os.path.join(out_dir, 'gamma_2_hist.pdf'))

	img_gamma = adjust_gamma(con_str, gamma=1.5)
	cv2.imwrite(os.path.join(out_dir, 'con_str_gamma_1.5.png'), img_gamma)
	plt.imshow(img_gamma, vmin=0, vmax=255, cmap='gray')
	plt.show()
	plot_hist(img_gamma, os.path.join(out_dir, 'con_str_gamma_1.5_hist.pdf'))

	# Histogram equalization
	equ = histogram_eqalization(img)
	cv2.imwrite(os.path.join(out_dir, 'equ.png'), equ)
	plt.imshow(equ, vmin=0, vmax=255, cmap='gray')
	plt.show()
	plot_hist(equ, os.path.join(out_dir, 'equ_hist.pdf'))
