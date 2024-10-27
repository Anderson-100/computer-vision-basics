import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import sys

from runDemosaicing import *
from utils import *

#Path to your data directory
data_dir = os.path.join('..', 'data', 'demosaic')

#Path to your output directory
out_dir = os.path.join('..', 'output', 'demosaic')
mkdir(out_dir)

image_name = 'puppy.jpg'
display = False

imgpath = os.path.join(data_dir, image_name)
color_img = imread(imgpath)

channels = ['red', 'green', 'blue']

for i, color in enumerate(channels):
    shifted = np.roll(color_img[:, :, i], 1, axis=1)
    shifted_flattened = shifted.flatten()
    print(shifted_flattened.shape)
    orig_flattened = color_img[:, :, i].flatten()

    plt.figure()
    plt.scatter(orig_flattened, shifted_flattened, s=1, color=color)
    # plt.plot(np.arange(len(orig_flattened)), orig_flattened, label='Original Pixel')
    plt.title(f'{color} current pixel vs. pixel to right')
    plt.ylabel('Intensity of pixel to the right')
    plt.xlabel('Intensity of current pixel')

    outfile_path = os.path.join(out_dir, '{}-graph.png'.format(color))
    plt.savefig(outfile_path)