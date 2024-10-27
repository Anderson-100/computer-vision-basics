import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from nms import nms

from utils import gaussian

mpl.use('tkagg')

def detectCorners(I, is_simple, w, th):
    I = I.astype(float)
    if I.ndim > 2:
        I = rgb2gray(I)

    if is_simple:
        corner_score = simple_score(I, w)
    else:
        corner_score = harris_score(I, w)
    plt.figure(2)
    plt.imshow(corner_score)
    plt.axis('off')
    if is_simple:
        plt.savefig('heatmap_simple.jpg')
    else:
        plt.savefig('heatmap_harris.jpg')
    corner_score[corner_score < th] = 0

    cx, cy, cs = nms(corner_score)
    return np.array([cx, cy, cs])


def simple_score(I, w):
    gs = gaussian(6*w+1, w)
    f = []

    f.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]))
    f.append(np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]]))
    f.append(np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]]))
    f.append(np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]]))
    f.append(np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]))
    f.append(np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]]))
    f.append(np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]]))
    f.append(np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]))

    corner_score = np.zeros_like(I)
    for i in range(8):
        diff = convolve(I, f[i], mode='nearest')
        diff_sum = convolve(diff**2, gs, mode='nearest')
        corner_score += diff_sum
    return corner_score


def harris_score(I, w):
    gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    gy = gx.T

    imgx = convolve(I, gx, mode='nearest')
    imgy = convolve(I, gy, mode='nearest')

    gs = gaussian(6*w+1, w)
    
    imgx2 = convolve(imgx**2, gs, mode='nearest')
    imgy2 = convolve(imgy**2, gs, mode='nearest')
    imgxgy = convolve(imgx*imgy, gs, mode='nearest')
    corner_score = (imgx2 * imgy2 - imgxgy**2) - 0.04*(imgx2 + imgy2)**2

    return corner_score
