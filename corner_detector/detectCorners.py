import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from nms import nms
import matplotlib.pyplot as plt

from utils import gaussian


def detectCorners(I, is_simple, w, th):
#Convert to float
    I = I.astype(float)

    #Convert color to grayscale
    if I.ndim > 2:
        I = rgb2gray(I)

    # Step 1: compute corner score
    if is_simple:
        corner_score = simple_score(I, w)
    else:
        corner_score = harris_score(I, w)

    # Plot as color map
    # plt.imshow(corner_score)
    # plt.savefig(f"checker_simple{is_simple}.png")

    # Step 2: Threshold corner score and find peaks
    corner_score[corner_score < th] = 0

    cx, cy, cs = nms(corner_score)
    return cx, cy, cs


#--------------------------------------------------------------------------
#                                    Simple score function (Implement this)
#--------------------------------------------------------------------------
def simple_score(I, w):
    # Please use convolve with mode="nearest" in your implementation
    # Use utils.gaussian with hsize = 6*sigma + 1
    corner_score = np.zeros_like(I)
    for u in range(-1, 2):
        for v in range(-1, 2):

            # Skip if center
            if u == 0 and v == 0:
                continue

            # Build filter matrix (aka f(u,v)) for current direction
            filter_matrix = np.zeros((3,3))
            filter_matrix[1, 1] = -1
            filter_matrix[1+u, 1+v] = 1

            image_diff = np.square(convolve(I, filter_matrix, mode="nearest"))
            image_diff = convolve(image_diff, gaussian(hsize=6*w + 1, sigma=w), mode="nearest")
            corner_score = corner_score + image_diff

    return corner_score


#--------------------------------------------------------------------------
#                                    Harris score function (Implement this)
#--------------------------------------------------------------------------
def harris_score(I, w):
    # Please use convolve with mode="nearest" in your implementation
    # Use utils.gaussian with hsize = 6*sigma + 1
    grad_x_filter = [[-1, 0, 1]]
    grad_y_filter = [[-1], [0], [1]]
    Ix = convolve(I, grad_x_filter, mode="nearest")
    Iy = convolve(I, grad_y_filter, mode="nearest")
    
    gauss = gaussian(hsize=6*w + 1, sigma=w)

    a = convolve(np.square(Ix), gauss, mode="nearest")
    b = convolve(Ix*Iy, gauss, mode="nearest")
    c = b.copy()
    d = convolve(np.square(Iy), gauss, mode="nearest")

    k = 0.04
    corner_score = (a*d - b*c) - k * np.square(a + d)
    
    return corner_score
