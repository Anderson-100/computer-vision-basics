import numpy as np
import matplotlib.pyplot as plt
import utils
from skimage.color import rgb2gray
from scipy.ndimage import convolve

def imageGradient(img, eps=1e-10):
    """
    Return the magnitude and angle of the gradient at each pixel in the image.
    Please use mode="nearest" in your convolution.
    You may have divide by zero issues when doing arctan(gy/gx) so please use arctan(gy/(gx + eps)) instead.
    """
    fx = np.array(
        [
            [0, 0, 0],
            [-1, 0, 1],
            [0, 0, 0]
        ]
    )
    fy = np.array(
        [
            [0, -1, 0],
            [0, 0, 0],
            [0, 1, 0]
        ]
    )

    gx = convolve(img, fx, mode="nearest")
    gy = convolve(img, fy, mode="nearest")

    m = np.sqrt(np.square(gx) + np.square(gy))
    a = np.arctan(gy / (gx + eps))
    return m, a

def histogram(m, a):
    """
    Create a histogram on the distribution of gradients based on magnitude m and angle a.
    Bin the angles into 9 bins, assign each pixel to a bin, then do a sum of the pixels in each bin weighted by magnitude.
    """
    bins = [-90, -70, -50, -30, -10, 10, 30, 50, 70, 90]
    return np.histogram(np.rad2deg(a), bins=bins, weights=m)[0]

def smooth(img, sigma=2):
    """
    Use utils.gaussian with hsize = 6*sigma + 1
    The imported convolve function will also be helpful.
    """
    gaussian = utils.gaussian(hsize=6*sigma + 1, sigma=sigma)
    smoothed = convolve(img, gaussian, mode='nearest')
    return smoothed

def vis_grad(img, m, a, aHist):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.title('Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(m, cmap=plt.get_cmap('gray'))
    plt.title('Gradient magnitude')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(a, cmap=plt.get_cmap('viridis'))
    plt.colorbar()
    plt.title('Gradient angle')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.bar(np.arange(9), aHist.tolist())
    plt.title('Gradient histogram')
    plt.xlabel('Bin')
    plt.ylabel('Total magnitude')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    img = rgb2gray(utils.imread('../data/parrot.jpg'))

    m, a = imageGradient(img.copy())
    aHist = histogram(m, a)
    vis_grad(img, m, a, aHist)

    # Same thing but with smoothing
    m, a = imageGradient(smooth(img.copy()))
    aHist = histogram(m, a)
    vis_grad(img, m, a, aHist)
