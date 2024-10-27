import numpy as np
import matplotlib.pyplot as plt 
from utils import imread
from scipy.ndimage import gaussian_filter, median_filter

def gaussian(img, sigma):
    """Apply Gaussian filtering to img with sigma=sigma and output the resulting image."""
    edited = gaussian_filter(img.copy(), sigma)
    return edited

def optimal_sigma():
    """
    Please return the optimal values of sigma that you found for both noisy images.
    Your output should be a tuple (sigma1, sigma2) 
    where sigma1 is optimal for noise1 and sigma2 is optimal for noise2.
    """
    return (1.3, 1.6)

def median(img, neighborhood):
    """Apply median filtering to img with a given neighborhood size and output the resulting image."""
    edited = median_filter(img.copy(), neighborhood)
    return edited

def optimal_neighborhood():
    """
    Please return the optimal values of neighborhood that you found for both noisy images.
    Your output should be a tuple (neighborhood1, neighborhood2) 
    where neighborhood1 is optimal for noise1 and neighborhood2 is optimal for noise2.
    """
    return (5, 3)

def experiment(img, noise1, noise2):
    """
    This function will not be graded! 
    This is for you to experiment and find the best values of each parameter/image pair. 
    Run/print whatever you'd like!
    """
    print("Use this space to run your experiments!")

    # Gaussian 
    sigma1, sigma2 = optimal_sigma()
    gauss_noise1 = gaussian(noise1.copy(), sigma1)
    error1 = SE(img, gauss_noise1)
    gauss_noise2 = gaussian(noise2.copy(), sigma2)
    error2 = SE(img, gauss_noise2)
    print('Gaussian Input, Errors: {:.2f} {:.2f}'.format(error1, error2))

    # Median
    neighborhood1, neighborhood2 = optimal_neighborhood()
    median_noise1 = median(noise1.copy(), neighborhood1)
    error3 = SE(img, median_noise1)
    median_noise2 = median(noise2.copy(), neighborhood2)
    error4 = SE(img, median_noise2)
    print('Median Input, Errors: {:.2f} {:.2f}'.format(error3, error4))


def plot_denoise(orig_img, noise1, error1, noise2, error2):
    plt.figure(1)

    plt.subplot(131)
    plt.imshow(orig_img, cmap="gray")
    plt.title('Input')

    plt.subplot(132)
    plt.imshow(noise1, cmap="gray")
    plt.title('SE {:.2f}'.format(error1))

    plt.subplot(133)
    plt.imshow(noise2, cmap="gray")
    plt.title('SE {:.2f}'.format(error2))

    plt.show()

def SE(x, y):
    return ((x - y)**2).sum()

if __name__=="__main__":
    img = imread('../data/peppers.png')
    noise1 = imread('../data/peppers_g.png')
    noise2 = imread('../data/peppers_sp.png')

    error1 = SE(img, noise1)
    error2 = SE(img, noise2)
    print('Input, Errors: {:.2f} {:.2f}'.format(error1, error2))
    plot_denoise(img, noise1, error1, noise2, error2)

    experiment(img, noise1, noise2)

    # Getting optimal parameter values
    sigma1, sigma2 = optimal_sigma()
    n1, n2 = optimal_neighborhood()

    # Filtering images
    opt1_gauss = gaussian(noise1.copy(), sigma1)
    opt1_median = median(noise1.copy(), n1)
    opt2_gauss = gaussian(noise2.copy(), sigma2)
    opt2_median = median(noise2.copy(), n2)
    plot_denoise(noise1, opt1_gauss, SE(img, opt1_gauss), opt1_median, SE(img, opt1_median))
    plot_denoise(noise2, opt2_gauss, SE(img, opt2_gauss), opt2_median, SE(img, opt2_median))
