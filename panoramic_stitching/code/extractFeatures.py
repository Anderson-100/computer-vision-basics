import numpy as np
from skimage.color import rgb2gray

def extractFeatures(im, c, patch_radius):
    '''
    Extract patches of size (2 * patch_radius + 1) ** 2
    around the coordinates given in c. Use rgb2gray to convert
    the image to grayscale first.
    '''
    if im.ndim > 2:
        im = rgb2gray(im)

    num_corners = c.shape[1]

    # pad image to accomodate for corners near the edge of image
    padded_im = np.pad(im, patch_radius)
    
    output_dims = ((2*patch_radius+1)**2, num_corners)
    f = np.zeros(output_dims)

    # Calculate patch features for each corner
    for i in range(num_corners):
        cx = int(c[0, i] + patch_radius)
        cy = int(c[1, i] + patch_radius)

        # First dim is y, second dim is x for arrays
        patch = padded_im[cy-patch_radius : cy+patch_radius+1, cx-patch_radius : cx+patch_radius+1]
        patch = patch.flatten()
        f[:, i] = patch

    return f
