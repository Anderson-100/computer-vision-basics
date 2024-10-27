import numpy as np

def mosaicImage(img):
    ''' Computes the mosaic of an image.

    mosaicImage computes the response of the image under a Bayer filter.

    Args:
        img: NxMx3 numpy array (image).

    Returns:
        NxM image where R, G, B channels are sampled according to RGRG in the
        top left.
    '''

    image_height, image_width, num_channels = img.shape
    assert(num_channels == 3) #Checks if it is a color image

    mosaiced_image = np.zeros_like(img[:,:,0])

    # Even rows: R G R G
    # Odd Rows: G B G B

    for row in range(image_height):
        for col in range(image_width):
            
            # R G R G row
            if row % 2 == 0:
                if col % 2 == 0: # red
                    mosaiced_image[row, col] = img[row, col, 0]
                else: # blue
                    mosaiced_image[row, col] = img[row, col, 1]

            # G B G B row
            else:
                if col % 2 == 0: # green
                    mosaiced_image[row, col] = img[row, col, 1]
                else: # blue
                    mosaiced_image[row, col] = img[row, col, 2]

    return mosaiced_image
