import numpy as np

def demosaicImage(image, method):
    ''' Demosaics image.

    Args:
        img: np.array of size NxM.
        method: demosaicing method (baseline or nn).

    Returns:
        Color image of size NxMx3 computed using method.
    '''

    if method.lower() == "baseline":
        return demosaicBaseline(image.copy())
    elif method.lower() == 'nn':
        return demosaicNN(image.copy()) # Implement this
    else:
        raise ValueError("method {} unkown.".format(method))


def demosaicBaseline(img):
    '''Baseline demosaicing.
    
    Replaces missing values with the mean of each color channel.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of sieze NxMx3 demosaiced using the baseline 
        algorithm.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    red_values = img[0:image_height:2, 0:image_width:2]
    mean_value = red_values.mean()
    mos_img[:, :, 0] = mean_value
    mos_img[0:image_height:2, 0:image_width:2, 0] = img[0:image_height:2, 0:image_width:2]

    blue_values = img[1:image_height:2, 1:image_width:2]
    mean_value = blue_values.mean()
    mos_img[:, :, 2] = mean_value
    mos_img[1:image_height:2, 1:image_width:2, 2] = img[1:image_height:2, 1:image_width:2]

    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_values = mos_img[mask > 0]
    mean_value = green_values.mean()

    green_channel = img
    green_channel[mask < 0] = mean_value
    mos_img[:, :, 1] = green_channel

    return mos_img


def demosaicNN(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of size NxMx3 demosaiced using the nearest neighbor 
        algorithm.
    '''
    height, width = img.shape
    mos_img = np.zeros((height, width, 3))
    # print(img.shape)
    # print(mos_img.shape)

    # Even rows: R G R G
    # Odd rows:  G B G B
    for i in range(height):
        for j in range(width):
            # Red
            if i % 2 == 0 and j % 2 == 0:
                # Set red from current pixel
                mos_img[i, j, 0] = img[i, j]

                # Nearest green
                mos_img[i, j, 1] = nearestGreen(img, i, j)

                # Nearest blue
                mos_img[i, j, 2] = nearestBlue(img, i, j)

            # Blue
            elif i % 2 != 0 and j % 2 != 0:
                # Set blue from current pixel
                mos_img[i, j, 2] = img[i, j]

                # Nearest red
                mos_img[i, j, 0] = nearestRed(img, i, j)

                # Nearest green
                mos_img[i, j, 1] = nearestGreen(img, i, j)

            # Green
            else:
                # Set green from current pixel
                mos_img[i, j, 1] = img[i, j]

                # Nearest red
                mos_img[i, j, 0] = nearestRed(img, i, j)

                # Nearest blue
                mos_img[i, j, 2] = nearestBlue(img, i, j)

    # print(mos_img)

    return mos_img

def nearestRed(img, i, j):
    # R G R G row, G pixel
    # Default to picking from the left (will always have red to left)
    if i % 2 == 0:
        return img[i, j-1]

    # On a row that does NOT have red pixels (G B G B row)
    # This is the case where we're on a GREEN pixel of this row
    # There will be a red pixel directly above and below
    # Default to picking from above
    elif j % 2 == 0:
        if i == 0:
            return img[i+1, j]
        else:
            return img[i-1, j]

    # On a row that does NOT have red pixels (G B G B row)
    # This is the case where we're on a BLUE pixel of this row
    # Default to up and left
    # If no up, do down and left
    # If no left, do up and right
    # If no up or left, do down and right
    else:
        if i == 0 and j == 0: # down right
            return img[i+1, j+1]

        elif i == 0: # down left
            return img[i+1, j-1]

        elif j == 0: # up right
            return img[i-1, j+1]

        else:
            return img[i-1, j-1]

def nearestBlue(img, i, j):
    # G B G B row
    # This hits when we reach a G in this row
    # Default to picking from left, except when j = 0
    if i % 2 != 0:
        if j == 0:
            return img[i, j+1]
        else:
            return img[i, j-1]

    # R G R G row, G pixel
    # Default up 
    elif j % 2 == 0:
        if i == 0:
            return img[i+1, j]
        else:
            return img[i-1, j]

    # R G R G row, R pixel
    else:
        # down right
        if i == 0 and j == 0:
            return img[i+1, j+1]

        # down left
        elif i == 0:
            return img[i+1, j-1]

        # up right
        elif j == 0:
            return img[i-1, j+1]

        # up left
        else:
            return img[i-1, j-1]

def nearestGreen(img, i, j):
    # R G R G row, R pixel
    # Default to picking from left
    if i % 2 == 0:
        if j == 0:
            return img[i, j+1]
        else:
            return img[i, j-1]

    # G B G B row, B pixel
    else:
        return img[i, j-1]