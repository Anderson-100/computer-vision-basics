import numpy as np
from skimage.transform import rescale
# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 4

def mergeImages(im1, im2, transf):
    s = transf[2]
    tx = np.round(transf[0]/s)
    ty = np.round(transf[1]/s)

    im2 = rescale(im2, 1/s, channel_axis=2)
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    #Coordinates of the four corners
    c1 = np.array([[0, w1, w1, 0],[0, 0, h1, h1]]).astype(int)
    c2 = np.array([[0, w2, w2, 0],[0, 0, h2, h2]]).astype(int)

    #Transformed coordinates
    #from IPython import embed; embed(); exit(-1)
    c2[0, :] -= int(tx)
    c2[1, :] -= int(ty)

    #Get new bounds
    xmin = min(0, np.min(c2[0, :]))
    xmax = max(w1, np.max(c2[0, :]))
    ymin = min(0, np.min(c2[1, :]))
    ymax = max(h1, np.max(c2[1, :]))

    im = np.zeros((ymax-ymin+1, xmax-xmin+1, 3))
    ox = max(0, 1-xmin)
    oy = max(0, 1-ymin)
    #from IPython import embed; embed(); exit(-1)
    im[oy:oy+h1, ox:ox+w1, :] = im1
    im[oy+c2[1, 0]:oy+c2[1, 3], ox+c2[0, 0]:ox+c2[0, 1], :] = im2
    return im

