import numpy as np
from scipy.ndimage.filters import maximum_filter
from skimage.morphology import local_maxima
from skimage.feature import peak_local_max

def nms(corner_score):

    #Pick local maxima in a window
    locs = peak_local_max(corner_score, 5)
    #Get scores at maximum peaks
    scores = corner_score[locs[:, 0], locs[:, 1]]
    #Sort scores
    idx_sorted = np.argsort(scores)
    #from IPython import embed; embed(); exit(-1)

    #Get scores and coordinates according to sorting
    cs = scores[idx_sorted]
    cx = locs[idx_sorted, 1]
    cy = locs[idx_sorted, 0]

    return cx, cy, cs
