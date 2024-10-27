import numpy as np
import time

def ransac(matches, c1, c2, max_iter=400, threshold=5):
    """
    Run max_iter iterations of RANSAC, which will randomly sample 2 matches
    and compute s, t, then evaluate the number of inliers with distance < threshold.
    Return (inliers, transf) where transf is the transformation with the highest number of inliers.
    """
    best_inliers = []
    best_transf = []
    for i in range(max_iter):
        # Grab necessary values to calculate s and tx and ty
        idx1 = np.random.randint(0, len(matches))
        x1 = c1[0, idx1]
        y1 = c1[1, idx1]
        
        j1 = matches[idx1]
        x1_prime = c2[0, j1]
        y1_prime = c2[1, j1]

        idx2 = np.random.randint(0, len(matches))
        x2 = c1[0, idx2]
        y2 = c1[1, idx2]
        
        j2 = matches[idx2]
        x2_prime = c2[0, j2]
        y2_prime = c2[1, j2]

        # Calculate s
        s = np.sqrt((x1_prime - x2_prime)**2 + (y1_prime - y2_prime)**2) / np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if s == 0:
            continue
        # Calculate t
        tx = x2_prime - s*x2
        ty = y2_prime - s*y2

        # Use s and t values to determine the number of inliers
        cur_inliers = []
        for m in range(len(matches)):
            second_im_idx = matches[m]
            x1_second_im = c2[0, second_im_idx]
            y1_second_im = c2[1, second_im_idx]
            Tx_prime = (x1_second_im - tx) / s
            Ty_prime = (y1_second_im - ty) / s

            x1_first_im = c1[0, m]
            y1_first_im = c1[1, m]

            dist = (x1_first_im - Tx_prime)**2 + (y1_first_im - Ty_prime)**2

            if dist < threshold:
                cur_inliers.append(m)

        # Save the transformation with the longest list of inliers
        if len(cur_inliers) > len(best_inliers):
            best_inliers = cur_inliers.copy()
            best_transf = [tx, ty, s]

    return [best_inliers, best_transf]
