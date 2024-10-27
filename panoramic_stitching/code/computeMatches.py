import numpy as np

# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 4

def computeMatches(f1, f2):
    """
    For features f1 (dxN), f2 (dxM), return an array matches of shape (N) such that
    matches[i] is the closest feature in f2 for feature i.
    """
    matches = []

    for i in range(f1.shape[1]):
        argmin = 0
        min_value = float('inf')
        for j in range(f2.shape[1]):
            val = np.linalg.norm(f1[:, i] - f2[:, j], ord=2)
            if val < min_value:
                min_value = val
                argmin = j
        matches.append(argmin)

    return np.array(matches)
