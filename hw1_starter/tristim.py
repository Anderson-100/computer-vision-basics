import numpy as np

def tristim_b_1(F1, F2, F3, Sr, Sg, Sb):
    """
    Write a Python function to compute the matrix R given F1, F2, F3, 
    Sr, Sg, and Sb. Give the value of the matrix R.
    """
    # TODO fill this in
    R = np.zeros((3,3))

    F = [F1, F2, F3]
    S = [Sr, Sg, Sb]

    for i, s in enumerate(S):
        for j, f in enumerate(F):
            R[i,j] = np.sum(f * s)
    return R

def tristim_b_2(R, C):
    """
    Write a Python function, which, given the matrix R and a set of desired color responses in the form of
    a vector C calculates the proper brightness multiplier weights b and returns them as a single vector of
    3 values
    """
    # TODO fill this in
    return np.linalg.inv(R).dot(C)


if __name__ == '__main__':
    # Flashlight spectrum
    F1 = np.array([0.08, 0.15, 0.25, 0.29, 0.13, 0.05, 0.02, 0.01, 0.00, 0.00])
    F2 = np.array([0.00, 0.04, 0.08, 0.10, 0.15, 0.20, 0.18, 0.14, 0.06, 0.00])
    F3 = np.array([0.00, 0.00, 0.00, 0.00, 0.01, 0.04, 0.15, 0.25, 0.37, 0.17])

    # Eye absortion theory
    Sr = np.array([0.16, 0.26, 0.28, 0.15, 0.10, 0.03, 0.02, 0.00, 0.00, 0.00])
    Sg = np.array([0.00, 0.03, 0.06, 0.20, 0.31, 0.21, 0.15, 0.03, 0.01, 0.00])
    Sb = np.array([0.00, 0.00, 0.00, 0.00, 0.01, 0.04, 0.08, 0.23, 0.35, 0.29])

    # Solution to the first part
    R = tristim_b_1(F1, F2, F3, Sr, Sg, Sb)
    print('Part 1 solution:\n', R)

    # Matching colors
    C_yg = np.array([0.508, 0.907, 0.803])
    C_teal = np.array([0.307, 0.609, 0.646])

    print('Part 2 yellow-green:', tristim_b_2(R, C_yg))
    print('Part 2 teal:', tristim_b_2(R, C_teal))
