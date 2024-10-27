# This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2024
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Homework 5

import pickle
import numpy as np
import matplotlib.pyplot as plt

def visualizeNeighbors(imgs, topk_idxs, topk_distances, title):
    '''
    Visualize the query image as well as its nearest neighbors
    Input:
        imgs: a list or numpy array, with length k+1.
            imgs[0] is the query image with shape hxw
            imgs[k] is k-th nearest neighbor image
        topk_idxs: a list or numpy array, with length k+1.
            topk_idxs[k] is the index in training set of the k-th nearest image 
            topk_idxs[0] is the query image index in the test set
        topk_distances: a list or numpy array, with length k+1.
            topk_idxs[k] is the distance of the k-th nearest image to the query
            topk_idxs[0] is 0
    '''
    n = len(imgs)
    fig, axs = plt.subplots(1, n, figsize=(2 * n, 3))
    fig.suptitle(title)
    np.set_printoptions(precision=2)
    for k in range(n):
        if k == 0:
            ax_title = f'query: test_idx={topk_idxs[0]}' 
        else:
            ax_title = '%d: idx=%d, d=%0.2f' % (k, topk_idxs[k], topk_distances[k])
        axs[k].set_title(ax_title)
        axs[k].imshow(imgs[k], cmap='gray')
        axs[k].axis('off')
    fig.tight_layout()
    plt.show()

    np.set_printoptions()

def getDistances(x_train, x_test):
    """
    Return a matrix distances of shape (x_test.shape[2], x_train.shape[2]) where
    distances[i, j] = the l2 distance from x_test[j] to x_train[i].
    """
    distances = np.zeros((x_test.shape[2], x_train.shape[2]))
    # TODO Implement this
    for i in range(x_test.shape[2]):
        for j in range(x_train.shape[2]):
            distances[i,j] = np.linalg.norm(x_test[:,:,i] - x_train[:,:,j])
    return distances

def sortDistances(distances):
    """
    Given a matrix distances from getDistances, return an array nearest_idxs with the same shape as distances
    where nearest_idxs[i, j] is the index of the j-th closest sample in x_train to
    the i-th sample in x_test.

    Hint: look up np.argsort and use kind="stable". 
    """
    # TODO Implement this
    nearest_idxs = np.argsort(distances, axis=1, kind="stable")
    return nearest_idxs

def callVis(x_train, x_test, distances, nearest_idxs, k, images_to_display):
    """
    Visualize the top-k matches for the test images listed in images_to_display, sorted by distance.
    """
    imgs = np.empty((k+1, 28, 28))
    topk_idxs = [0] * (k+1)
    topk_distances = [0] * (k+1)
    for test_i in images_to_display:
        #------------------------------------------------------------------
        # Prepare imgs, topk_idxs and topk_distances
        #------------------------------------------------------------------
        # TODO Implement this
        topk_idxs[0] = test_i
        imgs[0, :, :] = x_test[:, :, test_i]

        # Add test image data as first entry
        topk_idxs.append(test_i)

        for idx in range(k):
            nearest_image = nearest_idxs[test_i, idx]
            topk_idxs[idx+1] = nearest_image
            topk_distances[idx + 1] = distances[test_i, nearest_image]
            imgs[idx+1, :, :] = x_train[:, :, nearest_image]
		
        visualizeNeighbors(imgs, topk_idxs, topk_distances, 
            title=f'Test img {test_i}: Top {k} Neighbors')

def knnAccuracy(x_train, y_train, x_test, y_test, nearest_idxs, k):
    """
    Return the accuracy of a knn classifier on the test set with the given k and nearest_idxs.
    """
    acc = 0
    # TODO Implement this
    for i in range(len(y_test)):
        fives = 0
        nines = 0
        for idx in nearest_idxs[i, :k]:
            if y_train[idx] == 5:
                fives += 1
            else:
                nines += 1
        
        if fives > nines and y_test[i] == 5:
            acc += 1
        elif nines > fives and y_test[i] == 9:
            acc += 1

    return acc / len(y_test)

if __name__ == "__main__":
	data = pickle.load(open('data.pkl','rb'))
	x_train = np.asarray(data['train']['x'])
	y_train = np.asarray(data['train']['y'])
	x_test = np.asarray(data['test']['x'])
	y_test = np.asarray(data['test']['y'])

	distances = getDistances(x_train, x_test)
	nearest_idxs = sortDistances(distances)
	callVis(x_train, x_test, distances, nearest_idxs, 5, [10, 20, 110, 120])

	k_list = [1, 3, 5, 7, 9]
	for k in k_list:
		acc = knnAccuracy(x_train, y_train, x_test, y_test, nearest_idxs, k)
		print(f'k={k}: accuracy={acc*100:.2f}')
	