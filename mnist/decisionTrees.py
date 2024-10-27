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


def scoreFeatures(x, y):
    '''
    Calculate the number of examples that can be correctly classified solely by each pixel.
    Return the (scores, max_idxs, zero_is_five), where scores is the score for each pixel, max_idxs is the (i, j) 
    which gives the highest score, and zero_is_five is True if predicting 5 if pixel(i, j) == 0 is better.
    '''
    scores = np.zeros(x[:, :, 0].shape)
    max_score = 0
    max_idxs = np.array([0, 0])
    zero_is_five = False
    # TODO Implement this
    for cur_zero_is_five in [True, False]:
        for i in range(len(scores)):
            for j in range(len(scores[0])):
                pixel_score = 0
                for image in range(len(x[0][0])):
                    if zero_is_five:

                        # Correct
                        if x[i, j, image] == 0 and y[image] == 5:
                            pixel_score += 1
                        elif x[i, j, image] == 1 and y[image] == 9:
                            pixel_score += 1
                            
                    else:
                        # Correct
                        if x[i, j, image] == 0 and y[image] == 9:
                            pixel_score += 1
                        elif x[i, j, image] == 1 and y[image] == 5:
                            pixel_score += 1

                if pixel_score > max_score:
                    max_score = pixel_score
                    max_idxs = np.array([i, j])
                    zero_is_five = cur_zero_is_five
                scores[i,j] = pixel_score

    return scores, max_idxs, zero_is_five

def depthOneAccuracy(x_test, y_test, i, j, zero_is_five):
    """
    Given the most distinctive pixel is (i, j), return the accuracy on the test set (should be between 0-1).
    If zero_is_five==True, then pixel (i, j) == 0 implies we predict 5.
    Otherwise, pixel (i, j) == 0 implies we predict 9.
    """
    accuracy = 0
    # TODO Implement this
    for image in range(x_test.shape[2]):
        if zero_is_five:
            if x_test[i, j, image] == 0 and y_test[image] == 5:
                accuracy += 1
            elif x_test[i, j, image] == 1 and y_test[image] == 9:
                accuracy += 1
        else:
            if x_test[i, j, image] == 0 and y_test[image] == 9:
                accuracy += 1
            elif x_test[i, j, image] == 1 and y_test[image] == 5:
                accuracy += 1

    return accuracy / len(y_test)

def trainDepthTwo(x_train, y_train, i0, j0):
    """
    Return a depth 2 tree where the outputs are (i1, j1, zero_five1, i2, j2, zero_five2) which can be substituted to make the tree:
	if pixel(i0, j0) == 0:
        if pixel(i1, j1) == 0 and zero_five1:
            predict 5
        else:
            predict 9
    else:
        if pixel(i2, j2) == 0 and zero_five1:
            predict 5
        else:
            predict 9
		
    Note i0, j0 is the pixel from step 2.
    """
    i1 = j1 = i2 = j2 = 0
    zero_five1 = zero_five2 = False
    # TODO Implement this

    # First, split x_train into two different lists,
    # where one list has images where pixel(i0, j0) = 0
    # and the other list has images where pixel(i0, j0) = 1
    index_list_1 = []
    index_list_2 = []

    for image in range(len(y_train)):
        if x_train[i0, j0, image] == 0:
            index_list_1.append(image)
        else:
            index_list_2.append(image)

    x_train_1 = x_train[:,:,index_list_1]
    y_train_1 = y_train[index_list_1]

    x_train_2 = x_train[:,:,index_list_2]
    y_train_2 = y_train[index_list_2]

    # Find best feature from each of the two lists
    scores1, max_idxs1, zero_five1 = scoreFeatures(x_train_1, y_train_1)
    i1, j1 = max_idxs1

    scores2, max_idxs2, zero_five2 = scoreFeatures(x_train_2, y_train_2)
    i2, j2 = max_idxs2

    return i1, j1, zero_five1, i2, j2, zero_five2

def depthTwoAccuracy(x_test, y_test, i0, j0, i1, j1, zero_five1, i2, j2, zero_five2):
    """
    Given a depth two tree with the parameters above and described by the tree in trainDepthTwo, return the test accuracy.
    """
    # 1.3.b) test depth 2
    accuracy = 0
    # TODO Implement this
    five = 0
    nine = 1

    for image in range(x_test.shape[2]):
        if x_test[i0, j0, image] == 0:
            if zero_five1:
                five = 0
                nine = 1
            else:
                five = 1
                nine = 0

            if x_test[i1, j1, image] == five and y_test[image] == 5:
                accuracy += 1
            elif x_test[i1, j1, image] == nine and y_test[image] == 9:
                accuracy += 1

        else:
            if zero_five2:
                five = 0
                nine = 1
            else:
                five = 1
                nine = 0
            
            if x_test[i2, j2, image] == five and y_test[image] == 5:
                accuracy += 1
            elif x_test[i2, j2, image] == nine and y_test[image] == 9:
                accuracy += 1

    return accuracy / len(y_test)


if __name__ == "__main__":
    data = pickle.load(open('data.pkl','rb'))
    x_train = np.asarray(data['train']['x'])
    y_train = np.asarray(data['train']['y'])
    x_test = np.asarray(data['test']['x'])
    y_test = np.asarray(data['test']['y'])

    scores, best_feat, zero_is_five = scoreFeatures(x_train, y_train)
    print("Depth 1")
    print("Best feature:", best_feat)
    print("Best feature score:", scores[best_feat[0], best_feat[1]])
    print("Zero is five:", zero_is_five)

    plt.imshow(scores, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

    i0, j0 = best_feat[0], best_feat[1]

    print('Depth 1 accuracy on test set: ', depthOneAccuracy(x_test, y_test, i0, j0, zero_is_five))

    i1, j1, zero_five1, i2, j2, zero_five2 = trainDepthTwo(x_train, y_train, i0, j0)

    print()
    print('Depth 2:')
    print(f'(i1, j1) = ({i1}, {j1})')
    print(f'Zero is five 1: {zero_five1}')
    print(f'(i2, j2) = ({i2}, {j2})')
    print(f'Zero is five 2: {zero_five2}')
    print()

    print('depth 2 accuracy on test set: ', depthTwoAccuracy(x_test, y_test, i0, j0, i1, j1, zero_five1, i2, j2, zero_five2))
	
