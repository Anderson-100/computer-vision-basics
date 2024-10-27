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

def softmax(z):
    return 1.0/(1+np.exp(-z))

def linearTrain(x, y):
    #Training parameters
    maxiter = 50
    lamb = 0.01
    eta = 0.01
    
    #Add a bias term to the features
    x = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)
    
    class_labels = np.unique(y)
    num_class = class_labels.shape[0]
    assert(num_class == 2) # Binary labels
    num_feats = x.shape[0]
    num_data = x.shape[1]
    
    true_prob = np.zeros(num_data)
    true_prob[y == class_labels[0]] = 1
    
    #Initialize weights randomly
    model = {}
    model['weights'] = np.random.randn(num_feats)*0.01
    # print('w', model['weights'].shape)
    #Batch gradient descent
    verbose_output = False
    for it in range(maxiter):
        prob = softmax(model['weights'].dot(x))
        delta = true_prob - prob
        gradL = delta.dot(x.T)
        model['weights'] = (1 - eta*lamb)*model['weights'] + eta*gradL
    model['classLabels'] = class_labels

    return model


def linearPredict(model, x):
    #Add a bias term to the features
    x = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)

    prob = softmax(model['weights'].dot(x))
    ypred = np.ones(x.shape[1]) * model['classLabels'][1]
    ypred[prob > 0.5] = model['classLabels'][0]

    return ypred


def testLinear():
    #--------------------------------------------------------------------------
    # Your implementation to answer questions on Linear Classifier
    # This question will not be autograded. You should include the results you get in your report.
    #--------------------------------------------------------------------------
    data = pickle.load(open('data.pkl','rb'))
    x_train = np.asarray(data['train']['x'])
    y_train = np.asarray(data['train']['y'])
    x_test = np.asarray(data['test']['x'])
    y_test = np.asarray(data['test']['y'])

    # Flatten images
    x_train = np.reshape(x_train, (784, 200))
    x_test = np.reshape(x_test, (784, 200))

    model = linearTrain(x_train, y_train)
    y_pred = linearPredict(model, x_test)

    # print(y_test)
    # print(y_pred)
    acc = 0
    for i in range(len(y_test)):
        if y_test[i] == int(y_pred[i]):
            acc += 1
    
    acc /= len(y_test)
    print("Accuracy:", acc)

    w = model['weights']
    wp = np.clip(w, 0, None)[:-1]
    wn = np.clip(w, None, 0)[:-1]

    wp = np.reshape(wp, (28, 28))
    wn = np.reshape(wn, (28, 28))

    plt.imshow(wp)
    plt.title("Positive Components")
    plt.show()
    plt.imshow(wn)
    plt.title("Negative Components")
    plt.show()


if __name__ == "__main__":
    testLinear()
