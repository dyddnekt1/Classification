import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg

########################################
# Part 1. cross entropy loss
########################################
from sklearn import metrics


def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    """
        return cross entropy loss
    """
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))

    s = W @ x + b
    s = sp.softmax(s)
    loss = metrics.log_loss(y, s.T, labels=np.array(range(num_class)))

    return loss


########################################
# Part 2. SVM loss calculation
########################################
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    """
        return svm loss
    """
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))

    s = W @ x + b
    loss = metrics.hinge_loss(y, s.T, labels=np.array(range(num_class)))
    return loss
    pass


########################################
# Part 3. kNN classification
########################################
def knn_test(X_train, y_train, X_test, y_test, n_train_sample, n_test_sample, k):
    """
        return accuracy of knn test
    """

    dists = -2 * np.dot(X_test, X_train.T)\
            + np.sum(X_train ** 2, axis=1) + np.sum(X_test ** 2, axis=1)[:, np.newaxis]
    nearest = np.argsort(dists, axis=1)[:, :k]
    label = np.reshape(stats.mode(np.array(y_train[nearest]), axis=1)[0], (1, -1))
    accuracy = (label == y_test).sum() / label.size

    return accuracy
    pass


# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x_te, y_te, num_class,n_test):
    Wb = np.reshape(Wb, (-1, 1))
    dlen = len(x_te[0])
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * dlen)], (num_class, dlen))
    accuracy = 0;

    for i in range(n_test):
        # find the linear scores
        s = W @ x_te[i].reshape((-1, 1)) + b
        # find the maximum score index
        res = np.argmax(s)
        accuracy = accuracy + (res == y_te[i]).astype('uint8')

    return accuracy / n_test

# set hyperparameter
num_class = 4
sigma = 1.0

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
classifiers = 'knn'

if classifiers == 'svm':
    print('training SVM classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')

    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

elif classifiers == 'softmax':
    print('training softmax classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')

    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

else:  # knn
    # k value for kNN classifier. k can be either 1 or 3.
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, n_train, n_test, k)*100
          , '% for k value of ', k)
