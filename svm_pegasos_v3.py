from sklearn import svm
import numpy as np
import time
import pandas as pd
from dataset import *
from math import *


if __name__ == '__main__':
    start = time.time()
    X, y = getFeatures(x_features = ['log_goal', 'country', 'currency', 'backers_count',
                                     'duration_weeks', 'text_polarity', 'text_subjectivity'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0.2, ['country', 'currency'])
    end = time.time()
    print ("split data takes: ", end-start)
    y_train = y_train.replace(0, -1).values
    y_test = y_test.replace(0, -1).values

    # linear kernel
    print ("Start training linear kernel...")
    linear_model = svm.SVC(kernel='linear')
    linear_model.fit(X_train, y_train)
    print ("Finish training linear kernel")
    linear_train_acc = linear_model.score(X_train, y_train)
    linear_test_acc = linear_model.score(X_test, y_test)
    print ("Linear train accuracy: ", linear_train_acc)
    print ("Linear test accuracy: ", linear_test_acc)

    # polynomial kernel
    degrees = [2, 3, 5, 10, 25, 50]
    coefs = [0.1, 0.5, 1, 5, 10, 50, 100]
    poly_accs = {}
    for d in degrees:
        for coef in coefs:
            print ("Start training polynomial kernel...")
            poly_model = svm.SVC(kernel='poly', degree=d, coef0=coef)
            poly_model.fit(X_train, y_train)
            print ("Finish training polynomial kernel")
            poly_train_acc = poly_model.score(X_train, y_train)
            poly_test_acc = poly_model.score(X_test, y_test)
            print ("Polynomial train accuracy: ", poly_train_acc)
            print ("Polynomail test accuracy: ", poly_test_acc)
            poly_accs[(d, coef)] = (poly_train_acc, poly_test_acc)

    # gaussian kernel
    gaussian_accs = {}
    gammas = [1e-10, 1e-5, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e5, 'auto', 'scale']
    for g in gammas:
        print ("Start training gaussian kernel...")
        gaussian_model = svm.SVC(gamma=g, kernel='rbf')
        gaussian_model.fit(X_train, y_train)
        print ("Finish training gaussian kernel")
        gaussian_train_acc = gaussian_model.score(X_train, y_train)
        gaussian_test_acc = gaussian_model.score(X_test, y_test)
        print ("Gaussian train accuracy: ", gaussian_train_acc)
        print ("Gaussian test accuracy: ", gaussian_test_acc)
        gaussian_accs[g] = (gaussian_train_acc, gaussian_test_acc)


    print ("Finish all kernels")
    print ("Linear: ")
    print ("Linear train accuracy: ", linear_train_acc)
    print ("Linear test accuracy: ", linear_test_acc)
    print ("Polynomial: ")
    print (poly_accs)
    print ("Gaussian: ")
    print (gaussian_accs)
    
    

