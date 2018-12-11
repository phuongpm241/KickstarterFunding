from sklearn import svm
import numpy as np
import time
import pandas as pd
from math import *
from datetime import datetime
from textblob import TextBlob # sentiment analysis
from sklearn.model_selection import train_test_split, GridSearchCV
from parseData import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler

def getFeatures(x_features=None, y_feature='final_status'): 
    if len(x_features) == None:
        X = train_data
    else:
        X = train_data[x_features]
        
    y = train_data[y_feature]
    return X, y

def splitData(X, y, size, onehot): 
    """ Return train/test data splitted based on 'size' with certain features one-hot encoded
        All the data returned as numpy array
    """
    onehot_X = pd.get_dummies(X, prefix=onehot, columns=onehot).values
    X_train, X_test, y_train, y_test = train_test_split(onehot_X, y, test_size=size, random_state = 42)
    return X_train, X_test, y_train, y_test

def hingeLoss(X_test, y_test, model):
    y_predict = model.predict(X_test)
    num_data, d = X_test.shape
    loss = 0.0
    for i in range(num_data):
        loss += max(0.0, 1.0- y_test[i]*y_predict[i])
    return loss
    

def svc_param_selection(X, y, nfolds):
    Cs = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    gammas = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    param_grid = {'C': Cs, 'gamma':gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


if __name__ == '__main__':
    train_data = pd.read_csv('final_train_data.csv')
    print ("Splitting data...")
    X, y = getFeatures(x_features = ['log_goal', 'backers_count',
                                     'duration_weeks'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0.2, [])
    print ("finished splitting data.")
##    scaler = MinMaxScaler()
##    X_train = scaler.fit_transform(X_train)
##    X_test = scaler.fit_transform(X_test)
    y_train = y_train.replace(0, -1).values
    y_test = y_test.replace(0, -1).values
##
##    iters = [1e0, 1e1, 1e2, 1e3, 1e5, 1e7]
##    linear_acc = {}
##    grid_search = svc_param_selection(X_train, y_train, 2)
##    print (grid_search.best_params_, grid_search.best_score_)

##    # linear kernel
##    for iteration in iters:
##        print ("iteration ", iteration)
##        print ("Start training linear kernel...")
##        linear_model = svm.SVC(kernel='linear', max_iter = iteration)
##        linear_model.fit(X_train, y_train)
##        print ("Finish training linear kernel")
##        linear_train_acc = linear_model.score(X_train, y_train)
##        linear_test_acc = linear_model.score(X_test, y_test)
##        print ("Linear train accuracy: ", linear_train_acc)
##        print ("Linear test accuracy: ", linear_test_acc)
##        linear_acc[iteration] = (linear_train_acc, linear_test_acc)
##    print (linear_acc)

    # polynomial kernel
    degrees = [2, 4, 8]
    coefs = [1e-5, 1e-1, 1e2]
    poly_accs = {}
    iters = [1e5, 1e6]
    Cs = [1e-5, 1e-4]

    for c in Cs:
        print ("Start training polynomial kernel...")
        poly_model = svm.SVC(kernel='poly', C = c, max_iter = 1e6, degree=4).fit(X_train, y_train)
        print ("Finish training polynomial kernel")

        train_loss = hingeLoss(X_train, y_train, poly_model)
        test_loss = hingeLoss(X_test, y_test, poly_model)
        
        poly_train_acc = poly_model.score(X_train, y_train)
        poly_test_acc = poly_model.score(X_test, y_test)
        print ("iteration ", c)
        print ("Polynomial train loss: ", train_loss)
        print ("Polynomial test loss: ", test_loss)
        print ("Polynomial train accuracy: ", poly_train_acc)
        print ("Polynomail test accuracy: ", poly_test_acc)
        poly_accs[c] = (train_loss, test_loss)
    print (poly_accs)
##

##    # gaussian kernel
    gaussian_accs = {}
    Cs = [1e-1, 1e0, 1e1, 1e2, 1e3]
##    gammas = [1e-10, 1e-5, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e5]
    for c in Cs:
        print ("Start training gaussian kernel...")
        gaussian_model = svm.SVC(gamma=0.2, kernel='rbf', C=c)
        gaussian_model.fit(X_train, y_train)
        print ("Finish training gaussian kernel")
        gaussian_train_acc = gaussian_model.score(X_train, y_train)
        gaussian_test_acc = gaussian_model.score(X_test, y_test)
        print ("c ", c)
        print ("Gaussian train accuracy: ", gaussian_train_acc)
        print ("Gaussian test accuracy: ", gaussian_test_acc)
        gaussian_accs[c] = (gaussian_train_acc, gaussian_test_acc)
        print (gaussian_accs)


##    print ("Finish all kernels")
####    print ("Linear: ")
####    print ("Linear train accuracy: ", linear_train_acc)
####    print ("Linear test accuracy: ", linear_test_acc)
##    print ("Polynomial: ")
##    print (poly_accs)
##    print ("Gaussian: ")
##    print (gaussian_accs)
    
    

