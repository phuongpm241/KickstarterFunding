import numpy as np
import pandas as pd
from math import *
from sklearn.model_selection import train_test_split

import time
##import pycuda.driver as cuda
##import pycuda.autoinit
##from pycude.compiler import SourceModule

def getFeatures(train_data, x_features=None, y_feature='final_status'): 
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
#    if size == 0:
#        return onehot_X, [], y, []
    X_train, X_test, y_train, y_test = train_test_split(onehot_X, y.values, test_size=size, random_state = 42)
    return X_train, X_test, y_train, y_test

def PegasosSVM(X, y, X_val, y_val, l, max_epochs):
    num_data, d = X.shape
    t = 0
    w = np.zeros(d)
    b = 0

    acc_conv, test_acc_conv = list(), list()
    for epoch in range(max_epochs):
        for i in range(num_data):
            t += 1
            eta = 1.0/(t*l)
            if np.dot(y[i], np.dot(w, X[i]) + b) < 1.0:
                w = (1.0-eta*l)*w + eta*y[i]*X[i]
                b = b+eta*y[i]
            else:
                w = (1.0-eta*l)*w

        acc_conv.append(AccuracySVM(X, y, w, b))
        test_acc_conv.append(AccuracySVM(X_val, y_val, w, b))

    print (acc_conv)
    print (test_acc_conv)
    return (w, b)

def ConvergenceSVM(X, y, X_val, y_val, l):
    num_data, d = X.shape
    t = 0
    w = np.zeros(d)
    b = 0
    epochs = 1

    while AccuracySVM(X, y, w, b) < 0.78 and epochs < 1e4:
        for i in range(num_data):
            t += 1
            eta = 1.0/(t*l)
            if np.dot(y[i], np.dot(w, X[i]) + b) < 1.0:
                w = (1.0-eta*l)*w + eta*y[i]*X[i]
                b = b+eta*y[i]
            else:
                w = (1.0-eta*l)*w

        epochs += 1
        if epochs % 50 == 0: 
            print (epochs, AccuracySVM(X, y, w, b))

    print (AccuracySVM(X, y, w, b))
    print (AccuracySVM(X_val, y_val, w, b))
    print (epochs)
    return (w, b)

def PredictSVM(w, b, new_data):
    return np.dot(w, new_data)+b

def AccuracySVM(X, y, w, b):
    num_data, d = X.shape
    correct = 0
    for i in range(num_data):
        if PredictSVM(w, b, X[i])*y[i] > 0:
            correct+=1
    return float(correct)/num_data

def PegasosSVMGaussianKernel(X, y, l, gaussian_fn, max_epochs, gamma):
    num_data, d = X.shape
    t = 0
    alphas = np.zeros(num_data)

    for epoch in range(max_epochs):
        for i in range(num_data):
            t+=1
            eta = 1.0/(t*l)
            total = 0
            for j in range(num_data):
                total += alphas[j]*y[j]*gaussian_fn(X[j], X[i], gamma)
            if y[i]*total < 1:
                alphas[i] = (1-eta*l)*alphas[i] + eta
            else:
                alphas[i] = (1-eta*l)*alphas[i]
    return alphas

def PegasosSVMLinearKernel(X, y, l, linear_fn, max_epochs, c):
    # k(x, y) = x^T * y + c
    num_data, d = X.shape
    t = 0
    alphas = np.zeros(num_data)
    for epoch in range(max_epochs):
        for i in range(num_data):
            t+=1
            eta = 1.0/(t*l)
            total = 0
            for j in range(num_data):
                total += alphas[j]*y[j]*linear_fn(X[j], X[i], c)
            if y[i]*total < 1:
                alphas[i] = (1-eta*l)*alphas[i] + eta
            else:
                alphas[i] = (1-eta*l)*alphas[i]
        print ("epoch: ", epoch, " finshed")
    return alphas

def PegasosSVMPolynomialKernel(X, y, l, poly_fn, max_epochs, coef, c, degree):
    # k(x, y) = (coeff * x^T * y + c)^d
    num_data, d = X.shape
    t = 0
    alphas = np.zeros(num_data)
    for epoch in range(max_epochs):
        for i in range(num_data):
            t+=1
            eta = 1.0/(t*l)
            total = 0
            for j in range(num_data):
                total += alphas[j]*y[j]*poly_fn(X[j], X[i], coef, c, degree)
            if y[i]*total < 1:
                alphas[i] = (1-eta*l)*alphas[i] + eta
            else:
                alphas[i] = (1-eta*l)*alphas[i]
                
    return alphas
    
    
def PredictSVMGaussianKernel(X, y, new_data, alphas, gaussian_fn, gamma):
    num_data, d = X.shape
    total = 0
    for i in range(num_data):
        total += alphas[i]*y[i]*gaussian_fn(new_data, X[i], gamma)
    return total

def AccuracySVMGaussianKernel(X, y, alphas, gaussian_fn, gamma):
    num_data, d = X.shape
    correct = 0
    for i in range(num_data):
        if PredictSVMGaussianKernel(X, y, X[i], alphas, gaussian_fn, gamma) * y[i] > 0:
            correct += 1
    return float(correct)/num_data

def PredictSVMLinearKernel(X, y, new_data, alphas, linear_fn, c):
    num_data, d = X.shape
    total = 0
    for i in range(num_data):
        total += alphas[i]*y[i]*linear_fn(new_data, X[i], c)
    return total

def AccuracySVMLinearKernel(X, y, alphas, linear_fn, c):
    num_data, d = X.shape
    correct = 0
    for i in range(num_data):
        if PredictSVMLinearKernel(X, y, X[i], alphas, linear_fn, c) * y[i] > 0:
            correct += 1
    return float(correct)/num_data

def PredictSVMPolynomialKernel(X, y, new_data, alphas, poly_fn, coef, c, degree):
    num_data, d = X.shape
    total = 0
    for i in range(num_data):
        total += alphas[i]*y[i]*poly_fn(new_data, X[i], coef, c, degree)
    return total

def AccuracySVMPolynomialKernel(X, y, alphas, poly_fn, coef, c, degree):
    num_data, d = X.shape
    correct = 0
    for i in range(num_data):
        if PredictSVMPolynomialKernel(X, y, X[i], alphas, poly_fn, coef, c, degree) * y[i] > 0:
            correct += 1
    return float(correct)/num_data

if __name__ == '__main__':

    train_data = pd.read_csv('final_train_data.csv')

    X, y = getFeatures(train_data, x_features = ['log_goal', 'backers_count','duration_weeks'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0.2, [])

    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1 

    lambdas = [2e-5, 2e-2, 2e-1, 2e0, 2e1, 2e2, 2e3, 2e4]
    # w_b = {}
    # accs = {}

    # epochs = [5, 10, 50, 100]
    # for epoch in epochs:
    #     for l in lambdas:
    #         print ("epoch: ", epoch, "l: ", l)
    #         w, b = PegasosSVM(X_train, y_train, X_test, y_test, l, epoch)
    #         train_acc = AccuracySVM(X_train, y_train, w, b)
    #         test_acc = AccuracySVM(X_test, y_test, w, b)
    #         w_b[(epoch, l)] = (w, b)
    #         accs[(epoch, l)] = (train_acc, test_acc)
    # print ("w_b: ", w_b)
    # print ("accuracy: ", accs)

    # lambdas = [200, 600, 750, 790, 800, 810, 850, 900]
    # for l in lambdas:
    #     print (l)
    #     w, b = ConvergenceSVM(X_train, y_train, X_test, y_test, l)


##    # convert to single precision for pycuda
##    X_train = X_train.astype(np.float32)
##    X_test = X_test.astype(np.float32)
##    y_train = y_train.astype(np.float32)
##    y_test = y_test.astype(np.float32)
##
##    # allocate memory
##    X_train_gpu = cuda.mem_alloc(X_train.nbytes)
##    X_test_gpu = cuda.mem_alloc(X_test.nbytes)
##    y_train_gpu = cuda.mem_alloc(y_train.nbytes)
##    y_test_gpu = cuda.mem_alloc(y_test.nbytes)
##
##    # transfer data to GPU
##    cuda.memcpy_htod(X_train_gpu, X_train)
##    cuda.memcpy_htod(X_test_gpu, X_test)
##    cuda.memcpy_htod(y_train_gpu, y_train)
##    cuda.memcpy_htod(y_test_gpu, y_test)
##
##    # execute
##    mod = SourceModel("""
##      __global__ void 
##    """)
    
    

##    linear_fn = lambda p1, p2, c: np.dot(p1.T, p2) + c
##    poly_fn = lambda p1, p2, coef, c, degree: np.power(coef*np.dot(p1.T, p2)+c, d)
##    gaussian_fn = lambda p1, p2, gamma: exp(-gamma*np.dot(p1-p2, p1-p2))
##
    # print ("Start training...")
    # max_epochs = 5
    # alphas_linear = PegasosSVMLinearKernel(X_train, y_train, 2e-1, linear_fn, max_epochs, 0)
    # print ("Finish training")
    # train_acc = AccuracySVMLinearKernel(X_train, y_train, linear_fn, 0)
    # test_acc = AccuracySVMLinearKernel(X_test, y_test, linear_fn, 0)
    # print ("train_acc: ", train_acc)
    # print ("test_acc: ", test_acc)
    
    

    
    
