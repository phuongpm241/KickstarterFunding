import numpy as np
import pandas as pd
from dataset import *
from math import *
import time
##import pycuda.driver as cuda
##import pycuda.autoinit
##from pycude.compiler import SourceModule

def PegasosSVM(X, y, l, max_epochs):
    num_data, d = X.shape
    t = 0
    w = np.zeros(d)
    b = 0
    for epoch in range(max_epochs):
        for i in range(num_data):
            t += 1
            eta = 1.0/(t*l)
            if np.dot(y[i], np.dot(w, X[i]) + b) < 1.0:
                w = (1.0-eta*l)*w + eta*y[i]*X[i]
                b = b+eta*y[i]
            else:
                w = (1.0-eta*l)*w
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
    start = time.time()
    X, y = getFeatures(x_features = ['log_goal', 'country', 'currency', 'backers_count',
                                     'duration_weeks', 'text_polarity', 'text_subjectivity'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0.2, ['country', 'currency'])
    end = time.time()
    print ("split data takes: ", end-start)
    y_train = y_train.replace(0, -1).values
    y_test = y_test.replace(0, -1).values

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
    
    

    linear_fn = lambda p1, p2, c: np.dot(p1.T, p2) + c
    poly_fn = lambda p1, p2, coef, c, degree: np.power(coef*np.dot(p1.T, p2)+c, d)
    gaussian_fn = lambda p1, p2, gamma: exp(-gamma*np.dot(p1-p2, p1-p2))

    print ("Start training...")
    max_epochs = 5
    alphas_linear = PegasosSVMLinearKernel(X_train, y_train, 2e-1, linear_fn, max_epochs, 0)
    print ("Finish training")
    train_acc = AccuracySVMLinearKernel(X_train, y_train, linear_fn, 0)
    test_acc = AccuracySVMLinearKernel(X_test, y_test, linear_fn, 0)
    print ("train_acc: ", train_acc)
    print ("test_acc: ", test_acc)
    

    
    
