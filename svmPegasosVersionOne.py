from dataset import *
import numpy as np
import pylab as pl

def pegasos_svm_linear(X, Y, lambda_constant, max_epochs):
    num_data, dim = X.shape
    t = 0
    w = np.zeros(dim)
    offset = 0

    for epoch in range(max_epochs):
        for i in range(num_data):
            t += 1
            learning_rate = 1.0/(t*lambda_constant)
            if np.dot(Y[i], np.dot(w, X[i]) + offset) <= 1.0:
                w = (1.0 - learning_rate*lambda_constant)*w + learning_rate*Y[i]*X[i]
                offset = offset + learning_rate*Y[i]
            else:
                w = (1.0-learning_rate*lambda_constant)*w

    return (w, offset)

def pegasos_svm_gaussian(X, Y, lambda_constant, gaussian_kernel, gamma, max_epochs):
    num_data, dim = X.shape
    t = 0
    alphas = np.zeros((num_data, 1))

    for epoch in range(max_epochs):
        for i in range(num_data):
            t += 1
            learning_rate = 1.0/(t*lambda_constant)
            alpha = np.dot((alphas*Y).T, np.array([gaussian_kernel(X, X[i], gamma)]).T)
            print ("i is ", i, " with alpha ", alpha)

            if np.dot(Y[i], alpha) <= 1.0:
                alphas[i] = (1.0-learning_rate*lambda_constant)*alphas[i] + learning_rate
            else:
                alphas[i] = (1.0 - learning_rate*lambda_constant)*alphas[i]
    return alphas


def gaussian_kernel(X, x, gamma):
    x_tile = tile(x, (len(X), 1))
    return np.power(math.exp(1), -gamma*np.dot(X - x_tile, (X- x_tile).T).diagonal())

def predict_svm_gaussian(X, Y, new_data, gaussian_kernel, gamma, alphas):
    return np.dot((alphas*Y).T, np.array([gaussian_kernel(X, new_data, gamma)]).T)

def sign(val):
    return 1 if val > 0 else -1

def accuracy_kernel(X, Y, gaussian_kernel, gamma, alphas):
    num_data, dim = X.shape
    correct = 0
    for i in range(num_data):
        if predict_svm_gaussian(X, Y, X[i], gaussian_kernel, gamma, alphas)*Y[i] > 0:
            correct += 1

    return correct/float(num_data)

if __name__ == '__main__':
    max_epochs = 50
    lambda_constant = 2e-1
    gamma = 2e2
    x_features = ['log_goal','country', 'currency', 'backers_count', 'launched_year', 'launched_month', 'duration_weeks']
    y_features = 'final_status'

    X, y = getFeatures(x_features, y_features)
    X_train, X_test, y_train, y_test = splitData(X, y, 0.2, ['country', 'currency', 'launched_month'])

    alphas = pegasos_svm_gaussian(X_train, y_train.values, lambda_constant, gaussian_kernel, gamma, max_epochs)
    train_score = accuracy_kernel(X_train, y_train.values, gaussian_kernel, gamma, alphas)
    test_score = accuracy_kernel(X_test, y_test.values, gaussian_kernel, gamma, alphas)
    print ("train score: ", train_score)
    print ("test score: ", test_score)

    

    
