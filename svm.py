from dataset import *
import numpy as np
from plotBoundary import *
import pylab as pl
from sklearn.metrics import accuracy_score

def Pegasos_SVM(X, Y, lambda_constant, max_epochs):
    num_data, d = X.shape
    t = 0
    w = np.zeros(d)
    offset = 0
    epoch = 0
    while epoch < max_epochs:
        for i in range(num_data):
            t += 1
            learning_rate = 1.0/(t*lambda_constant)
            #if np.dot(Y[i], (np.dot(w, X[i])+offset)) < 1.0:
            if np.dot(Y[i], np.dot(w, X[i])) < 1.0:
                w = (1.0 - learning_rate*lambda_constant)*w + learning_rate*Y[i]*X[i]
                #offset = offset + learning_rate*Y[i]
            else:
                w = (1.0-learning_rate*lambda_constant)*w
        epoch+=1
    return w
#    return (w, offset)


def pegasos_no_offset(X, y, lambda_constant, max_epochs):
    n = len(X)
    t = 0
    w = np.zeros(len(X[0]))
    epoch = 0
    while epoch < max_epochs:
        for i in range(n):
            t += 1
            learning_rate = 1.0/(t*lambda_constant)
            if np.dot(y[i], (np.dot(w, X[i]))) < 1.0:
                w = (1.0-learning_rate*lambda_constant)*w + learning_rate*y[i]*X[i]
            else:
                w = (1.0-learning_rate*lambda_constant)*w
        epoch+=1
    return w

def predict_no_offset(w):
    def predict_SVM(new_data):
        return np.dot(w, new_data)
    return predict_SVM
def accuracy_no_offset(X, y, w):
    num_misclassified = 0
    for i in range(len(X)):
        if predict_no_offset(w, X[i]) * y[i] < 0:
            num_misclassified += 1
    return 1 - num_misclassified/len(X)

def predict(w, offset, new_data):
    return np.dot(w, new_data) + offset

def accuracy(X, y, w, offset):
    num_misclassified = 0
    for i in range(len(X)):
        if predict(w, offset, X[i]) * y[i] < 0:
            num_misclassified += 1
    return 1 - num_misclassified/float(len(X))

def find_accuracy(X, y, w):
    predicted = []
    weight = np.ones(y.shape[0])
    for i in range(len(X)):
        predicted.append(predict(w, offset, X[i]))
    print ("predicted", predicted)
    accuracy_score(y, np.array(predicted), sample_weight=weight)

if __name__ == '__main__':
    lambda_constant = 1e-1
    X, y = getFeatures(x_features = ['log_goal','country', 'currency', 'backers_count',
                                     'launched_year', 'launched_month', 'duration_weeks', 
                                     'buzzword_count'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0.2, ['country', 'currency', 'launched_month'])
    w = Pegasos_SVM(X_train, y_train.values, lambda_constant, 50)
    #acc = find_accuracy(X_train, y_train.values, w)
    #print (acc)
    #print (w)
    #print (offset)
    
    
    #acc = accuracy(X_test, y_test, w, offset)
    #print ("accuracy of linear SVM is ", acc)

    
