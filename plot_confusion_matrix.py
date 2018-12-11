import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from parseData import *
from svm_pegasos_v2 import *
from get_metrics import *


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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def SVMGaussianPrediction(X_train, y_train, X_test, c, g):
    classifier = svm.SVC(kernel='rbf', C=c, gamma=g)
    classifier.fit(X_train, y_train)
    y_predit = classifier.predict(X_test)
    return y_predit

def SVMPegasosPrediction(X_train, y_train, l, max_epochs, X_test):
    w, b = PegasosSVM(X_train, y_train, l, max_epochs)

    def sign(value):
        if value > 0:
            return 1
        else:
            return -1

    def PredictLabel(X, w, b):
        num_data, d = X.shape
        predicted = np.zeros(num_data)
        for i in range(num_data):
            predicted[i] = sign(PredictSVM(w, b, X[i]))
        return predicted

    return PredictLabel(X_test, w, b)

def SVMLinearPrediction(X_train, y_train, X_test, c):
    classifier = svm.SVC(kernel='linear', C=c, max_iter=1e7)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    return y_predict

def SVMPolyPrediction(X_train, y_train, X_test, c, d):
    classifier = svm.SVC(kernel='poly', C=1e-3, degree=2, max_iter=1e7)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    return y_predict
    
    

if __name__ == '__main__':
    train_data = pd.read_csv('final_train_data.csv')
    X, y = getFeatures(x_features = ['log_goal', 'backers_count',
                                     'duration_weeks'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0.2, [])
    y_train = y_train.replace(0, -1).values
    y_test = y_test.replace(0, -1).values
    print ("finish split data")

##    pegasos_predict = SVMPegasosPrediction(X_train, y_train, 200, 100, X_test)
##    pegasos_metric = metrics(y_test, pegasos_predict)

##    svm_gaussian_predict = SVMGaussianPrediction(X_train, y_train, X_test, 1, 0.2)
##    svm_gaussian_metric = metrics(y_test, svm_gaussian_predict)

##    svm_linear_predict = SVMLinearPrediction(X_train, y_train, X_test, 1)
##    svm_linear_metric = metrics(y_test, svm_linear_predict)

    svm_poly_predict = SVMPolyPrediction(X_train, y_train, X_test, 1e-3, 2)
    svm_poly_metric = metrics(y_test, svm_poly_predict)

    
##
##    
##    
##
##
##    # Compute confusion matrix
##    cnf_matrix = confusion_matrix(y_test, y_pred)
##    np.set_printoptions(precision=2)
##
##    # Plot non-normalized confusion matrix
##    plt.figure()
##    plot_confusion_matrix(cnf_matrix, classes=class_names,
##                          title='Confusion matrix, without normalization')
##
##
##    plt.show()
