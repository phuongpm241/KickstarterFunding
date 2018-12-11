import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
import numpy as np
from parseData import *


if __name__ == '__main__':
    train_data = pd.read_csv('final_train_data.csv')
    print ("Splitting data...")
    X, y = getFeatures(x_features = ['log_goal', 'backers_count', 'duration_weeks'])
    X_train, X_test, y_train , y_test = splitData(X, y, 0.2, [])
    y_train = y_train.replace(0, -1).values
    y_test = y_test.replace(0, -1).values

    filename = 'finalized_model.sav'
    # polynomial kernel
    degrees = [2, 4, 8, 16]
    coefs = [1e-5, 1e-1, 1, 1e1, 1e2]
    Cs = [1e-3, 1e-1, 1e0, 1e1, 1e3]
    poly_accs = {}
    for c in Cs:
        for d in degrees:
            print ("Start training polynomial kernel...")
            poly_model = svm.SVC(kernel='poly', degree=d, max_iter=1)
            poly_model.fit(X_train, y_train)
            pickle.dump(model, open(filename, 'wb'))
            print ("Finish training polynomial kernel")
            poly_train_acc = poly_model.score(X_train, y_train)
            poly_test_acc = poly_model.score(X_test, y_test)
            print ("Degree: ", d)
            print ("Polynomial train accuracy: ", poly_train_acc)
            print ("Polynomail test accuracy: ", poly_test_acc)
            poly_accs[d] = (poly_train_acc, poly_test_acc)
            break
        break
    print (poly_accs)
