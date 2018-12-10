#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:39:32 2018

@author: maggiewu
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

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

def getLogisticRegression(X_train, y_train, solver='liblinear', C=1.0, max_iter=100, multi_class='multinomial', tol=1e-4): 
	lr = LogisticRegression(random_state=0, solver=solver, C=C, max_iter=max_iter, multi_class=multi_class, tol=tol)
	clf = lr.fit(X_train, y_train)
	return lr, clf 

def accuracy(clf, X_train, y_train, X_test, y_test): 
	return clf.score(X_train, y_train), clf.score(X_test, y_test)

def specificity(true, pred):
	matrix = confusion_matrix(true, pred) 
	return (matrix[0][0]/ (matrix[0][0] + matrix[0][1]))

def metrics(true, pred): 
	print ('confusion_matrix', confusion_matrix(true, pred))
	print ('precision_score: ' + str(round(precision_score(true, pred), 3)))
	print ('recall score: ' + str(round(recall_score(true, pred), 3)))
	print ('specificity: ' + str(round(specificity(true, pred), 3)))
	print ('f1 score: ' + str(round(f1_score(true, pred), 3)))

def testLR(): 
	train_data = pd.read_csv('final_train_data.csv')
	train_data['goal_backer_ratio'] = np.where(train_data['backers_count'] == 0, train_data['backers_count'], train_data['goal']/train_data['backers_count'])

	X, y = getFeatures(train_data, x_features=['log_goal', 'backers_count', 'duration_weeks'])
	X_train, X_test, y_train, y_test = splitData(X, y, 0.2, [])

	lr, clf = getLogisticRegression(X_train, y_train, solver='lbfgs')

	pred = clf.predict(X_test)
	pred = np.round(pred)

	metrics(y_test, pred)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

def runLR(): 
	train_data = pd.read_csv('final_train_data.csv')

	X, y = getFeatures(train_data, x_features=['log_goal', 'backers_count', 'duration_weeks'])
	X_train, X_test, y_train, y_test = splitData(X, y, 0.2, [])


	##############################
	##							##
	##		  	MODELS			##
	##							##
	##############################

	print ('VARYING MODELS')

	######## liblinear ########

	lf, clf = getLogisticRegression(X_train, y_train, multi_class='ovr')

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## lbfgs ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs')

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## saga ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='saga')

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## sag ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='sag')

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))


	##############################
	##							##
	##		  	C VALUE			##
	##							##
	##############################

	print ('VARYING C')


	######## C = 0.000001 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', C=0.000001)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))


	######## C = 0.00001 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', C=0.00001)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))


	######## C = 0.0001 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', C=0.0001)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))


	######## C = 0.001 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', C=0.001)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## C = 0.01 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', C=0.01)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## C = 0.1 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', C=0.1)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## C = 1 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', C=1)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## C = 10 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', C=10)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## C = 100 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', C=100)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## C = 1000 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', C=1000)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## C = 10000 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', C=10000)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))


	##############################
	##							##
	##		  ITERATIONS		##
	##							##
	##############################

	print ('VARYING ITERATIONS')

	######## Iterations = 1 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', max_iter=1)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## Iterations = 5 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', max_iter=5)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))


	######## Iterations = 10 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', max_iter=10)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## Iterations = 50 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', max_iter=50)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## Iterations = 100 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', max_iter=100)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))



	##############################
	##							##
	##		  TOLERANCE			##
	##							##
	##############################


	print ('VARYING TOLERANCE')

	######## Tolerance = 1e-6 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', tol=1e-6)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## Tolerance = 1e-5 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', tol=1e-5)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## Tolerance = 1e-4 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', tol=1e-4)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))


	######## Tolerance = 1e-3 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', tol=1e-3)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## Tolerance = 1e-2 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', tol=1e-2)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## Tolerance = 1e-1 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', tol=1e-1)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## Tolerance = 1e0 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', tol=1e0)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## Tolerance = 1e1 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', tol=1e1)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))


if __name__ == '__main__':
	testLR()




