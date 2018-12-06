#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:39:32 2018

@author: maggiewu
"""

from dataset import *
from parseData import *
from sklearn.linear_model import LogisticRegression

def getLogisticRegression(X_train, y_train, solver='liblinear', C=1.0, max_iter=100, multi_class='multinomial'): 
	lr = LogisticRegression(random_state=0, solver=solver, C=C, max_iter=max_iter, multi_class=multi_class)
	clf = lr.fit(X_train, y_train)
	return lr, clf 

def accuracy(clf, X_train, y_train, X_test, y_test): 
	return clf.score(X_train, y_train), clf.score(X_test, y_test)

if __name__ == '__main__':
	train_data = pd.read_csv('train.csv')
	train_data = parseData(train_data)

	# X, y = getFeatures(x_features=['log_goal','country','currency', 'backers_count', 
	# 	'launched_year', 'launched_month', 'duration_weeks'])

	# X_train, X_test, y_train, y_test = splitData(X, y, 0.2, ['country', 'currency', 'launched_month'])


	X, y = getFeatures(x_features=['log_goal', 'backers_count', 'duration_weeks'])

	X_train, X_test, y_train, y_test = splitData(X, y, 0.2, [])



	##############################
	##							##
	##		  	MODELS			##
	##							##
	##############################

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


	##############################
	##							##
	##		  ITERATIONS		##
	##							##
	##############################

	######## C = 100 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', max_iter=100)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## C = 500 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', max_iter=500)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))

	######## C = 1000 ########

	lf, clf = getLogisticRegression(X_train, y_train, solver='lbfgs', max_iter=1000)

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))




