import matplotlib.pyplot as plt
import pylab as pl
import numpy as np 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
import random

from logistic_regression import getLogisticRegression, getFeatures, splitData, accuracy

def plot3D(X_train, y_train, pred, coef):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	log_goal = X_train[:,0]
	backers_count = X_train[:,1]
	duration_weeks = X_train[:,2]

	colors = list() 
	markers = list() 

	new_log_goal, new_backers_count, new_duration_weeks = list(), list(), list() 

	for p in range(len(pred)):
		if pred[p] != y_train[p]: 
			if pred[p] == 1: 
				colors.append('r')
			else:
				colors.append('b')
			new_log_goal.append(X_train[:,0][p])
			new_backers_count.append(X_train[:,1][p])
			new_duration_weeks.append(X_train[:,2][p])

	log_goal, backers_count, duration_weeks = new_log_goal, new_backers_count, new_duration_weeks

	# for p in range(len(pred)):
	# 	if pred[p] == y_train[p]: 
	# 		if pred[p] == 0: 
	# 			colors.append('b')
	# 		else:
	# 			colors.append('g')
	# 	else: 
	# 		colors.append('r')

	log_goal_max, log_goal_min = max(log_goal), min(log_goal)
	backers_max, backers_min = max(backers_count), min(backers_count)
	duration_max, duration_min = max(duration_weeks), min(duration_weeks)

	xx, yy = np.mgrid[log_goal_min:log_goal_max:(log_goal_max-log_goal_min)/100,
						backers_min:500:(500-backers_min)/100]
	zz = (-coef[0]*xx - coef[1]*yy) * 1./coef[2]

        # plot decision boundary
	plt3d = plt.figure()
	ax = plt3d.add_subplot(111, projection='3d')
	ax.plot_surface(xx, yy, zz)

        # make sure next plot doesn't overwrite the first plot


        # plot data on same plot
	ax.scatter(log_goal, backers_count, duration_weeks, c=colors)
	ax.set_xlabel('Log Goal')
	ax.set_ylabel('Backers Count')
	ax.set_zlabel('Duration Weeks')

	plt.show()

def plot2D(X_train, y_train, pred):
	log_goal = X_train[:,0]
	backers_count = X_train[:,1]

	new_log_goal, new_backers_count = list(), list() 

	colors = list() 
	false_pos, false_neg = 0, 0
	count = 0

	for p in range(len(pred)):
		if pred[p] != y_train[p]: 
			if backers_count[p] == 0: 
				count += 1
			if pred[p] == 1: 
				# false positive
				colors.append('r')
				false_pos += 1 
			else: 
				# false negative
				colors.append('b')
				false_neg += 1 

			new_log_goal.append(log_goal[p])
			new_backers_count.append(backers_count[p])

	print ('backer = 0: ' + str(count))
	print ('false positive: ' + str(false_pos / (false_pos + false_neg)))
	print ('false negative: ' + str(false_neg / (false_pos + false_neg)))


	plt.scatter(new_log_goal, new_backers_count, c=colors, alpha=0.1, marker='.')
	plt.ylabel('Backers Count')
	plt.xlabel('Log Goal')

	plt.show() 
	

def findThirdCoord(coefs, y_coord, z_coord):
        return (0.0 - coefs[1]*y_coord + coefs[2]*z_coord)/coefs[0]

if __name__ == '__main__':
	train_data = pd.read_csv('final_train_data.csv')

	X, y = getFeatures(train_data=train_data, x_features=['log_goal', 'backers_count', 'duration_weeks'])
	X_train, X_test, y_train, y_test = splitData(X, y, 0.2, [])

	lr, clf = getLogisticRegression(X_train, y_train, solver='lbfgs')

	train_score, test_score = accuracy(clf, X_train, y_train, X_test, y_test)
	print ('- train_acc: ' + str(train_score) + ' - test_acc: ' + str(test_score))


	pred = clf.predict(X_train)

	# coef = lr.coef_[0]*1000

	# plot3D(X_train, y_train, pred, coef)
	plot2D(X_train, y_train, pred)

