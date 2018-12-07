import matplotlib.pyplot as plt
import pylab as pl
import numpy as np 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
import random

from logistic_regression import getLogisticRegression, getFeatures, splitData

def plot3D(X_train, y_train, pred, coef):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	log_goal = X_train[:,0]
	backers_count = X_train[:,1]
	duration_weeks = X_train[:,2]

	colors = list() 
	markers = list() 

	for p in range(len(pred)):
		if pred[p] != y_train[p]:
			colors.append('r')
		else:
			colors.append('g')

	log_goal_max, log_goal_min = max(log_goal), min(log_goal)
	backers_max, backers_min = max(backers_count), min(backers_count)
	duration_max, duration_min = max(duration_weeks), min(duration_weeks)

        # normal is coef
        # find point to plot decision surface
	random_backer = random.randint(backers_min, backers_max)
	random_duration = random.randint(duration_min, duration_max)
	random_log_goal = findThirdCoord(coef, random_backer, random_duration)
	point = np.array([random_log_goal, random_backer, random_duration])
	
	


	xx, yy, zz = np.mgrid[log_goal_min:log_goal_max:(log_goal_max-log_goal_min)/100,
						backers_min:backers_max:(backers_max-backers_min)/100,
						duration_min:duration_max:(duration_max-duration_min)/100]

        # plot decision boundary
	plt3d = plt.figure().gca(projection='3d')
	plt3d.plot_surface(xx, yy, zz)

        # make sure next plot doesn't overwrite the first plot
	ax = plt.gca()
	ax.hold(True)

        # plot data on same plot
	ax.scatter(log_goal, backers_count, duration_weeks, c=colors)
	ax.set_xlabel('Log Goal')
	ax.set_ylabel('Backers Count')
	ax.set_zlabel('Duration Weeks')

	plt.show()
	

def findThirdCoord(coefs, y_coord, z_coord):
        return (0.0 - coefs[1]*y_coord + coefs[2]*z_coord)/coefs[0]

if __name__ == '__main__':
	train_data = pd.read_csv('final_train_data.csv')

	X, y = getFeatures(train_data=train_data, x_features=['log_goal', 'backers_count', 'duration_weeks'])
	X_train, X_test, y_train, y_test = splitData(X, y, 0.2, [])

	lr, clf = getLogisticRegression(X_train, y_train, solver='lbfgs')

	pred = clf.predict(X_train)

	coef = lr.coef_

	plot3D(X_train, y_train, pred, coef)

