# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Visualization
import seaborn as sns 
import re
import matplotlib.pyplot as plt
# Datetime
from datetime import datetime
# Sklearn import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import preprocessing
# Text processing
from textblob import TextBlob
import string
from math import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

train_data = pd.read_csv("train.csv")

### Find if any entries are null
##for i in train_data.columns:
##    print(i, train_data[i].isnull().sum().sum())

# Fill in missing data by empty string
train_data['name'].fillna(" ")
train_data['desc'].fillna(" ")

# Convert UNIX time format to standard time format
date_column = ['deadline', 'state_changed_at', 'created_at', 'launched_at']
for i in date_column:
    train_data[i]=train_data[i].apply(lambda x: datetime.fromtimestamp(int(x)).strftime("%Y-%m-%d %H:%M:%S"))


#Remove some of the outliers and replot the histograms
P = np.percentile(train_data['goal'], [0, 95])
new_goal = train_data[(train_data['goal'] > P[0]) & (train_data['goal'] < P[1])]

train_data['log_goal'] = np.log(train_data['goal'])

#Remove some of the outliers and replot the histograms
P_backer = np.percentile(train_data['backers_count'], [0, 95])
new_backers = train_data[(train_data['backers_count'] > P_backer[0]) & (train_data['backers_count'] < P_backer[1])]

# with respect to launched time 
def countQuarter(dt):
    month = int(dt[5:7])
    if month <= 3: return '01'
    elif month <= 6:return '02'
    elif month <= 9: return '03'
    else: return '04'
train_data['launched_month'] = train_data['launched_at'].apply(lambda dt: dt[5:7])
train_data['launched_year'] = train_data['launched_at'].apply(lambda dt: dt[0:4])
train_data['launched_quarter'] = train_data['launched_at'].apply(lambda dt: countQuarter(dt))


def measureDuration(dt): # Duration in hours
    launch = datetime.strptime(dt[0], "%Y-%m-%d %H:%M:%S")
    deadline = datetime.strptime(dt[1], "%Y-%m-%d %H:%M:%S")
    difference = deadline-launch
    hr_difference = int (difference.total_seconds() / 3600)
    return hr_difference
train_data['duration'] = train_data[['launched_at', 'deadline']].apply(lambda dt: measureDuration(dt), axis=1)

def measureDurationByWeek(dt):
    # count by hr / week 
    week = 168 
    return int (dt / 168)
train_data['duration_weeks'] = train_data['duration'].apply(lambda dt: measureDurationByWeek(dt))

def getFeatures(x_features, y_feature): 
    X = train_data[x_features]
    y = train_data[y_feature]
    return X, y

def splitData(X, y, size): 
    onehot_X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(onehot_X, y, test_size=size, random_state = 42)
    return X_train, X_test, y_train, y_test

# Project name in alphabetical order
def parseName(name):
    if str(name)[0] not in string.ascii_lowercase + string.ascii_uppercase: 
        return '*'
    else:
        return str(name)[0].lower()

# Keyword Search 

buzzwords = ['app', 'platform', 'technology', 'service', 'solution', 'data', 
            'manage', 'market', 'help', 'mobile', 'users', 'system', 'software', 
           'customer', 'application', 'online', 'web', 'create', 'health', 
           'provider', 'network', 'cloud', 'social', 'device', 'access']

def countBuzzwords(desc):
    lowerCase = str(desc).lower() 
    count = 0
    for bw in buzzwords: 
        count += lowerCase.count(bw)
    return count

def Pegasos_SVM(X, Y, lambda_constant, max_epochs):
    num_data, d = X.shape
    t = 0
    w = np.zeros(d)
    offset = 0
    epoch = 0
    while epoch < max_epochs:
        for index, row in X.iterrows():
            t += 1
            eta = 1.0/(t*lambda_constant)
            if np.dot(Y.loc[index], (np.dot(w, row.values) + offset)) < 1.0:
                w = (1.0-eta*lambda_constant)*w + eta*Y.loc[index]*row.values
                offset = offset + eta*Y.loc[index]
            else:
                w = (1.0-eta*lambda_constant)*w
        epoch+= 1
    return (w, offset)

def predict_linearSVM(w, offset, new_data):
    return np.dot(w, new_data)+offset

def find_accuracy(X, y, w, offset):
    num_data, d = X.shape
    predicted = []
    weight = np.ones(y.shape[0])
    correct = 0
    for index, row in X.iterrows():
        if predict_linearSVM(w, offset, row.values)* y.loc[index] > 0:
            correct += 1
    return correct/num_data

def Pegasos_KernelSVM(X, Y, lambda_constant, K, max_epochs, gamma):
    X = X.values
    Y = Y.values
    num_data, d = X.shape
    t = 0
    alphas = np.zeros(num_data)
    num_epoch = 0
    while num_epoch < max_epochs:
        for i in range(num_data):
            t += 1
            eta = 1.0/(t*lambda_constant)
            total = 0
            for j in range(num_data):
                total += alphas[j]*Y[j]*K(X[i], X[j], gamma)
            if Y[i]* total<1:
                alphas[i] = (1.0-eta*lambda_constant)*alphas[i] + eta
            else:
                alphas[i] = (1.0-eta*lambda_constant)*alphas[i]
        num_epoch += 1

    return alphas


def predict_KernelSVM(X, Y, alphas, K, gamma, new_data):
    X = X.values
    Y = Y.values
    num_data, d = X.shape
    total = 0
    for i in range(num_data):
        total += alphas[i]*Y[i]*K(new_data, X[i], gamma)
    return total

def find_accuracyKernel(X, y, alphas, K, gamma):
    num_data, d = X.shape
    predicted = []
    correct = 0
    for index, row in X.iterrows():
        if predict_KernelSVM(X, Y, alphas, K, gamma, row.values)* y.loc[index] > 0:
            correct += 1
    return correct/num_data


# kernels:
linear_kernel_fn = lambda p1, p2, constant: np.dot(p1, p2) + constant
polynomial_kernel_fn = lambda p1, p2, slope, constant, degree: np.power(slope*np.dot(p1, p2)+constant, d)
gaussian_kernel_fn = lambda p1, p2, gamma: exp(-gamma*np.dot(p1-p2, p1-p2))



x_features = ['log_goal','country', 'currency', 'backers_count', 'launched_year', 'launched_month', 'duration_weeks']
y_feature = 'final_status'


max_epochs = 50

X, y = getFeatures(x_features, y_feature)
X_train, X_test, y_train, y_test = splitData(X, y, 0.2)
y_train = y_train.replace(0, -1)
y_test = y_test.replace(0, -1)

# linear kernel 
lambda_constant = 2e-1
w, offset = Pegasos_SVM(X_train, y_train, lambda_constant, max_epochs)

train_score = find_accuracy(X_train, y_train, w, offset)
test_score = find_accuracy(X_test, y_test, w, offset)
print ("SVM train score: ", train_score)
print ("SVM test score: ", test_score)


gamma = 2e2;
alphas_gaussian = Pegasos_KernelSVM(X_train, y_train, lambda_constant, gaussian_kernel_fn, max_epochs, gamma)
train_score_gaussian = find_accuracyKernel(X_train, y_train, alphas_gaussian, gaussian_kernel_fn, gamma)
test_score_gaussian = find_accuracyKernel(X_test, y_test, alphas_gaussian, gaussian_kernel_fn, gamma)

print ("Gaussian train score: ", train_score_gaussian)
print ("Gaussian test score: ", test_score_gaussian)






