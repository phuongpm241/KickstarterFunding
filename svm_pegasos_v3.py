from sklearn import svm
import numpy as np
import time
import pandas as pd
from math import *
from datetime import datetime
from textblob import TextBlob # sentiment analysis
from sklearn.model_selection import train_test_split

#######Pre-process the data############
train_data = pd.read_csv("train.csv")

# Fill in missing data
train_data['name'].fillna(" ")
train_data['desc'].fillna(" ")

# Convert time
date_column = ['deadline', 'state_changed_at', 'created_at', 'launched_at']
for i in date_column:
    train_data[i]=train_data[i].apply(lambda x: datetime.fromtimestamp(int(x)).strftime("%Y-%m-%d %H:%M:%S"))
    
# Take the log of goal
train_data['log_goal'] = np.log(train_data['goal'])

# Time-related features
def countQuarter(dt):
    month = int(dt[5:7])
    if month <= 3: return '01'
    elif month <= 6:return '02'
    elif month <= 9: return '03'
    else: return '04'

train_data['launched_month'] = train_data['launched_at'].apply(lambda dt: int(dt[5:7]))
train_data['launched_year'] = train_data['launched_at'].apply(lambda dt: int(dt[0:4]))
train_data['launched_quarter'] = train_data['launched_at'].apply(lambda dt: int(countQuarter(dt)))

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
    return int (dt / week)

train_data['duration_weeks'] = train_data['duration'].apply(lambda dt: measureDurationByWeek(dt))

# Keyword search
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

train_data['buzzword_count'] = train_data['desc'].apply(lambda d: countBuzzwords(d))

def sentimentAnalysis(text):
    analysis = TextBlob(str(text)).sentiment
    return analysis

train_data['text_polarity'] = train_data['desc'].apply(lambda text: sentimentAnalysis(text).polarity)
train_data['text_subjectivity'] = train_data['desc'].apply(lambda text: sentimentAnalysis(text).subjectivity)
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
#    if size == 0:
#        return onehot_X, [], y, []
    X_train, X_test, y_train, y_test = train_test_split(onehot_X, y, test_size=size, random_state = 42)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    start = time.time()
    X, y = getFeatures(x_features = ['log_goal', 'country', 'currency', 'backers_count',
                                     'duration_weeks', 'text_polarity', 'text_subjectivity'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0.2, ['country', 'currency'])
    end = time.time()
    print ("split data takes: ", end-start)
    y_train = y_train.replace(0, -1).values
    y_test = y_test.replace(0, -1).values

    max_iters = [1e8]
    linear_accs = {}
    # linear kernel
    for iters in max_iters:
        print ("Start training linear kernel...")
        linear_model = svm.SVC(kernel='linear', max_iter=iters)
        linear_model.fit(X_train, y_train)
        print ("Finish training linear kernel")
        linear_train_acc = linear_model.score(X_train, y_train)
        linear_test_acc = linear_model.score(X_test, y_test)
        print ("Linear train accuracy: ", linear_train_acc)
        print ("Linear test accuracy: ", linear_test_acc)
        linear_accs[iters] = (linear_train_acc, linear_test_acc)
    print (linear_accs)

##    # polynomial kernel
##    degrees = [1, 2, 4, 8]
##    coefs = [1e-5, 1e-1, 1, 5, 10, 25, 50, 100]
##    poly_accs = {}
##    for iters in max_iters:
##        for d in degrees:
##            for coef in coefs:
##                print ("Start training polynomial kernel...")
##                poly_model = svm.SVC(kernel='poly', degree=d, coef0=coef, max_iter = iters)
##                poly_model.fit(X_train, y_train)
##                print ("Finish training polynomial kernel")
##                poly_train_acc = poly_model.score(X_train, y_train)
##                poly_test_acc = poly_model.score(X_test, y_test)
##                print ("Polynomial train accuracy: ", poly_train_acc)
##                print ("Polynomail test accuracy: ", poly_test_acc)
##                poly_accs[(iters, d, coef)] = (poly_train_acc, poly_test_acc)
##    print (poly_accs)
##
##    # gaussian kernel
##    gaussian_accs = {}
##    gammas = [1e5]
##    for g in gammas:
##        print ("Start training gaussian kernel...")
##        gaussian_model = svm.SVC(gamma=g, kernel='rbf', max_iter = 1e5)
##        gaussian_model.fit(X_train, y_train)
##        print ("Finish training gaussian kernel")
##        gaussian_train_acc = gaussian_model.score(X_train, y_train)
##        gaussian_test_acc = gaussian_model.score(X_test, y_test)
##        print ("Gaussian train accuracy: ", gaussian_train_acc)
##        print ("Gaussian test accuracy: ", gaussian_test_acc)
##        gaussian_accs[g] = (gaussian_train_acc, gaussian_test_acc)


##    print ("Finish all kernels")
####    print ("Linear: ")
####    print ("Linear train accuracy: ", linear_train_acc)
####    print ("Linear test accuracy: ", linear_test_acc)
##    print ("Polynomial: ")
##    print (poly_accs)
##    print ("Gaussian: ")
##    print (gaussian_accs)
    
    

