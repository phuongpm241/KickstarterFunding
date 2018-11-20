#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:23:59 2018

@author: phuongpham
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from textblob import TextBlob # sentiment analysis
# Datetime
from datetime import datetime

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

########## Format dataset ##########
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


