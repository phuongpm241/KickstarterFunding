#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:31:45 2018

@author: phuongpham
"""

from dataset import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adamax



def baseline(input_size, rate=0.1):
    model = Sequential()
    model.add(Dense(units=1000, activation='relu', input_dim=input_size))
    model.add(Dropout(rate))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer=Adamax(),
              metrics=['accuracy'])
    return model

if __name__ == '__main__':
    X, y = getFeatures(x_features = ['log_goal','country', 'currency', 'backers_count',
                                     'launched_year', 'launched_month', 'duration_weeks', 
                                     'buzzword_count'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0, ['country', 'currency', 'launched_month'])

    model = baseline(X_train.shape[1])
    model.fit(X_train, y_train, batch_size = 128,\
                       epochs = 50, validation_split=0.2)        
    
    
