#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:31:45 2018

@author: phuongpham
"""

from train_data import *
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adamax
from sklearn.utils import class_weight

train_data = pd.read_csv('final_train_data.csv')
def simplemodel(input_size, rate = 0.2):
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_dim=input_size, kernel_initializer='random_uniform'))
    model.add(Dropout(rate))
    model.add(Dense(units=10, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(rate))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='random_uniform'))
    model.compile(loss='binary_crossentropy',
              optimizer=Adamax(),
              metrics=['accuracy'])
    return model


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
    X, y = getFeatures(x_features = ['log_goal', 'backers_count',
                                     'duration_weeks','sentiment'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0, ['sentiment'])

    model = simplemodel(X_train.shape[1])
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
    
    model.fit(X_train, y_train, batch_size = 128,\
                       epochs = 30, validation_split=0.2) # class_weight = class_weights)        
    
    
