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
from keras.layers import Dense, Dropout#, Merge
from keras.optimizers import Adamax, SGD
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping, ModelCheckpoint
from get_metrics import * 

train_data = pd.read_csv('final_train_data.csv')

def basemodel(input_size, rate=0.0):
    model=Sequential()
    model.add(Dense(units=1, activation='sigmoid', input_dim=input_size, kernel_initializer='random_uniform'))
    model.compile(loss='binary_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])
    return model 

def two_branches(input_size, rate = 0.2):
    model_1 = Sequential()
    model_1.add(Dense(units=input_size, activation='relu', input_dim=input_size, kernel_initializer='random_uniform'))
    model_1.add(Dropout(rate))
    
    model_2 = Sequential()
    model_2.add(Dense(units=1, activation='sigmoid', input_dim=1))
    model_2.add(Dropout(rate))
    
    model = Sequential()
    model.add(Merge([model_1, model_2], mode = 'concat'))
    model.add(Dense(units=50, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(rate))
    model.add(Dense(units=10, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(rate))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='random_uniform'))
    model.compile(loss='binary_crossentropy',
              optimizer=Adamax(),
              metrics=['accuracy'])
    return model
    
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
    X, y = getFeatures(x_features = ['log_goal','backers_count','duration_weeks'])
    # X['backers_count'] = X['backers_count'].apply(lambda x: np.log(x) if x > 0 else 0)
#    X['backers_count'] = np.sqrt(X['backers_count'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0.2, [])

    
    model = basemodel(X_train.shape[1])
    model.fit(X_train, y_train, batch_size = 128, \
                        epochs = 10, validation_split=0)

    pred = model.predict(X_test)
    pred = np.round(pred)

    metrics(y_test, pred)

    
    # model = simplemodel(X_train.shape[1])
    # class_weights = class_weight.compute_class_weight('balanced',
    #                                              np.unique(y_train),
    #                                              y_train)
    # callbacks = [EarlyStopping(monitor='val_loss', patience=2),
    #          ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    
    # model.fit(X_train, 
    #           y_train, 
    #           batch_size = 128,
    #           callbacks = callbacks,
    #           epochs = 50, validation_split=0.2) # class_weight = class_weights)        
#    model = two_branches(X_train.shape[1])
#    model.fit([X_train, X['backers_count'].values], y_train, epochs = 30, validation_split = 0.2 )
    
