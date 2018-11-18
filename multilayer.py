#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:15:32 2018

@author: maggiewu
"""

from dataset import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adagrad, Adamax, Adam, Adadelta, SGD

def multilayer(input_size, rate=0.1):
    model = Sequential()
    model.add(Dense(units=100, activation='relu', input_dim=input_size, kernel_initializer='random_uniform'))
    model.add(Dropout(rate))
    model.add(Dense(units=50, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(rate))
    model.add(Dense(units=10, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(rate))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='random_uniform'))
    model.compile(loss='binary_crossentropy',
              optimizer=Adamax(),
              metrics=['accuracy'])
    return model

if __name__ == '__main__':
    X, y = getFeatures(x_features = ['log_goal', 'country', 'currency', 'backers_count',
                                     'launched_year', 'launched_month', 'duration_weeks', 
                                     'text_polarity', 'text_subjectivity'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0.2, ['country', 'currency', 'launched_month'])

    model = multilayer(X_train.shape[1])
    model.fit(X_train, y_train, batch_size = 128,\
                       epochs = 50, validation_split=0.1)

    score = model.evaluate(X_test, y_test, batch_size=128)
    print ('- test_loss: ' + str(score[0]) + ' - test_acc: ' + str(score[1]))


# 100 neurons to avoid overfitting 
# Adamax > Adagrad > SGD 
# Adadelta ~ Adamax ~ Adam 

# Random weight initialization 
# Doesn't help much with performance 

# Lower batch size 
# Doesn't help much with performance 


