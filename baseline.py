#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:31:45 2018

@author: phuongpham
"""

from train_data import *
import pandas as pd
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Merge
from keras.optimizers import Adamax, SGD
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Model
from keras import backend as K


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
    model.add(Dense(units=20, activation='relu', kernel_initializer='random_uniform'))
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
    model.add(Dense(units=20, activation='relu', input_dim=input_size, kernel_initializer='random_uniform'))
#    model.add(Dropout(rate))
    model.add(Dense(units=10, activation='relu', kernel_initializer='random_uniform'))
#    model.add(Dropout(rate))
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

#def get_activations(model, layer, X_batch):
#    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
#    activations = get_activations(X_batch) # same result as above
#    return activations

if __name__ == '__main__':
    X, y = getFeatures(x_features = ['log_goal','backers_count','duration_weeks'])
    X['backers_count'] = X['backers_count'].apply(lambda x: np.log(x) if x > 0 else 0)
#    X['backers_count'] = np.sqrt(X['backers_count'])
    X_train, X_test, y_train, y_test = splitData(X, y, 0, [])
    
#    model = basemodel(X_train.shape[1])
#    model.fit(X_train, y_train, batch_size = 128, \
#                        epochs = 30, validation_split=0.2)


    model = simplemodel(X_train.shape[1])
    get_layer_output = K.function([model.layers[0].input],
                                      [model.layers[1].output])
#    outputs = []
#    print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: outputs.append())
    
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    
    print(X_train.shape[1])
    print(model.summary()) 

    history = model.fit(X_train, y_train, 
                        batch_size = 128, 
                        epochs = 50, 
                        validation_split=0.2) 
    
#    # Plot training & validation accuracy values
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('Model accuracy')
#    plt.ylabel('Accuracy')
#    plt.xlabel('Epoch')
#    plt.legend(['Train', 'Test'], loc='upper left')
#    plt.show()
#    
#    # Plot training & validation loss values
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('Model loss')
#    plt.ylabel('Loss')
#    plt.xlabel('Epoch')
#    plt.legend(['Train', 'Test'], loc='upper left')
#    plt.show()     
    
#    plot_model(model, to_file='model.png')
#    model = two_branches(X_train.shape[1])
#    model.fit([X_train, X['backers_count'].values], y_train, epochs = 30, validation_split = 0.2 )
    
