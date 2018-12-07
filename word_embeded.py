#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:25:09 2018

@author: phuongpham
"""


# Data management
from train_data import *
import pandas as pd

# Numpy
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, LSTM, Merge, TimeDistributed
from keras.optimizers import Adamax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Early stopping condition
from keras.callbacks import EarlyStopping, ModelCheckpoint

X, y = getFeatures(x_features = ['log_goal', 'country', 'backers_count','duration_weeks',
                                 'clean_desc', 'sentiment'])

#X['clean_keyword'] = X['keywords'].apply(lambda x: ' '.join(x.split('-')))
docs = X['clean_desc'].apply(str)

#### Prepare the description 
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
# get the max_length for a sequence of text
input_length = max([len(i) for i in encoded_docs])
# pad the sequence
padded_docs = pad_sequences(encoded_docs, maxlen=input_length, padding='post')


# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B/glove.6B.50d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 50))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
        
################### DEFINE MODELS ######################

def complex_model(input_size, rate = 0.2):
    
    # Branch 1
    B1 = Sequential()
    B1.add(Dense(units=50, activation='relu', input_dim=input_size, kernel_initializer='random_uniform')) 
    B1.add(Dropout(rate))
    print(B1.summary())

    # Branch 2
    B2 = Sequential()
    e = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    B2.add(e)
    B2.add(LSTM(10, return_sequences=True, dropout = rate))
#    B2.add(Dropout(rate))
    B2.add(Flatten())
    B2.add(Dense(50, activation = 'relu'))
    B2.add(Dropout(rate))
    print(B2.summary())
    
    # Branch 3
    B3 = Sequential()
    e = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    
    
    # Model
    model = Sequential()
    model.add(Merge([B1, B2], mode = 'concat'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dropout(rate))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dropout(rate))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer=Adamax(lr=0.0005),
              metrics=['accuracy'])
    return model

X_train, X_test, y_train, y_test = splitData(X[['log_goal', 'backers_count',
                                                'duration_weeks','sentiment', 'country']], 
                                                y, 0, ['sentiment', 'country'])

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model = complex_model(X_train.shape[1], 0.3)
print(model.summary())
model.fit([X_train,padded_docs], 
          y_train, 
          epochs=50, 
          callbacks = callbacks,
          validation_split=0.2)


