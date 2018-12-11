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

from get_metrics import *

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, LSTM, Merge, TimeDistributed
from keras.optimizers import Adamax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import scale

X, y = getFeatures(x_features = ['log_goal', 'country', 'backers_count','duration_weeks',
                                 'disable_communication','clean_desc', 'sentiment', 'keywords'])

#X['backers_count'] = np.log(X['backers_count'])
X['clean_keyword'] = X['keywords'].apply(lambda x: ' '.join(x.split('-')))
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
        
# prepare tokenizer
t1 = Tokenizer()
t1.fit_on_texts(X['clean_keyword'])
vocab_size1 = len(t1.word_index) + 1
# integer encode the documents
encoded_docs1 = t1.texts_to_sequences(X['clean_keyword'])
# get the max_length for a sequence of text
input_length1 = max([len(i) for i in encoded_docs1])
# pad the sequence
padded_docs1 = pad_sequences(encoded_docs1, maxlen=input_length1, padding='post')

# create a weight matrix for words in training docs
embedding_matrix1 = zeros((vocab_size1, 50))
for word, i in t1.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix1[i] = embedding_vector
        
################### DEFINE MODELS ######################

def complex_model(input_size, rate = 0.2):
    
    # Branch 1
    B1 = Sequential()
    B1.add(Dense(units=50, activation='sigmoid', input_dim=input_size, kernel_initializer='random_uniform')) 
    B1.add(Dropout(rate))
    print(B1.summary())

    # Branch 2
    B2 = Sequential()
    e = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=input_length, trainable=False)
    B2.add(e)
    B2.add(LSTM(32, return_sequences=True, dropout = rate))
#    B2.add(Dropout(rate))
    B2.add(Flatten())
    B2.add(Dense(40, activation = 'sigmoid'))
    B2.add(Dropout(rate))
    print(B2.summary())
    
#    # Branch 3
#    B3 = Sequential()
#    e = Embedding(vocab_size1, 50, weights=[embedding_matrix1], input_length=input_length1, trainable=False)
#    B3.add(e)
#    B3.add(LSTM(10, return_sequences=True, dropout = rate))
#    B3.add(Flatten())
#    B3.add(Dense(40, activation = 'sigmoid'))
#    B3.add(Dropout(rate))
#    print(B3.summary())
    
    
    # Model
    model = Sequential()
    model.add(Merge([B1, B2], mode = 'concat'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dropout(rate))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dropout(rate))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer=Adamax(lr=0.01),
              metrics=['accuracy'])
    return model

X_train, X_test, y_train, y_test = splitData(X[['log_goal', 'disable_communication', 'backers_count',
                                                'duration_weeks','sentiment']], 
                                                y, 0.0, ['sentiment', 'disable_communication'])
#X_train['backers_count'] = X['backers_count'].apply(lambda x: np.log(x) if x > 0 else 0)

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model_word.h5', monitor='val_loss', save_best_only=True)]

model = complex_model(X_train.shape[1], 0.3)
print(model.summary())
model.fit([X_train, padded_docs], 
          y_train, 
          epochs=20, 
          callbacks = callbacks,
          validation_split=0.2)


pred = model.predict([X_train, padded_docs])
pred = np.round(pred)

metrics(y_train, pred)


