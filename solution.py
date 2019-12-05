# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:33:36 2019

@author: Griffin
"""
#rom __future__ import absolute_import, division, print_function, unicode_literals
#import functools
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
#from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
import numpy as np
#import tensorflow as tf

#from tensorflow.python.framework import ops
# ops.reset_default_graph()


def solution():
    # Load data
    testData = pd.read_csv("./data/test_data.csv")
    trainData = pd.read_csv("./data/test_data.csv")

    testLabel = testData.pop("label").to_numpy()
    trainLabel = trainData.pop("label").to_numpy()
    # print(trainLabel.shape)
    testData = testData.to_numpy()
    trainData = trainData.to_numpy()
    
    for n in range(len(testLabel)): 
        if testLabel[n] == -1: 
            testLabel[n] = 0
            
            
    for n in range(len(trainLabel)): 
        if trainLabel[n] == -1: 
            trainLabel[n] = 0
    # print(trainData.shape[2])
    # print(testLabel)
    # print(trainLabel)

    print(evaluate_model(trainData,
                   trainLabel, testData, testLabel))


def evaluate_model(trainX, trainy, testX, testy):
    trainX = np.expand_dims(trainX, axis=2) 
    trainy= np.expand_dims(trainy, axis=2) 
    testX = np.expand_dims(testX, axis=2) 
    
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)

    
    print(trainX.shape)
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs,
              batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(
        testX, testy, batch_size=batch_size, verbose=0)
    #plot_model(model, to_file='model_plot.png',
     #          show_shapes=True, show_layer_names=True)
    return accuracy


solution()
