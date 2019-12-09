import os
import pandas as pd
import numpy as np
import tensorflow as tf
# REMOVE tensorflow. if not working for keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from keras.utils import to_categorical

#import matplotlib.pyplot as plt
#from sklearn.metrics import classification_report, confusion_matrix


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def main():

    # Load data
    testData = pd.read_csv("./data/test_data.csv")
    trainData = pd.read_csv("./data/train_data.csv")

    # Adjust data
    testLabel = testData.pop("label").to_numpy()
    trainLabel = trainData.pop("label").to_numpy()
    testData = testData.to_numpy()
    trainData = trainData.to_numpy()

    for n in range(len(testLabel)):
        if testLabel[n] == -1:
            testLabel[n] = 0

    for n in range(len(trainLabel)):
        if trainLabel[n] == -1:
            trainLabel[n] = 0

    print("Accuracy: ", evaluate_model(trainData,
                                       trainLabel, testData, testLabel))


# [26x1] Input
# []

def evaluate_model(trainX, trainy, testX, testy):
    trainX = np.expand_dims(trainX, axis=2)
    trainy = np.expand_dims(trainy, axis=2)
    testX = np.expand_dims(testX, axis=2)

    trainy = to_categorical(trainy)
    testy = to_categorical(testy)

    epochs, batch_size = 3, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=11, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))

    model.add(MaxPooling1D(pool_size=3))

    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit(trainX, trainy, epochs=epochs,
                        batch_size=batch_size)
    # evaluate model
    _, accuracy = model.evaluate(
        testX, testy, batch_size=batch_size, verbose=0)

    # create model png
    # tf.keras.utils.plot_model(model, to_file='model_plot.png',
    #                           show_shapes=True, show_layer_names=True)

    # Plot training & validation accuracy values
    # print(history.history.keys())
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    # print('Confusion Matrix')
    # y_pred = model.predict(testX)
    # y_pred = (y_pred > 0.5)
    # cm = confusion_matrix(testy, y_pred)

    #print(confusion_matrix(validation_generator.classes, y_pred))

    model.summary()
    return accuracy


main()
