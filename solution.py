import tensorflow as tf
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import Conv1D
from keras.models import Sequential
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# For pydot model
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def main():

    # Load data
    dataset = pd.read_csv("data.txt")
    dataset.columns = ['name', 'games', 'games_started', 'minutes_played',
                       'field_goals', 'field_goals_attempted', 'field_goal_percent', 'two_pointers', 'two_pointers_attempted', 'two_pointers_percent', 'three_pointers',
                       'three_pointers_attempted', 'three_pointers_percent', 'free_throws', 'free_throws_attempted', 'free_throws_percent', 'offensive_rebounds',
                       'defensive_rebounds', 'total_rebounds', 'assists', 'steals', 'blocks', 'turn_overs', 'personal_fouls', 'points', 'classification', ]
    X = dataset.drop(['name', 'classification'], axis=1)
    y = dataset['classification']

    epochs = 20

    # Fill holes in data
    pd.DataFrame(X).fillna(0, inplace=True)
    pd.DataFrame(y).fillna(0, inplace=True)

    print("Data entries (rows):", X.shape[0])
    print("Features (columns):", X.shape[1])

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape)

    # Create model
    model = Sequential()
    model.add(Dense(32, input_dim=24, activation='relu'))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Set learning rate
    adam = Adam(learning_rate=.01)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy', metrics=['accuracy'])

    # Fit model
    history = model.fit(X_train, y_train, epochs=epochs,
                        shuffle=True, validation_split=0.25)

    # Evaluate model
    accuracy = model.evaluate(X_test, y_test)

    # print(model.layers[3].get_weights())
    print(model.summary())

    # Confusion matrix from predictions
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)

    # Output
    print("Epochs:", epochs)
    print("Loss, Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("[True Negative | False Negatives]")
    print("[False Positives | True Positives]")

    # print(model.get_config())

    # Create model.png
    plot_model(
        model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir="LR", expand_nested=True)

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


main()
