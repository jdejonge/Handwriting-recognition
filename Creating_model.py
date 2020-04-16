from extra_keras_datasets import emnist
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import np_utils

print("Loaded models")
IMG_SIZE = 28
(X_train, y_train), (X_test, y_test) = emnist.load_data(type='letters')
print("Loaded datasets")

# Reshaping to format which CNN expects (batch, height, width, channels)
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')
print("Reshaped datasets")

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255.0
print("Normalized datasets")

# one hot encode
number_of_classes = 26
y_train = y_train - 1
y_train = np_utils.to_categorical(y_train, number_of_classes)

#making the model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1:])))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(number_of_classes))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10)
model.save('saved_model/EMNIST_letter_model.h5')
print("Model saved")

