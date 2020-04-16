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
import xlsxwriter
print("Loaded models")

CATEGORIES = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
IMG_SIZE = 28
(X_train, y_train), (X_test, y_test) = emnist.load_data(type='letters')

def create_testing_data(directory, X, y, catego, size):
    for category in catego:
        path = os.path.join(directory, category)
        class_num = CATEGORIES.index(category)
        notpng = 0
        for img in os.listdir(path):
            img_file_name = os.path.join(path, img)
            if ".png" in img_file_name:
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    img_array = cv2.resize(img_array, (size, size))
                    img_array = cv2.bitwise_not(img_array)
                    X.append(img_array)
                    y.append(class_num)
                except Exception as e:
                    print("There was a problem with a png file")
            else:
                notpng = notpng + 1
        print("not .png in " + path + " is " + str(notpng))

X_test1=[]
y_test1=[]
create_testing_data("Photos/1/", X_test1, y_test1, CATEGORIES, IMG_SIZE)
create_testing_data("Photos/1_thicker/", X_test1thick, y_test1thick, CATEGORIES, IMG_SIZE)
print(len(X_test1))
print(len(y_test1))
print("Loaded datasets")

# Reshaping to format which CNN expects (batch, height, width, channels)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')
X_test1 = np.array(X_test1).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')
print("Reshaped datasets")

# normalize inputs from 0-255 to 0-1
X_test = X_test / 255.0
X_test1 = X_test1 / 255.0
print("Normalized datasets")

# one hot encode
number_of_classes = 26
y_test = y_test - 1
y_test = np_utils.to_categorical(y_test, number_of_classes)
y_test1 = np_utils.to_categorical(np.array(y_test1), number_of_classes)
print("Normalized labels")

#loading the model
model = tf.keras.models.load_model('saved_model/EMNIST_letter_model.h5')
print("Model loaded")

#Transfering the predictions data to an excel file
def create_excelfiles_probabilities(X, y, name, catego, mod):
    predictions = mod.predict(X)
    print("Predictions has been made")
    workbookname = 'Probabilities_of_' + name + '.xlsx'
    workbook = xlsxwriter.Workbook(workbookname)
    letter = 0
    worksheet = workbook.add_worksheet(catego[letter])
    row=0
    colminus=0
    label = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    for a in range(len(X)):
        if np.array_equal(y[a], label):
            worksheet.write_column(row, a - colminus, predictions[a])
        else:
            letter = letter + 1
            worksheet = workbook.add_worksheet(catego[letter])
            label = y[a]
            colminus = a
            worksheet.write_column(row, a - colminus, predictions[a])
    workbook.close()
    print("Excel file has been made")

create_excelfiles_probabilities(X_test, y_test, 'test', CATEGORIES, model)
create_excelfiles_probabilities(X_test1, y_test1, '1', CATEGORIES, model)

