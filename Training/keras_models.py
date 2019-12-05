import tensorflow as tf
import sys
import os
import numpy as np
from DataPreparation import object_io
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG16
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
import matplotlib.pyplot as plt
import pandas as pd
import json

# suppress deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# get path of dataset
script_path = os.path.dirname(os.path.realpath(sys.argv[0]))


def LC_128_Binary_3Conv_512D(weights_path=None):

    # sequential model with 3 Conv2d + MaxPooling2D layers
    model = models.Sequential()

    # layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(128, 128, 4),
                            data_format="channels_last"))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def LC_128_Binary_5Conv_1024D(weights_path=None):

    # sequential model with 3 Conv2d + MaxPooling2D layers
    model = models.Sequential()

    # layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(128, 128, 4),
                            data_format="channels_last"))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 4
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 5
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))    # added
    model.add(layers.Dropout(0.5))                      # added
    model.add(layers.Dense(2, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def LC_256_Binary_3Conv_512D(weights_path=None):
    # sequential model with 3 Conv2d + MaxPooling2D layers
    model = models.Sequential()

    # layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(256, 256, 4),
                            data_format="channels_last"))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def LC_256_Binary_5Conv_1024D(weights_path=None):

    # sequential model with 3 Conv2d + MaxPooling2D layers
    model = models.Sequential()

    # layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(256, 256, 4),
                            data_format="channels_last"))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.10))

    # layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.20))

    # layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.3))

    # layer 4
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.4))

    # layer 5
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.5))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))    # added
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))                      # added
    model.add(layers.Dense(2, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def VGG16_Transfer_Binary():
    VGG16_Net = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(256, 256, 3))

    VGG16_Net.trainable = False
    VGG16_Net.summary()

    model = models.Sequential()
    model.add(VGG16_Net)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def LC_128_Categorical_3Conv_512D(weights_path=None):
    # sequential model with 3 Conv2d + MaxPooling2D layers
    model = models.Sequential()

    # layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(128, 128, 4),
                            data_format="channels_last"))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(26, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def LC_128_Categorical_5Conv_1024D(weights_path=None):

    # sequential model with 3 Conv2d + MaxPooling2D layers
    model = models.Sequential()

    # layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(128, 128, 4),
                            data_format="channels_last"))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.10))

    # layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.20))

    # layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.3))

    # layer 4
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.4))

    # layer 5
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.5))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))    # added
    model.add(layers.Dropout(0.5))                      # added
    model.add(layers.Dense(26, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def LC_256_Categorical_5Conv_1024D(weights_path=None):

    # sequential model with 3 Conv2d + MaxPooling2D layers
    model = models.Sequential()

    # layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(256, 256, 4),
                            data_format="channels_last"))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.10))

    # layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.20))

    # layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.3))

    # layer 4
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.4))

    # layer 5
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.5))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))    # added
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))                      # added
    model.add(layers.Dense(26, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def LC_256_Categorical_3Conv_512D(weights_path=None):
    # sequential model with 3 Conv2d + MaxPooling2D layers
    model = models.Sequential()

    # layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(256, 256, 4),
                            data_format="channels_last"))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(26, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def VGG16_Transfer_Categorical():
    VGG16_Net = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(256, 256, 3))

    VGG16_Net.trainable = False
    VGG16_Net.summary()

    model = models.Sequential()
    model.add(VGG16_Net)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(26, activation='softmax'))

    return model


if __name__ == '__main__':

    test_model = LC_256_Categorical_3Conv_512D()
    print(test_model.summary())





