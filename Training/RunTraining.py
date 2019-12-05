import numpy as np
import keras
from keras.utils import Sequence
from keras import models, layers
from keras.optimizers import Adam
from DataPreparation import object_io
from DataPreparation import data_preparation_functions as dpf
import os
import sys
import pickle
from collections import Counter
from sklearn.utils import class_weight
from matplotlib import pyplot as plt
from keras.applications import VGG16
from Training.BatchGeneration import DataGenerator
from Training.keras_models import LC_128_Binary_3Conv_512D, LC_128_Binary_5Conv_1024D
from Training.keras_models import LC_256_Binary_3Conv_512D, LC_256_Binary_5Conv_1024D
from Training.keras_models import VGG16_Transfer_Binary, VGG16_Transfer_Categorical
from Training.keras_models import LC_128_Categorical_3Conv_512D, LC_128_Categorical_5Conv_1024D
from Training.keras_models import LC_256_Categorical_3Conv_512D, LC_256_Categorical_5Conv_1024D

# reference to external directory
ext_data_path = dpf.get_external_data_directory()


def plot_history(training_history, name):
    """
    Saves training data object to external data directory.
    :param training_history: Keras history object
    :param name: The name of the model being trained
    """

    training_history_dir = os.path.join(ext_data_path, 'Training', 'TrainHistory')
    figure_dir = os.path.join(training_history_dir, 'Figures')
    history_dict_dir = os.path.join(training_history_dir, 'Dicts')

    acc = training_history.history['acc']
    val_acc = training_history.history['val_acc']
    loss = training_history.history['loss']
    val_loss = training_history.history['val_loss']

    epochs = range(len(acc))

    # plot training & validation accuracy
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(name + ' Training and validation accuracy')
    plt.legend()

    # save accuracy figure
    acc_out_png = os.path.join(figure_dir, name + '_accuracy.png')
    acc_out_pdf = os.path.join(figure_dir, name + '_accuracy.pdf')
    plt.savefig(acc_out_png, bbox_inches='tight')
    plt.savefig(acc_out_pdf, bbox_inches='tight')

    plt.show()  # show accuracy

    plt.figure()

    # plot training & validation loss
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(name + ' Training and validation loss')
    plt.legend()

    # save loss figure
    loss_out_png = os.path.join(figure_dir, name + '_loss.png')
    loss_out_pdf = os.path.join(figure_dir, name + '_loss.pdf')
    plt.savefig(loss_out_png, bbox_inches='tight')
    plt.savefig(loss_out_pdf, bbox_inches='tight')

    # show loss plot
    plt.show()

    # save accuracy and loss values from each epoch
    with open(os.path.join(history_dict_dir, name + '.dict'), 'wb') as f:
        pickle.dump(training_history, f)


def run_categorical_training(model, name, tile_size=256, channels=4, classes=26):
    """
    Trains a neural net for categorical classification.
    :param model: Model from keras_models.py
    :param name: Name of model
    :param tile_size: Size of input demensions
    :param channels: Number of input data channels (i.e. 3 for RGB)
    :param classes: Number of categorical classes
    """

    data_path = ext_data_path  # external data path

    # obtain training and validation directories
    training_dir = os.path.join(data_path, 'Data', str(tile_size), 'Training')
    validation_dir = os.path.join(data_path, 'Data', str(tile_size), 'Validation')

    # training label path - generator knows training data path
    training_label_dict_path = os.path.join(training_dir, 'Labels', 'combined_labels.dict')

    # validation label path - generator knows validation data path
    validation_label_dict_path = os.path.join(validation_dir, 'Labels', 'combined_labels.dict')

    # read training label dict and unpack items
    training_label_dict = object_io.read_object(training_label_dict_path)
    training_ids, training_labels = zip(*training_label_dict.items())

    # read validation label dict and unpack items
    validation_label_dict = object_io.read_object(validation_label_dict_path)
    validation_ids, validation_labels = zip(*validation_label_dict.items())

    # instantiate DataGenerator objects for training and validation data
    training_generator = DataGenerator(training_ids, training_label_dict,
                                       training=True, n_channels=channels,
                                       n_classes=classes, dim=(tile_size, tile_size))
    validation_generator = DataGenerator(validation_ids, validation_label_dict,
                                         training=False, n_channels=channels,
                                         n_classes=classes, dim=(tile_size, tile_size))

    print(model.summary())

    if classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    model.compile(loss=loss,
                  optimizer=Adam(lr=1e-5),
                  metrics=['acc'])

    # Fit model to generator data and output training history
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=100, shuffle=True, verbose=2)

    # save model weights
    weights_dir = os.path.join(ext_data_path, 'Training', 'Weights')
    model.save(os.path.join(weights_dir, name + '.h5'))

    plot_history(history, name)  # plot loss & accuracy


def run_binary_training(model, name, tile_size=256, channels=4, classes=2):
    """
    Trains a neural net for binary classification.
    :param model:
    :param name:
    :param tile_size:
    :param channels:
    :param classes:
    :return:
    """
    data_path = ext_data_path  # external data path

    training_dir = os.path.join(data_path, 'Data', str(tile_size), 'Training')
    validation_dir = os.path.join(data_path, 'Data', str(tile_size), 'Validation')

    # training data and labels
    training_binary_label_dict_path = os.path.join(training_dir, 'Labels', 'combined_binary_labels.dict')

    # validation data and labels
    validation_binary_label_dict_path = os.path.join(validation_dir, 'Labels', 'combined_binary_labels.dict')

    # read training label dict and unpack items
    training_label_dict = object_io.read_object(training_binary_label_dict_path)
    training_ids, training_labels = zip(*training_label_dict.items())

    # read validation label dict and unpack items
    validation_label_dict = object_io.read_object(validation_binary_label_dict_path)
    validation_ids, validation_labels = zip(*validation_label_dict.items())

    training_generator = DataGenerator(training_ids, training_label_dict,
                                       training=True, n_channels=channels,
                                       n_classes=classes, dim=(tile_size, tile_size))
    validation_generator = DataGenerator(validation_ids, validation_label_dict,
                                         training=False, n_channels=channels,
                                         n_classes=classes, dim=(tile_size, tile_size))

    print(model.summary())

    if classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    model.compile(loss=loss,
                  optimizer=Adam(lr=1e-5),
                  metrics=['acc'])

    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=100, shuffle=True, verbose=2)

    # save model
    weights_dir = os.path.join(ext_data_path, 'Training', 'Weights')
    model.save(os.path.join(weights_dir, name + '.h5'))

    plot_history(history, name)  # plot loss & accuracy history


if __name__ == '__main__':


    # define binary models
    #binary_model_128_3conv = LC_128_Binary_3Conv_512D()
    #binary_model_128_5conv = LC_128_Binary_5Conv_1024D()
    #binary_model_256_3conv = LC_256_Binary_3Conv_512D()
    binary_model_256_5conv = LC_256_Binary_5Conv_1024D()
    #binary_vgg16_transfer = VGG16_Transfer_Binary()

    # run binary training sessions
    #run_binary_training(binary_model_128_3conv, 'LC_128_Binary_3Conv_512D_100e_b', tile_size=128, channels=4)
    #run_binary_training(binary_model_128_5conv, 'LC_128_Binary_5Conv_1024D_100e_b', tile_size=128, channels=4)
    #run_binary_training(binary_model_256_3conv, 'LC_256_Binary_3conv_512D_100e_b', tile_size=256, channels=4)
    run_binary_training(binary_model_256_5conv, 'LC_256_Binary_5conv_1024D_100e_b', tile_size=256, channels=4)
    #run_binary_training(binary_vgg16_transfer, 'LC_256_Binary_VGG16_Transfer_100e_b', tile_size=256, channels=3)

    """
    # define categorical models
    categorical_model_128_3conv = LC_128_Categorical_3Conv_512D()
    categorical_model_128_5conv = LC_128_Categorical_5Conv_1024D()
    categorical_model_256_3conv = LC_256_Categorical_3Conv_512D()
    categorical_model_256_5conv = LC_256_Categorical_5Conv_1024D()
    categorical_vgg16_transfer = VGG16_Transfer_Categorical()

    # run categorical training sessions
    
    run_categorical_training(categorical_model_128_3conv, 'LC_128_Categorical_3Conv_512D_100e_b',
                             tile_size=128, channels=4, classes=26)
    run_categorical_training(categorical_model_128_5conv, 'LC_128_Categorical_5Conv_1024D_100e_b',
                             tile_size=128, channels=4, classes=26)
    run_categorical_training(categorical_model_256_3conv, 'LC_256_Categorical_3Conv_512D_100e_b',
                             tile_size=256, channels=4, classes=26)
    
    run_categorical_training(categorical_model_256_5conv, 'LC_256_Categorical_5Conv_1024D_100e_b',
                             tile_size=256, channels=4, classes=26)
    run_categorical_training(categorical_vgg16_transfer, 'LC_256_Categorical_VGG16_Transfer_100e_b',
                             tile_size=256, channels=3, classes=26)
    """
