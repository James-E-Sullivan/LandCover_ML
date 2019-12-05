import numpy as np
import keras
from keras.utils import Sequence
from keras import models, layers
from keras.optimizers import Adam
from DataPreparation import object_io
import os
import sys
from collections import Counter
from sklearn.utils import class_weight
from matplotlib import pyplot as plt
from DataPreparation import data_preparation_functions as dpf

script_path = os.path.dirname(os.path.realpath(sys.argv[0]))

ext_data_path = dpf.get_external_data_directory()


class DataGenerator(Sequence):
    """
    DataGenerator class copied/modified from
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(256,256), n_channels=4,
                 n_classes=2, shuffle=True, training=True, tile_size=256):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.training = training
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # .npy file directory
        train_npy_dir = os.path.join(ext_data_path, 'Data', str(self.dim[0]), 'Training', 'Data_NPY')
        val_npy_dir = os.path.join(ext_data_path, 'Data', str(self.dim[0]), 'Validation', 'Data_NPY')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # look for .npy file location in training or validation folders
            if self.training is True:
                npy_file_path = os.path.join(train_npy_dir, ID + '.npy')
            else:
                npy_file_path = os.path.join(val_npy_dir, ID + '.npy')

            raw_npy = np.load(npy_file_path)
            raw_npy = raw_npy / 255  # divide each value by 255
            reshaped_data = np.transpose(raw_npy, (1, 2, 0))  # channels last

            # if we need to convert to 3-band data
            if self.n_channels == 3:
                # only take first 3 entries of last array dimension (bands)
                reshaped_data = reshaped_data[:, :, :3]

            # Store sample
            X[i,] = reshaped_data

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

