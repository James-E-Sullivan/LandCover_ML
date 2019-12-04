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
#ext_data_path = os.path.dirname(script_path)
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
            #X[i,] = np.load(npy_file_path)
            X[i,] = reshaped_data

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


if __name__ == '__main__':

    name = 'combined'

    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    ext_data_path = os.path.dirname(script_path)
    training_dir = os.path.join(ext_data_path, 'Data', 'Training')
    validation_dir = os.path.join(ext_data_path, 'Data', 'Validation')

    # training data and labels
    training_npy_dir = os.path.join(training_dir, 'Data_NPY')
    training_label_dict_path = os.path.join(training_dir, 'Labels', 'combined_labels.dict')
    training_binary_label_dict_path = os.path.join(training_dir, 'Labels', 'combined_binary_labels.dict')

    # validation dat and labels
    validation_npy_dir = os.path.join(validation_dir, 'Data_NPY')
    validation_label_dict_path = os.path.join(validation_dir, 'Labels', 'combined_labels.dict')
    validation_binary_label_dict_path = os.path.join(validation_dir, 'Labels', 'combined_binary_labels.dict')

    # read training label dict and unpack items
    training_label_dict = object_io.read_object(training_binary_label_dict_path)
    training_ids, training_labels = zip(*training_label_dict.items())

    # read validation label dict and unpack items
    validation_label_dict = object_io.read_object(validation_binary_label_dict_path)
    validation_ids, validation_labels = zip(*validation_label_dict.items())




    '''
    # dict for binary classification of developed land
    training_binary_label_dict = {}
    validation_binary_label_dict = {}

    for k, v in training_label_dict.items():
        if 1 < v < 6:
            training_binary_label_dict[k] = 1
        else:
            training_binary_label_dict[k] = 0

    for k, v in validation_label_dict.items():
        if 1 < v < 6:
            validation_binary_label_dict[k] = 1
        else:
            validation_binary_label_dict[k] = 0
    '''




    # define class weights
    training_class_weights = {}
    validation_class_weights = {}

    for i in range(0, 26):
        training_class_weights[i] = 0
        validation_class_weights[i] = 0

    training_class_count = Counter(training_label_dict.values())
    validation_class_count = Counter(validation_label_dict.values())

    training_samples = len(training_label_dict)
    validation_samples = len(validation_label_dict)

    training_class_weights.update(training_class_count)
    validation_class_weights.update(validation_class_count)

    for key, value in training_class_weights.items():
        training_class_weights[key] = value / training_samples

    for key, value in validation_class_weights.items():
        validation_class_weights[key] = value / validation_samples


    weight_test = [1, 1, 1, 2]

    class_weight = class_weight.compute_class_weight('balanced', np.unique(weight_test),
                                                     weight_test)


    # instantiate training and validation generator objects
    training_generator = DataGenerator(training_ids, training_label_dict, training=True)
    validation_generator = DataGenerator(validation_ids, validation_label_dict, training=False)



    # The model is augmented code from:
    # https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/
    # 5.2-using-convnets-with-small-datasets.ipynb

    # sequential model with 3 Conv2d + MaxPooling2D layers
    model = models.Sequential()

    # layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(128, 128, 4),
                            data_format="channels_last"))
    model.add(layers.MaxPooling2D((2, 2)))
   # model.add(layers.Dropout(0.20))

    # layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.30))

    # layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.4))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-5),
                  metrics=['acc'])

    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=50, shuffle=True, verbose=2)

    # save model
    model.save('land_cover_' + name + '_3_conv_512D_binary.h5')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # plot training & validation accuracy
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.show()  # show accuracy

    plt.figure()

    # plot training & validation loss
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    # show accuracy and loss plots
    plt.show()
