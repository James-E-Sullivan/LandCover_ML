from sklearn import svm
from sklearn import metrics
import os
import sys
import numpy as np
from DataPreparation import object_io
from DataPreparation import data_preparation_functions as dpf
from sklearn.linear_model import SGDClassifier
import collections
import random

tile_size = 256

script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
#ext_data_path = os.path.dirname(script_path)
ext_data_path = dpf.get_external_data_directory()



training_dir = os.path.join(ext_data_path, 'Data', str(tile_size), 'Training')
validation_dir = os.path.join(ext_data_path, 'Data', str(tile_size), 'Validation')

training_binary_label_dict_path = os.path.join(training_dir, 'Labels', 'combined_binary_labels.dict')
validation_binary_label_dict_path = os.path.join(validation_dir, 'Labels', 'combined_binary_labels.dict')

# read training label dict and unpack items
training_label_dict = object_io.read_object(training_binary_label_dict_path)
training_ids, training_labels = zip(*training_label_dict.items())

# read validation label dict and unpack items
validation_label_dict = object_io.read_object(validation_binary_label_dict_path)
validation_ids, validation_labels = zip(*validation_label_dict.items())


training_npy_dir = os.path.join(training_dir, 'Data_NPY')
validation_npy_dir = os.path.join(validation_dir, 'Data_NPY')


'''
counter = 0
training_list = []
for a in training_label_dict.keys():
    counter += 1
    if counter == 1000:
        break

    training_list.append(a)

counter = 0
validation_list = []
for b in validation_label_dict.keys():
    counter += 1
    if counter == 1000:
        break

    validation_list.append(b)

t = collections.Counter(training_label_dict)
v = collections.Counter(validation_label_dict)

print("Training Counter:\n", training_list)
print("\nValidation Counter:\n", validation_list)
'''

def get_svm_data(label_dict, training=True, dim=(tile_size, tile_size), n_channels=4, batch_size=32):

    X = np.empty((1000, (tile_size * tile_size * n_channels)))
    y = np.empty(1000, dtype=int)

    ids, labels = zip(*label_dict.items())
    ids = list(ids)
    random.shuffle(ids)  # randomize IDs to reduce imbalances

    for i, ID in enumerate(ids):

        # look for .npy file location in training or validation folders
        if training is True:
            npy_file_path = os.path.join(training_npy_dir, ID + '.npy')
        else:
            npy_file_path = os.path.join(validation_npy_dir, ID + '.npy')

        raw_npy = np.load(npy_file_path)
        raw_npy = raw_npy / 255  # divide each value by 255
        reshaped_data = np.transpose(raw_npy, (1, 2, 0))  # channels last

        reshaped_data = reshaped_data.reshape((1, -1))

        # if we need to convert to 3-band data
        if n_channels == 3:
            # only take first 3 entries of last array dimension (bands)
            reshaped_data = reshaped_data[:, :, :3]

        # Store sample
        X[i,] = reshaped_data

        # Store class
        y[i] = label_dict[ID]

        if i == 999:
            break

    return X, y


X_train, y_train = get_svm_data(training_label_dict, training=True)
X_test, y_test = get_svm_data(validation_label_dict, training=False)

print('Training Data Shape:', X_train.shape)
print('Training Labels Shape', y_train.shape)
print('Test Data Shape', X_test.shape)
print('Test Labels Shape', y_test.shape)

kernel = 'poly'

classifier = svm.SVC(kernel=kernel, gamma='auto')

print('Fitting classifier to training data. Kernel=' + kernel)

# train the model using training set
classifier.fit(X_train, y_train)

# predict the response for test dataset
y_pred = classifier.predict(X_test)

# model accuracy
print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))

