import tensorflow as tf
from DataPreparation import object_io as obj_io
import os
import sys
import numpy as np
import gc


def label_changer(label, classes):
    """
    Converts integer labels to proper format for tensorflow labels
    :param label:
    :param classes: Number of output classes for the NN
    :return:
    """
    empty_classes = np.zeros(classes, dtype="float32")
    empty_classes[label] = 1.0  # label corresponds to array position
    return empty_classes


def labels_tf(label_set, classes):

    # this isn't efficient, but it works - update if there is time
    updated_label_list = []
    for label in label_set:
        updated_label_list.append(label_changer(label, classes))

    return np.asarray(updated_label_list)


def load_dataset(ds):
    """
    Load dataset into training and testing sets
    :param ds:
    :return:
    """

    # break dataset into training and testing sets
    print("Input Data Shape: ", ds[0].shape)
    input_data_points = ds[0].shape[0]
    training_data_points = int(input_data_points * 0.9)
    testing_data_points = input_data_points - training_data_points
    print("training points", training_data_points)
    print("testing points", testing_data_points)

    '''
    training_rasters = input_dataset[0][0:training_data_points]
    # print(training_data[-1])
    training_labels = input_dataset[1][0:training_data_points]
    # print(training_labels[-1])

    testing_rasters = input_dataset[0][training_data_points:input_data_points - 1]
    # print(testing_data[-1])
    testing_labels = input_dataset[1][training_data_points:input_data_points - 1]
    # print(testing_labels[-1])
    '''


    # method of dividing by 255
    rasters = (ds[0].T / 255).T
    #training_rasters = (training_rasters.T / 255).T
    #testing_rasters = (testing_rasters.T / 255).T

    # map each label into an array of shape (26)
    labels = labels_tf(ds, 26)
    #training_labels = labels_tf(training_labels, 24)
    #testing_labels = labels_tf(testing_labels, 24)
    # print('training labels:', training_labels[1])

    data_reformatted = (rasters, labels)

    return data_reformatted


def reshape_rasters(raster_set):

    raster_set = (raster_set.T / 255).T

    '''
    for four_band_arrays in raster_set:
        four_band_arrays = four_band_arrays.T
    '''

    np.transpose(raster_set, (0, 2, 3, 1))

    #training_rasters_reshape = np.transpose(training_rasters, (0, 2, 3, 1))

    return raster_set


def split_tuple_from_file(input_path):

    try:

        input_dataset = obj_io.read_object(input_path)
        return input_dataset[0], input_dataset[1]

    except Exception as e:
        print(e)


def split_dictionary_from_file(input_path):

    try:
        input_dataset = obj_io.read_object(input_path)
        return input_dataset['data'], input_dataset['labels']
    except Exception as e:
        print(e, '- could not properly split data and labels from dict.')


name = 'berkshires_128'

# obtain path of dataset
script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
project_path = os.path.dirname(script_path)
data_folder_path = os.path.join(project_path, "Data")
input_data_path = os.path.join(data_folder_path, "labeled_data_" + name + ".obj")


if __name__ == '__main__':

    #input_rasters, input_labels = split_tuple_from_file(input_data_path)
    input_rasters, input_labels = split_dictionary_from_file(input_data_path)

    print(input_rasters[0].shape, "\n")
    print(input_labels[0])

    input_rasters = np.divide(input_rasters, 255, dtype="float32")

    input_rasters = np.transpose(input_rasters, (0, 2, 3, 1))

    print(input_rasters.shape)
    print(input_rasters[0])

    input_labels = labels_tf(input_labels, 26)

    print(input_labels[0])

    keras_ready_data = (input_rasters, input_labels)
    output_path = os.path.join(script_path, 'KerasData', 'labeled_data_' + name + '_keras.obj')

    obj_io.write_object(keras_ready_data, output_path)

    #input_rasters = reshape_rasters(input_rasters)

    #print("Reshaped raster shape:", input_rasters[0].shape)

    #gc.collect()




