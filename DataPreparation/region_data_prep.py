import DataPreparation.match_raster as mr
import DataPreparation.split_raster as sr
from DataPreparation import object_io
from DataPreparation import data_preparation_functions as dpf
import os
import sys
import pathlib
import numpy as np

ext_data_path = dpf.get_external_data_directory()


def get_raw_filenames(directory, tile_size):

    raw_file_dict = {}

    try:
        path = pathlib.Path(directory)

        for entry in path.iterdir():
            if entry.is_file():
                if entry.name.startswith('lc'):
                    raw_file_dict['lc'] = entry.name
                elif entry.name.startswith('sat'):
                    raw_file_dict['sat'] = entry.name
            if entry.is_dir():
                if entry.name.startswith('sat_tiles_' + str(tile_size)):
                    raw_file_dict['sat_output_dir'] = entry.name
                if entry.name.startswith('lc_tiles_' + str(tile_size)):
                    raw_file_dict['lc_output_dir'] = entry.name

        return raw_file_dict

    except Exception as e:
        print(e)


def generate_tiles(region_name, region_directory, raw_sat, sat_output, tile_size):

    # split raw sat photos into smaller tiles
    sr.split(raw_sat, tile_size, sat_output)

    # find & split land-cover tiles corresponding to sat tiles
    mr.create_lc_tiles(region_name, region_directory, sat_output, tile_size)


def generate_dataset(region_name, lc_output, sat_output, tile_size):

    dataset_dict = mr.create_dataset_dict(region_name, ext_data_path, lc_output, sat_output, tile_size)

    project_folder = ext_data_path
    training_folder = os.path.join(project_folder, 'Data', str(tile_size), 'Training')
    validation_folder = os.path.join(project_folder, 'Data', str(tile_size), 'Validation')

    # training data and label folders
    train_data_output_folder = os.path.join(training_folder, 'Data_NPY')
    train_label_output_path = os.path.join(training_folder, 'Labels', region_name + '_train.dict')

    # validation output folders
    val_data_output_folder = os.path.join(validation_folder, 'Data_NPY')
    val_label_output_path = os.path.join(validation_folder, 'Labels', region_name + '_val.dict')

    # used to split dataset into training and validation sets
    dataset_length = len(dataset_dict['data'])
    counter = 0
    val_split = 0.2  # fraction of dataset to be used for validation

    # unfilled training and validation label dictionaries
    training_labels = {}
    validation_labels = {}

    for idx, raster_data in dataset_dict['data'].items():

        if counter <= (dataset_length * (1 - val_split)):
            npy_file_path = os.path.join(train_data_output_folder, idx + '.npy')
            np.save(npy_file_path, raster_data)

            # ensures that the matching label is in the training set
            training_labels[idx] = dataset_dict['labels'][idx]

        if counter > (dataset_length * (1 - val_split)):
            npy_file_path = os.path.join(val_data_output_folder, idx + '.npy')
            np.save(npy_file_path, raster_data)

            # ensures that the matching label is in the validation set
            validation_labels[idx] = dataset_dict['labels'][idx]

        counter += 1  # iterate counter

    object_io.write_object(training_labels, train_label_output_path)
    object_io.write_object(validation_labels, val_label_output_path)


def combine_region_dicts(directory):
    """
    Combine region-based label dictionaries within a directory
    :param directory: training or validation directory
    """
    try:
        path = pathlib.Path(directory)

        combined_dict = {}

        for entry in path.iterdir():
            filename, ext = os.path.splitext(str(entry))

            if entry.is_file() and ext == '.dict':

                entry_dict = object_io.read_object(entry)

                combined_dict.update(entry_dict)

        combined_output_path = os.path.join(directory, 'combined_labels.dict')
        object_io.write_object(combined_dict, combined_output_path)

    except Exception as e:
        print(e)


def create_binary_labels(region_list, tile_size):

    temp_list = []
    train_output_dir = os.path.join(ext_data_path, 'Data', str(tile_size), 'Training', 'Labels')
    val_output_dir = os.path.join(ext_data_path, 'Data', str(tile_size), 'Validation', 'Labels')

    training_binary_dict = {}
    validation_binary_dict = {}

    for region_name in region_list:
        lc_dir_path = os.path.join(ext_data_path, 'DataPreparation', 'RawData', region_name, 'lc_tiles_' + str(tile_size))

        train_region_label_dict_path = os.path.join(ext_data_path, 'Data', str(tile_size), 'Training',
                                                    'Labels', region_name + '_train.dict')
        val_region_label_dict_path = os.path.join(ext_data_path, 'Data', str(tile_size), 'Validation',
                                                  'Labels', region_name + '_val.dict')

        train_region_label_dict = object_io.read_object(train_region_label_dict_path)
        val_region_label_dict = object_io.read_object(val_region_label_dict_path)

        for item in train_region_label_dict.keys():

            file_name = str(item).replace(region_name + '_', '')
            file_name = file_name + '_lc_tile.tif'
            file_path = os.path.join(lc_dir_path, file_name)

            try:
                binary_value = mr.find_binary_class(file_path)
                training_binary_dict[item] = binary_value
            except FileNotFoundError as e:
                print(e)

        for item in val_region_label_dict.keys():
            file_name = str(item).replace(region_name + '_', '')
            file_name = file_name + '_lc_tile.tif'
            file_path = os.path.join(lc_dir_path, file_name)

            try:
                binary_value = mr.find_binary_class(file_path)
                validation_binary_dict[item] = binary_value
            except FileNotFoundError as e:
                print(e)

    training_binary_dict_path = os.path.join(train_output_dir, 'combined_binary_labels.dict')
    validation_binary_dict_path = os.path.join(val_output_dir, 'combined_binary_labels.dict')

    object_io.write_object(training_binary_dict, training_binary_dict_path)
    object_io.write_object(validation_binary_dict, validation_binary_dict_path)


if __name__ == '__main__':

    tilesize = 256

    regions = ['amherst', 'berkshires', 'boston', 'brockton', 'hyannis']

    for region in regions:

        region_lower = region.lower()

        region_dir = os.path.join(ext_data_path, "DataPreparation", "RawData", region.capitalize())

        raw_filenames = get_raw_filenames(region_dir, tilesize)

        sat_output_path = os.path.join(region_dir, raw_filenames['sat_output_dir'])
        lc_output_path = os.path.join(region_dir, raw_filenames['lc_output_dir'])
        raw_lc_path = os.path.join(region_dir, raw_filenames['lc'])
        raw_sat_path = os.path.join(region_dir, raw_filenames['sat'])

        generate_tiles(region_lower, region_dir, raw_sat_path, sat_output_path, tilesize)

        generate_dataset(region, lc_output_path, sat_output_path, tilesize)

    # dictionaries to be combined
    training_label_dir = os.path.join(ext_data_path, "Data", str(tilesize), "Training", "Labels")
    validation_label_dir = os.path.join(ext_data_path, "Data", str(tilesize), "Validation", "Labels")

    combine_region_dicts(training_label_dir)
    combine_region_dicts(validation_label_dir)

    # create binary labels for each of the regions in the following list
    create_binary_labels(regions, tilesize)





