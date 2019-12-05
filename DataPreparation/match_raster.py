import gdal
from DataPreparation import coordinates as ct
import os
import sys
import pathlib
import numpy as np
from scipy import stats
import DataPreparation.object_io as obj_io
import collections


def get_matching_raster(small_raster_path, large_raster_path, output_name):
    """
    Cuts a small raster from a larger raster that matches the geographic
    coordinates of a smaller raster.
    :param small_raster_path: file path to the smaller raster
    :param large_raster_path: file path to the larger raster
    :param output_name: output file name
    """
    driver = gdal.GetDriverByName('GTIFF')

    # get small and large raster filenames
    large_raster_filename = small_raster_path
    small_raster_filename = large_raster_path

    # open small and large rasters with gdal
    small_raster = gdal.Open(large_raster_filename)
    large_raster = gdal.Open(small_raster_filename)

    # get No. of columns and rows of small raster
    cols = small_raster.RasterXSize
    rows = small_raster.RasterYSize

    # get corners of the small raster
    corner_coords = ct.get_corners(small_raster)

    corner_pixels = []  # list to store corner pixel values

    # finds large pixel values for each corner coordinate
    for corner in corner_coords:
        corner_pixels.append(ct.get_pixel_value(corner, large_raster))

    # cut tile using corner_pixels
    # 0=top_left, 1=top_right, 2=bot_left, 3=bot_right
    x_origin = corner_pixels[0][0]
    y_origin = corner_pixels[0][1]
    w = corner_pixels[1][0] - x_origin
    h = corner_pixels[2][1] - y_origin

    # execute GDAL command to create tiles
    gdal_trans_string = "gdal_translate -of GTIFF -srcwin " + str(x_origin) + \
                        ", " + str(y_origin) + ", " + str(w) + ", " + str(h) + \
                        " " + small_raster_filename + " " + output_name
    os.system(gdal_trans_string)


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def find_predominant_class(lc_tile_name):
    """
    Obtains mode pixel value from a raster file
    :param lc_tile_name: filepath of raster
    :return: integer corresponding to mode pixel value
    """
    lc_tile_raster = gdal.Open(lc_tile_name)
    lc_tile_array = lc_tile_raster.ReadAsArray()

    # returns mode land-cover pixel value (corresponds to class)
    return stats.mode(lc_tile_array, axis=None)[0][0]


def find_binary_class(lc_tile_name):
    """
    Returns 1 if there is developed land, 0 if none
    :param lc_tile_name: Name of lc_tile raster
    :return: 1 or 0
    """
    lc_tile_raster = gdal.Open(lc_tile_name)
    lc_tile_array = lc_tile_raster.ReadAsArray()

    a = [2, 3, 4, 5]

    if np.isin(a, lc_tile_array).any():
        return 1
    else:
        return 0


def get_lc_tile_list(folder_path):

    lc_tile_list = []

    try:
        path = pathlib.Path(folder_path)

        file_counter = 0  # counter for console output

        for entry in path.iterdir():

            ext = os.path.splitext(entry)[-1].lower()
            file_string = os.path.splitext(entry)[0]

            file_counter += 1

            #print("file string:", file_string)
            #print("ext:", ext, "\n")

            if ext == ".tif" and file_string.endswith("lc_tile"):
                print("Matching... :", entry.name)

                # this is a roundabout way
                #pred_class = find_predominant_class(file_string + ext)
                pred_class = find_predominant_class_80(file_string + ext)

                lc_tile_list.append((entry.name, pred_class))

        return lc_tile_list

    except Exception as e:
        print(e)


def get_sat_tile_list(folder_path):

    sat_tile_list = []

    try:
        path = pathlib.Path(folder_path)

        print('Obtaining sat tiles from', path)

        for entry in path.iterdir():

            ext = os.path.splitext(entry)[-1].lower()
            file_string = os.path.splitext(entry)[0]

            if ext == ".tif" and file_string.endswith("sat_tile"):

                print(entry.name)

                sat_tile_list.append(entry.name)

        return sat_tile_list

    except Exception as e:
        print(e)


def get_label_dict(folder_path):

    lc_tile_dict = {}

    try:
        path = pathlib.Path(folder_path)

        for entry in path.iterdir():

            file_string, ext = os.path.splitext(entry)

            if ext == ".tif" and file_string.endswith("lc_tile"):

                #diagnostic
                print("Finding predicted class...")

                # this is a roundabout way
                pred_class = find_predominant_class_80(file_string + ext)

                print(pred_class)

                # only add if it is not -1
                if pred_class >= 0:

                    lc_tile_dict[entry.name] = pred_class

        return lc_tile_dict

    except Exception as e:
        print(e)


def get_labels(folder_path):
    """
    Obtains predominant (mode) pixel values from each .tif file in a
    directory of .tif files.
    :param folder_path: Directory containing land-cover .tif files
    :return label_list: a list of labels
    """

    label_dict = {}

    try:
        path = pathlib.Path(folder_path)

        print('\nObtaining label data:')
        counter = 0

        for entry in path.iterdir():

            file_base = os.path.basename(entry)
            file_string, ext = os.path.splitext(file_base)

            if ext == ".tif" and file_string.endswith("lc_tile"):
                counter += 1

                pred_class = find_predominant_class(entry.as_posix())
                #label_dict[entry.name] = pred_class
                label_dict[file_string] = pred_class

                if counter % 1000 is 0:
                    print('...parsed through', counter, 'land-cover files.')

        return label_dict

    except Exception as e:
        print(e)


def get_sat_data(folder_path, tile_size):

    #sat_data_list = []
    #label_list = []

    sat_data_dict = {}

    try:
        path = pathlib.Path(folder_path)

        print('\nObtaining sat data:')
        counter = 0

        for entry in path.iterdir():

            file_base = os.path.basename(entry)
            file_string, ext = os.path.splitext(file_base)

            if ext.endswith(".tif"):
                counter += 1

                sat_tile_raster = gdal.Open(str(entry))
                sat_tile_array = sat_tile_raster.ReadAsArray()

                # only grab data from 128x128 .tif files
                if sat_tile_array.shape[1] == tile_size and sat_tile_array.shape[2] == tile_size:

                    #sat_data_dict[entry.name] = sat_tile_array
                    sat_data_dict[file_string] = sat_tile_array

                if counter % 1000 is 0:
                    print('...parsed through', counter, 'sat files.')


        return sat_data_dict

    except Exception as e_1:
        print(e_1)


def create_sat_tiles(name, region_directory, lc_tile_directory):

    # list of tuples (lc_tile name, label integer)
    lc_tiles = get_lc_tile_list(lc_tile_directory)

    sat_filename = "sat_" + name + ".tif"
    sat_filepath = os.path.join(region_directory, sat_filename)

    for tile in lc_tiles:

        try:

            lc_tile_filename, label = tile  # unpack tile tuple
            file_path = os.path.join(region_directory, "lc_tiles", lc_tile_filename)

            sat_tile_filename = lc_tile_filename.replace("lc", "sat")

            output_path = os.path.join(region_directory, "sat_tiles", sat_tile_filename)

            get_matching_raster(file_path, sat_filepath, output_path)

        except Exception as e:
            print(e)


def create_lc_tiles(name, region_directory, sat_tile_directory, tilesize):

    # list of sat tiles
    sat_tiles = get_sat_tile_list(sat_tile_directory)

    raw_lc_filename = "lc_" + name + ".tif"
    raw_lc_filepath = os.path.join(region_directory, raw_lc_filename)

    for tile in sat_tiles:

        try:
            tile_file_path = os.path.join(region_directory, "sat_tiles_" + str(tilesize), tile)

            lc_tile_filename = tile.replace("sat", "lc")

            output_path = os.path.join(region_directory, "lc_tiles_" + str(tilesize), lc_tile_filename)

            get_matching_raster(tile_file_path, raw_lc_filepath, output_path)
            print(tile)

        except Exception as e:
            print(e)


def create_dataset_dict(name, ext_data_dir, lc_tile_directory, sat_tile_directory, tile_size):
    """
    Creates a dictionary dataset in the format of {data: label}
    :param name: Name of dataset region
    :param ext_data_dir: external data source path
    :param lc_tile_directory: directory of land-cover tiles, used for labels
    :param sat_tile_directory: directory of sat/aerial tiles, used for data
    :param tile_size: tile dimensions
    :return dataset_dict: dictionary of 'data' and 'label' entries
    """

    label_dict = get_labels(lc_tile_directory)
    sat_dict = get_sat_data(sat_tile_directory, tile_size)

    print('\nLabel Dictionary Length:', len(label_dict))
    print('Sat Dictionary Length:', len(sat_dict))

    dataset_dict = {}

    data_list = []
    label_list = []
    id_list = []

    for lc_name in label_dict:

        sat_name = lc_name.replace("lc", "sat")
        id_name = name + "_" + lc_name.replace("_lc_tile", "")

        if sat_name in sat_dict:

            data_list.append(sat_dict[sat_name])
            label_list.append(label_dict[lc_name])
            id_list.append(id_name)  # should append the filename (w/out ext)

    if len(data_list) == len(label_list):

        data_dict = dict(zip(id_list, data_list))
        label_dict = dict(zip(id_list, label_list))

        """
        data_array = np.asarray(data_list)
        label_array = np.asarray(label_list)
        """

        dataset_dict['data'] = data_dict
        dataset_dict['labels'] = label_dict

        print('\nData and label list lengths match. Values loaded.')

        #project_folder = os.path.dirname(ext_data_dir)
        data_output_folder = os.path.join(ext_data_dir, 'Data')
        new_obj_path = os.path.join(data_output_folder,
                                    'labeled_data_' + name + '.obj')

        return dataset_dict
        #obj_io.write_object(dataset_dict, new_obj_path)

    else:
        print('Data and Label lists did not match. Values were not saved.')

