import keras
import os
import sys
import gdal
import numpy as np
import pathlib
from DataPreparation import data_preparation_functions as dpf
from Training.keras_models import LC_256_Binary_3Conv_512D, LC_256_Categorical_3Conv_512D


def prepare_image(filepath):
    img = gdal.Open(filepath)
    img_array = img.ReadAsArray()
    new_array = img_array / 255
    new_array = np.transpose(new_array, [1, 2, 0])
    return new_array


script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
#ext_data_path = os.path.dirname(script_path)
ext_data_path = dpf.get_external_data_directory()

img_path = 'C:/Users/james/PycharmProjects/LandCover_ML/DataPreparation/' \
           'RawData/Amherst/sat_tiles_256/7936_4608_sat_tile.tif'


img_folder = os.path.join(ext_data_path, 'ExampleImages', '256', 'Sat')

path = pathlib.Path(img_folder)

binary_dict = {0: 'Undeveloped',
               1: 'Developed'}

cat_dict = {0: 'Background',
            1: 'Unclassified',
            2: 'Developed, High Intensity',
            3: 'Developed, Medium Intensity',
            4: 'Developed, Low Intensity',
            5: 'Developed, Open Space',
            6: 'Cultivated Crops',
            7: 'Pasture/Hay',
            8: 'Grassland/Herbaceous',
            9: 'Deciduous Forest',
            10: 'Evergreen Forest',
            11: 'Mixed Forest',
            12: 'Scrub/Shrub',
            13: 'Palustrine Forested Wetland',
            14: 'Palustrine Scrub/Shrub Wetland',
            15: 'Palustrine Emergent Wetland (Persistent)',
            16: 'Estuarine Forested Wetland',
            17: 'Estuarine Scrub/Shrub Wetland',
            18: 'Estuarine Emergent Wetland',
            19: 'Unconsolidated Shore',
            20: 'Barren Land',
            21: 'Open Water',
            22: 'Palustrine Aquatic Bed',
            23: 'Estuarine Aquatic Bed',
            24: 'Tundra',
            25: 'Perennial Ice/Snow'}

training_dir = os.path.join(ext_data_path, 'Training', 'Weights')

binary_model_name = 'LC_256_Binary_3conv_512D_100e_a.h5'
binary_weight_path = os.path.join(training_dir, binary_model_name)

cat_model_name = 'LC_256_Categorical_3conv_512D_100e_a.h5'
cat_weight_path = os.path.join(training_dir, cat_model_name)

binary_model = LC_256_Binary_3Conv_512D(weights_path=binary_weight_path)
categorical_model = LC_256_Categorical_3Conv_512D(weights_path=cat_weight_path)


example_image_dict = {}
example_label_dict = {}

for entry in path.iterdir():

    entry_basename = os.path.basename(str(entry))
    filename, ext = os.path.splitext(entry_basename)

    img_array = prepare_image(str(entry))
    example_image_dict[entry_basename] = img_array


for filename, img_array in example_image_dict.items():

    model_input_array = img_array.reshape((1, 256, 256, 4))

    categorical_prediction = categorical_model.predict(model_input_array)
    binary_prediction = binary_model.predict(model_input_array)

    print('\nInput File: ' + filename)
    print('Input Shape:', model_input_array.shape)

    max_class_binary = np.argmax(binary_prediction[0])
    print('Development Status:', binary_dict[max_class_binary])

    max_class = np.argmax(categorical_prediction[0])
    print('Land Cover Category:', cat_dict[max_class])



