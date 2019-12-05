import os
import gdal
import sys
from DataPreparation import data_preparation_functions as dpf
import numpy as np


def split(raster_name, tile_size, output_dir, raster_type='sat'):
    """
    Splits raster image into smaller rasters with tile_size x tile_size
    dimensions. Saves images into a specified output directory.
    :param raster_name: The name of the input raster
    :param tile_size: The dimensions of the new raster images
    :param output_dir: The output directory for new raster images
    :param raster_type: string used to label new raster filenames
    """
    raster = gdal.Open(raster_name)  # open raster image

    width = raster.RasterXSize
    height = raster.RasterYSize

    # iterate through x pixels with tile-size steps
    for i in range(0, width, tile_size):

        # iterate through y pixels with tilesize steps
        for j in range(0, height, tile_size):
            # i = x_origin, j = y_origin for each tile

            # gets width and height, ensuring it is w/in bounds of parent w/h
            w = min(i + tile_size, width) - i
            h = min(j + tile_size, height) - j

            # get path to new output file
            filename = str(i) + "_" + str(j) + "_" + raster_type + "_tile" + ".tif"
            filepath = os.path.join(output_dir, filename)

            # execute GDAL command
            gdal_trans_string = "gdal_translate -of GTIFF -srcwin " + str(i) + \
                                ", " + str(j) + ", " + str(w) + ", " + str(h) +\
                                " " + raster_name + " " + filepath
            os.system(gdal_trans_string)


