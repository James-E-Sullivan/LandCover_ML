import os

def get_external_data_directory():

    external_data_string = 'C:/Users/james/BU_MET/CS767/LandCover_ML_External_Data'
    external_data_directory = os.path.normpath(external_data_string)

    return external_data_directory



