import pandas as pd
import os
import sys
import keras
import pathlib
from DataPreparation import data_preparation_functions as dpf
from DataPreparation import object_io


ext_data_directory = dpf.get_external_data_directory()


train_dict_dir = os.path.join(ext_data_directory, 'Training', 'TrainHistory', 'Dicts')

train_dict_path = pathlib.Path(train_dict_dir)

history_df_dict = {}


for entry in pathlib.Path.iterdir(train_dict_path):

    # converts all training history dicts to csv
    if str(entry).endswith('_a.dict'):
        print('Reading ' + entry.name + ' into DataFrame')

        entry_dict = object_io.read_object(entry)

        # index range is the number of epochs
        entry_df = pd.DataFrame(entry_dict.history, index=range(1, 101))
        entry_df.index.name = 'Epochs'

        history_df_dict[entry.name] = entry_df


history_df = pd.concat(history_df_dict.values(),
                       keys=[name for name in history_df_dict.keys()],
                       axis=1)

print(history_df)

combined_history_csv_name = 'combined_history.csv'
combined_history_csv_path = os.path.join(ext_data_directory,
                                         'Training',
                                         'TrainHistory',
                                         'CSV',
                                         'combined_history.csv')


with open(combined_history_csv_path, mode='w') as f:
    history_df.to_csv(f)
