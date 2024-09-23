import pandas as pd
import json
import yaml
import os
from copy import deepcopy
from utils.printing_utils import printd

class SaveRun:

    #todo: this class will save the file for a run. it will save a dictionary of scores based on derivatives
    # todo; there should also be a number iteration?
    '''we save the a base config (either model specific or global) and change it with deltas'''

    def __init__(self, model_name, ds_name, global_config_base, save_path):
        self.model_name = model_name
        self.ds_name = ds_name
        self.global_config_base = global_config_base
        self.save_path = save_path
        # make a new file

        
        
        if os.path.exists(self.save_path):
            # Find the next available file name
            i = 0
            while os.path.exists(f"{self.save_path[:-5]}_{i}.json"):
                i += 1
            self.save_path = f"{self.save_path[:-5]}_{i}.json"

        # the first lines in the file is the base config to which we make deltas
        with open(self.save_path, 'w') as file:
            hypers_path = os.path.join('..', 'hypers', 'hypers_link_prediction' + '.yaml')
            with open(hypers_path, 'r') as hypers_file:
                params_dict = yaml.safe_load(hypers_file)
            if self.global_config_base:
                configs_dict = deepcopy(params_dict['MightyConfigs'+'_' + model_name])
            else:
                configs_dict = deepcopy(params_dict[ds_name + '_' + model_name])
            first_entry = {'base_config': configs_dict}
            json.dump(first_entry, file, indent=4)
    
    def update_file(self, acc, config_triplets):
        with open(self.save_path, 'r') as file:
            loaded_acc_configs = json.load(file)

        loaded_acc_configs.update({str(acc): config_triplets})

        with open(self.save_path, 'w') as file:
            json.dump(loaded_acc_configs, file, indent=4)


    @staticmethod
    def load_saved(file_path, export_path=None, print_base=False):
        
        # Load JSON data
        with open(file_path) as f:
            data = json.load(f)

        # Separate base config and runs
        base_config = data.pop("base_config")
        if print_base:
            printd(base_config)

        # List to hold processed data for DataFrame
        processed_data = []

        # Iterate through each run acc
        for acc_str, changes in data.items():
            acc = eval(acc_str)
            row = {}
            if type(acc) == float:
                row['acc'] = acc  # Convert acc key to a float
            elif type(acc) == tuple:
                row['test_acc'] = acc[0]
                row['val_acc'] = acc[1]

            for change in changes:
                section, key, value = change
                row[f"{section}_{key}"] = value
            processed_data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        df.reset_index(drop=True, inplace=True)

        # Fill missing columns (parameters that were not changed in some runs) with base config values
        # for section, params in base_config.items():
        #     for key, value in params.items():
        #         column_name = f"{section}_{key}"
        #         if column_name not in df.columns:
        #             df[column_name] = value

        # Group by unique parameter configurations and calculate mean and std of accs
        if 'acc' in df.columns:
            grouped = df.groupby([col for col in df.columns if col != 'acc']).agg(
                avg_acc=('acc', 'mean'),
                std_acc=('acc', 'std')
            ).reset_index()
            grouped = grouped.sort_values(by='avg_acc', ascending=False).reset_index(drop=True)
        
        elif 'val_acc' in df.columns:
            grouped = df.groupby([col for col in df.columns if col not in ['test_acc', 'val_acc']]).agg(
                avg_test_acc=('test_acc', 'mean'),
                std_test_acc=('test_acc', 'std'),
                avg_val_acc=('val_acc', 'mean'),
                std_val_acc=('val_acc', 'std')
            ).reset_index()
            grouped = grouped.sort_values(by='avg_val_acc', ascending=False).reset_index(drop=True)
       

        # Save the grouped data to a CSV for further analysis
        if export_path:
            grouped.to_csv('grouped_accs.csv', index=False)

        return grouped
