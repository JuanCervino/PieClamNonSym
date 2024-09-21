import json
import yaml
import os
from copy import deepcopy

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
            if self.global_config_base:
                hypers_path = os.path.join('..', 'hypers', 'hypers_link_prediction' + '.yaml')
                with open(hypers_path, 'r') as hypers_file:
                    params_dict = yaml.safe_load(hypers_file)
                configs_dict = deepcopy(params_dict['MightyConfigs'+'_' + model_name])
                first_entry = {'base_config': configs_dict}
                json.dump(first_entry, file, indent=4)
    
    def update_file(self, acc, config_triplets):
        with open(self.save_path, 'r') as file:
            loaded_acc_configs = json.load(file)

        loaded_acc_configs.update({str(acc): config_triplets})

        with open(self.save_path, 'w') as file:
            json.dump(loaded_acc_configs, file, indent=4)
