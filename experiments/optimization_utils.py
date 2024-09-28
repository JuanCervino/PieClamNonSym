
import torch
import pandas as pd
import json
import yaml
import os
from copy import deepcopy
import itertools

import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')

from utils.printing_utils import printd
from utils import utils_pyg as up
from datasets.import_dataset import import_dataset
import link_prediction as lp
from trainer import Trainer
from datetime import datetime

class SaveRun:

    #todo: this class will save the file for a run. it will save a dictionary of scores based on derivatives
    # todo; there should also be a number iteration?
    '''we save the a base config (either model specific or global) and change it with deltas'''

    def __init__(self, model_name, ds_name, global_config_base, save_path, config_ranges=None):
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



        first_entry = {'date_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        if config_ranges is not None:
            first_entry['config_ranges'] = config_ranges

        # Load your configuration
        curr_file_dir = os.path.dirname(os.path.abspath(__file__))
        hypers_path = os.path.join(curr_file_dir, '..', 'hypers', 'hypers_link_prediction' + '.yaml')
        with open(hypers_path, 'r') as hypers_file:
            params_dict = yaml.safe_load(hypers_file)
        if self.global_config_base:
            configs_dict = deepcopy(params_dict['MightyConfigs' + '_' + model_name])
        else:
            configs_dict = deepcopy(params_dict[ds_name + '_' + model_name])

        # Add the base config to the dictionary
        first_entry['base_config'] = configs_dict

        # Write everything to the file at once
        with open(self.save_path, 'w') as file:
            json.dump(first_entry, file, indent=4)





        # with open(self.save_path, 'w') as file:
        #     first_entry = {'date_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        #     json.dump(first_entry, file, indent=4)

        # # the first lines in the file is the base config to which we make deltas
        # with open(self.save_path, 'w') as file:
        #     hypers_path = os.path.join('..', 'hypers', 'hypers_link_prediction' + '.yaml')
        #     with open(hypers_path, 'r') as hypers_file:
        #         params_dict = yaml.safe_load(hypers_file)
        #     if self.global_config_base:
        #         configs_dict = deepcopy(params_dict['MightyConfigs'+'_' + model_name])
        #     else:
        #         configs_dict = deepcopy(params_dict[ds_name + '_' + model_name])
            
        #     #todo: also insert the date and time somehow into the file.
        #     second_entry = {'base_config': configs_dict}
        #     json.dump(second_entry, file, indent=4)
    
    def update_file(self, acc, config_triplets):
        with open(self.save_path, 'r') as file:
            loaded_acc_configs = json.load(file)

        loaded_acc_configs.update({str(acc): config_triplets})

        with open(self.save_path, 'w') as file:
            json.dump(loaded_acc_configs, file, indent=4)


    @staticmethod
    def load_saved(file_path, export_path=None, print_base=False, print_config_ranges=False, print_date_time=False, sort_by='val_acc'):
        # Load JSON data
        with open(file_path) as f:
            data = json.load(f)

        # Separate base config and runs
        if "date_time" in data.keys():
            date_time = data.pop("date_time")
        if "config_ranges" in data.keys():
            config_ranges = data.pop("config_ranges")
        base_config = data.pop("base_config")
        if print_base:
            print(base_config)
        if print_config_ranges:
            print(config_ranges)
        if print_date_time:
            print(date_time)

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

        # Group by unique parameter configurations and calculate mean, std, and count
        if 'acc' in df.columns:
            grouped = df.groupby([col for col in df.columns if col != 'acc']).agg(
                avg_acc=('acc', 'mean'),
                std_acc=('acc', 'std'),
                count=('acc', 'size')  # Add count of occurrences
            ).reset_index()
            
            # Rearrange columns so mean, std, and count are first
            cols = ['avg_acc', 'std_acc', 'count'] + [col for col in grouped.columns if col not in ['avg_acc', 'std_acc', 'count']]
            grouped = grouped[cols]

            # Sort by avg_acc
            grouped = grouped.sort_values(by='avg_acc', ascending=False).reset_index(drop=True)

        elif 'val_acc' in df.columns:
            grouped = df.groupby([col for col in df.columns if col not in ['test_acc', 'val_acc']]).agg(
                avg_test_acc=('test_acc', 'mean'),
                std_test_acc=('test_acc', 'std'),
                avg_val_acc=('val_acc', 'mean'),
                std_val_acc=('val_acc', 'std'),
                count=('val_acc', 'size')  # Add count of occurrences
            ).reset_index()

            # Rearrange columns so mean, std, and count are first
            cols = ['avg_test_acc', 'std_test_acc', 'avg_val_acc', 'std_val_acc', 'count'] + [col for col in grouped.columns if col not in ['avg_test_acc', 'std_test_acc', 'avg_val_acc', 'std_val_acc', 'count']]
            grouped = grouped[cols]

            # Sort by specified column
            if sort_by == 'val_acc':
                grouped = grouped.sort_values(by='avg_val_acc', ascending=False).reset_index(drop=True)
            elif sort_by == 'test_acc':
                grouped = grouped.sort_values(by='avg_test_acc', ascending=False).reset_index(drop=True)

        return grouped

   

def cross_val_link(
        ds_name, 
        model_name,
        range_triplets,
        n_reps,
        global_config_base,
        densify,
        attr_opt,
        test_p,
        val_p,
        device,
        plot_every=10000):
    
    ds = None
    ds_test_omitted = None
    ds_test_val_omitted = None
    
    # ============ OMIT TEST =============

    try:
        
        curr_file_dir = os.path.dirname(os.path.abspath(__file__))
        
        save_path = os.path.join(curr_file_dir, 'results', 'link_prediction', ds_name, model_name, 'acc_configs.json')
        run_saver = SaveRun(model_name, ds_name, global_config_base, save_path, config_ranges=range_triplets)
        #!   
        ds = import_dataset(ds_name)
        # OMIT TEST
        ds_test_omitted = ds.clone()
        ds_test_omitted.omitted_dyads_test, ds_test_omitted.edge_index, ds_test_omitted.edge_attr = lp.get_dyads_to_omit(
                                                                                                            ds.edge_index, 
                                                                                                            ds.edge_attr, 
                                                                                                            test_p)
        #!
        for values in itertools.product(*[triplet[2] for triplet in range_triplets]):
            for _ in range(n_reps): 
        

                ds_test_val_omitted = ds_test_omitted.clone()
                
                # OMIT VALIDATION DYADS
                ds_test_val_omitted.omitted_dyads_val, ds_test_val_omitted.edge_index, ds_test_val_omitted.edge_attr = lp.get_dyads_to_omit(
                                                                                        ds_test_omitted.edge_index, 
                                                                                        ds_test_omitted.edge_attr, 
                                                                                        ((val_p)/(1-test_p)))
                
                # ============ OMIT VALIDATION =============

                if densify:
                    ds_test_val_omitted.edge_index, ds_test_val_omitted.edge_attr = up.my_two_hop(ds_test_val_omitted)
                                            
                outers = []
                inners = []
                for i in range(len(range_triplets)):
                    outers.append(range_triplets[i][0])
                    inners.append(range_triplets[i][1])
                    
                config_triplets = [
                    [outers[i], inners[i], values[i]] for i in range(len(range_triplets))
                            ]
                # if model_name in {'iegam', 'piegam'}:
                #     if 's_reg' in inners:
                #         ind_s = inners.index('s_reg')
                #         config_triplets.append([outers[ind_s], inners[ind_s], values[ind_s]])





                trainer = Trainer(
                            dataset=ds_test_val_omitted,
                            model_name=model_name,
                            task='link_prediction',
                            config_triplets_to_change=config_triplets,
                            mighty_configs_dict=global_config_base,
                            attr_opt=False,
                            device=device,
                )

                losses, acc_test, acc_val = trainer.train(
                            init_type='small_gaus',
                            init_feats=True,
                            acc_every=20,
                            plot_every=plot_every,
                            verbose=False,
                            verbose_in_funcs=False
                        )
                #todo: i need a way to print nothing. verbose should be a number between 0 and 2. verbose 2 is the verbose we have now. verbose 1 is the not verbose we have now. verbose 0 is print nothing nothing.
                #todo: change iegam to ieclam everywhere.
                
                last_acc_test = acc_test['auc'][-1]
                if acc_val['auc']:
                    last_acc_val = acc_val['auc'][-1]
                    run_saver.update_file((last_acc_test, last_acc_val), config_triplets)
                else:
                    last_acc_val = None
                    run_saver.update_file(last_acc_test, config_triplets)


                del ds_test_val_omitted
                ds_test_val_omitted = None
                torch.cuda.empty_cache()
    except Exception as e:
        raise e
    finally:
        if ds is not None:
            del ds
        if ds_test_omitted is not None:
            del ds_test_omitted
        if ds_test_val_omitted is not None:
            del ds_test_val_omitted
        torch.cuda.empty_cache()
        printd('\n\nFinished CrossVal!\n\n')    