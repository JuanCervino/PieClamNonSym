
import torch
import pandas as pd
import json
import yaml
import os
from copy import deepcopy
import itertools
from torch_geometric.transforms import TwoHop

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.join(current_dir, '..') not in sys.path:
    sys.path.insert(0, os.path.join(current_dir, '..'))
# if '..' not in sys.path:
#     sys.path.insert(0, '..')

from utils.printing_utils import printd
from utils import utils_pyg as up
from datasets.import_dataset import import_dataset
import utils.link_prediction as lp
from trainer import Trainer
from datetime import datetime

def load_hyper_config(task, model_name, use_global_config_base=True, ds_name=None):
    '''load the hyperparameters for the model and the dataset'''
    curr_file_dir = os.path.dirname(os.path.abspath(__file__))
    hypers_path = os.path.join(curr_file_dir, '..', 'hypers', 'hypers_'+ task + '.yaml')
    with open(hypers_path, 'r') as hypers_file:
        params_dict = yaml.safe_load(hypers_file)
    if use_global_config_base:
        configs_dict = deepcopy(params_dict['GlobalConfigs' + '_' + model_name])
    else:
        assert ds_name is not None
        configs_dict = deepcopy(params_dict[ds_name + '_' + model_name])
    return configs_dict


def perturb_config(task, model_name, deltas, use_global_config, ds_name=None):
    '''perturb the config with the deltas'''
    config_dict = load_hyper_config(task, model_name, use_global_config, ds_name)
    config_range = {}
    if model_name in ['bigclam', 'ieclam']:
        deltas = {key: deltas[key] for key in ['clamiter_init', 'feat_opt'] if key in deltas}
    for outer_key in deltas:
        for inner_key in deltas[outer_key]:
            value = deltas[outer_key][inner_key]
            if outer_key not in config_range:
                config_range[outer_key] = {}
            config_range[outer_key][inner_key] = [max(config_dict[outer_key][inner_key] - value, 0), config_dict[outer_key][inner_key] + value]

    config_range_list = []
    for outer_key in config_range:
        for inner_key in config_range[outer_key]:
            config_range_list.append([outer_key, inner_key, config_range[outer_key][inner_key]])
    return config_range_list
    
    

class SaveRun:

    '''we save the a base config (either model specific or global) and change it with deltas. each experiment result is the config delta and the result of the experiment in a json file. to gather all of the results together there is an analysis.py in every results folder.'''
    #todo: modify to fit the anomaly setting. 
    #todo: several ds and have them as part of the table
    #todo: for link prediction i want to know many results for each dataset. for anomaly detection i want to know many results for one configuration
    def __init__(self, model_name, ds_name, task, use_global_config_base, save_path, config_ranges=None):
        self.model_name = model_name
        self.task = task
        self.ds_name = ds_name
        self.use_global_config_base = use_global_config_base
        self.save_path = save_path
        # make a new file

        
         
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

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
        hypers_path = os.path.join(curr_file_dir, '..', 'hypers', 'hypers_'+ task + '.yaml')
        with open(hypers_path, 'r') as hypers_file:
            params_dict = yaml.safe_load(hypers_file)
        if self.use_global_config_base:
            configs_dict = deepcopy(params_dict['GlobalConfigs' + '_' + model_name])
        else:
            configs_dict = deepcopy(params_dict[ds_name + '_' + model_name])

        # Add the base config to the dictionary
        first_entry['base_config'] = configs_dict

        # Write everything to the file at once
        with open(self.save_path, 'w') as file:
            json.dump(first_entry, file, indent=4)



    
    def update_file(self, acc, config_triplets):
        with open(self.save_path, 'r') as file:
            loaded_acc_configs = json.load(file)

        loaded_acc_configs.update({str(acc): config_triplets})

        with open(self.save_path, 'w') as file:
            json.dump(loaded_acc_configs, file, indent=4)


    @staticmethod
    def load_saved_old(task, file_path, print_base=False, print_config_ranges=False, print_date_time=False, sort_by='val_acc'):

        '''results are saved as config - acc. the function loads the results as a pandas dataframe.
        '''

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
                    row['acc'] = acc
                
            elif type(acc) == tuple:
                if task == 'link_prediction':
                    row['test_acc'] = acc[0]
                    row['val_acc'] = acc[1]
                
                elif task == 'anomaly_unsupervised':
                    row['vanilla_star'] = acc[0]
                    if len(acc) > 1:
                        row['prior'] = acc[1]
                        row['prior_star'] = acc[2]

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


    @staticmethod
    def load_saved(task, file_path, sort_by, print_base=False, print_config_ranges=False, print_date_time=False):

        '''results are saved as config - acc. the function loads the results of a single file as a pandas dataframe. to load all files of a directory, a "print_folder" function is defined in the analysis.py file of every task.
        '''
        
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
                    row['acc'] = acc
                
            elif type(acc) == tuple:
                if task == 'link_prediction':
                    row['test_acc'] = acc[0]
                    row['val_acc'] = acc[1]
                
                elif task == 'anomaly_unsupervised':
                    row['vanilla_star'] = acc[0]
                    if len(acc) > 1:
                        row['prior'] = acc[1]
                        row['prior_star'] = acc[2]

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

        elif 'val_acc' in df.columns or 'vanilla_star' in df.columns:
            if task == 'link_prediction':
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

            elif task == 'anomaly_unsupervised':
                grouped = df.groupby([col for col in df.columns if col not in ['vanilla_star', 'prior', 'prior_star']]).agg(
                    avg_vanilla_star=('vanilla_star', 'mean'),
                    std_vanilla_star=('vanilla_star', 'std'),
                    avg_prior=('prior', 'mean'),
                    std_prior=('prior', 'std'),
                    avg_prior_star=('prior_star', 'mean'),
                    std_prior_star=('prior_star', 'std'),
                    count=('vanilla_star', 'size')  # Add count of occurrences
                ).reset_index()

                # Rearrange columns so mean, std, and count are first
                cols = ['avg_vanilla_star', 'std_vanilla_star', 'avg_prior', 'std_prior', 'avg_prior_star', 'std_prior_star', 'count'] + [col for col in grouped.columns if col not in ['avg_vanilla_star', 'std_vanilla_star', 'avg_prior', 'std_prior', 'avg_prior_star', 'std_prior_star', 'count']]
                grouped = grouped[cols]

                # Sort by specified column
                if sort_by == 'vanilla_star':
                    grouped = grouped.sort_values(by='avg_vanilla_star', ascending=False).reset_index(drop=True)
                elif sort_by == 'prior':
                    grouped = grouped.sort_values(by='avg_prior', ascending=False).reset_index(drop=True)
                elif sort_by == 'prior_star':
                    grouped = grouped.sort_values(by='avg_prior_star', ascending=False).reset_index(drop=True)

        return grouped

   
#todo: change cross val link to be cross val and have link and anomaly as options. the saving of the file should be the same, but the analysis file in the results folder
def cross_val_link(
        ds_name, 
        model_name,
        range_triplets,
        n_reps,
        use_global_config_base,
        densify,
        test_p,
        val_p,
        device,
        test_dyads_to_omit=None,
        val_dyads_to_omit=None,
        attr_opt=False,
        plot_every=10000):
    
    ds = None
    ds_test_omitted = None
    ds_test_val_omitted = None
    
    # ============ OMIT TEST =============
    '''The dyad omitting process for the algorithm is described in the paper. if a test set is provided it's used and if not the test set is taken randomly with the percentage given and 5X the number of negative samples. The same goes to the val set: if it is not given it is sampled from the dyad set for every parameter configuration.'''

    try:
        
        curr_file_dir = os.path.dirname(os.path.abspath(__file__)) 
        
        save_path = os.path.join(curr_file_dir, 'results', 'link_prediction', ds_name, model_name, 'acc_configs.json')
        run_saver = SaveRun(model_name, ds_name, 'link_prediction', use_global_config_base, save_path, config_ranges=range_triplets)
        
        ds = import_dataset(ds_name)
        
        if hasattr(ds, 'val_dyads_to_omit'):
            val_dyads_to_omit = ds.val_dyads_to_omit
        if hasattr(ds, 'test_dyads_to_omit'):
            test_dyads_to_omit = ds.test_dyads_to_omit

        # OMIT TEST
        ds_test_omitted = ds.clone()
        if test_dyads_to_omit is None: 
            ds_test_omitted.omitted_dyads_test, ds_test_omitted.edge_index, ds_test_omitted.edge_attr = lp.get_dyads_to_omit(
                                                ds.edge_index, 
                                                ds.edge_attr, 
                                                test_p)
            
        else:
            assert type(test_dyads_to_omit) == torch.tensor
            assert test_dyads_to_omit.shape[0] == 2

            ds_test_omitted.omitted_dyads_test, ds_test_omitted.edge_index, ds_test_omitted.edge_attr = lp.omit_dyads(ds_test_omitted.edge_index,
                                      ds_test_omitted.edge_attr,
                                      test_dyads_to_omit)
            
        
        if val_dyads_to_omit is not None:
            #todo: if this condition holds also dont do the sampling at every iteration
            assert type(val_dyads_to_omit) == torch.tensor
            assert val_dyads_to_omit.shape[0] == 2
            ds_test_omitted.omitted_dyads_val = val_dyads_to_omit
            ds_test_omitted.omitted_dyads_val, ds_test_omitted.edge_index, ds_test_omitted.edge_attr = lp.omit_dyads(
                            ds_test_omitted.edge_index, 
                            ds_test_omitted.edge_attr,
                            val_dyads_to_omit)

        for values in itertools.product(*[triplet[2] for triplet in range_triplets]):
            for _ in range(n_reps): 
        
                ds_test_val_omitted = ds_test_omitted.clone()
                
                # OMIT VALIDATION DYADS
                '''edge attr signifies if the edge is omitted or not. if the edge_attr is 0 then the edge is an omitted dyad.'''

                if val_dyads_to_omit is None:
                    ds_test_val_omitted.omitted_dyads_val, ds_test_val_omitted.edge_index, ds_test_val_omitted.edge_attr = lp.get_dyads_to_omit(
                                            ds_test_omitted.edge_index, 
                                            ds_test_omitted.edge_attr, 
                                            ((val_p)/(1-test_p)))# the amount to extract from the remaining edges to get the initial extraction we wanted for val (size changes after removal).

                # ============ OMIT VALIDATION =============

                if densify:
                    ds_test_val_omitted.edge_index, ds_test_val_omitted.edge_attr = up.two_hop_link(ds_test_val_omitted)
                                            
                outers = []
                inners = []
                for i in range(len(range_triplets)):
                    outers.append(range_triplets[i][0])
                    inners.append(range_triplets[i][1])
                    
                config_triplets = [
                    [outers[i], inners[i], values[i]] for i in range(len(range_triplets))
                            ]
                # if model_name in {'ieclam', 'pieclam'}:
                #     if 's_reg' in inners:
                #         ind_s = inners.index('s_reg')
                #         config_triplets.append([outers[ind_s], inners[ind_s], values[ind_s]])





                trainer = Trainer(
                            dataset=ds_test_val_omitted,
                            model_name=model_name,
                            task='link_prediction',
                            config_triplets_to_change=config_triplets,
                            use_global_config_base=use_global_config_base,
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
                
                
                if acc_test['auc']:
                    last_acc_test = acc_test['auc'][-1]
                else:
                    last_acc_test = None
                
                if acc_val['auc']:
                    last_acc_val = acc_val['auc'][-1]                    
                else:
                    last_acc_val = None
                run_saver.update_file((last_acc_test, last_acc_val), config_triplets)
                
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



def multi_ds_anomaly(
        model_name,
        range_triplets,
        n_reps,
        use_global_config_base,
        device,
        ds_names=['reddit', 'photo', 'elliptic'], 
        densifiable_ds=['reddit', 'photo'],
        attr_opt=False,
        plot_every=10000):
    
    '''here we test a single configuration for a list of datasets since the setting is unsupervised. '''
    
    ds = None
    ds_for_optimization = None
    trainer_anomaly = None

    assert model_name in ['ieclam', 'bigclam', 'pieclam', 'pclam']

    try:
        
        curr_file_dir = os.path.dirname(os.path.abspath(__file__))
        
        save_paths = [os.path.join(curr_file_dir, 'results', 'anomaly_unsupervised', model_name, ds_name, 'acc_configs.json')for ds_name in ds_names]
        # a different run saver for every dataset
        run_savers = [SaveRun(model_name, 
                              ds_name,
                              task='anomaly_unsupervised', 
                              use_global_config_base=use_global_config_base, 
                              save_path=save_paths[i], 
                              config_ranges=range_triplets) for i, ds_name in enumerate(ds_names)]
        
        for values in itertools.product(*[triplet[2] for triplet in range_triplets]):
            '''for each configuration run all of the datasets and save the results in the corresponding folder'''
            outers = []
            inners = []
            for i in range(len(range_triplets)):
                outers.append(range_triplets[i][0])
                inners.append(range_triplets[i][1])
                
            config_triplets = [
                [outers[i], inners[i], values[i]] for i in range(len(range_triplets))]
            for _ in range(n_reps): 
                for i, ds_name in enumerate(ds_names):
                    ds = import_dataset(ds_name)
                    ds_to_use = ds
                    
                    if ds_name in densifiable_ds:
                        fat_ds = TwoHop()(ds)
                        fat_ds.edge_attr = torch.ones(fat_ds.edge_index.shape[1]).bool()
                        ds_to_use = fat_ds

                    losses = []
                    acc_test = []
                    acc_val = []                  
                    
                    

                    trainer = Trainer(
                                dataset=ds_to_use,
                                model_name=model_name,
                                task='anomaly_unsupervised',
                                config_triplets_to_change=config_triplets,
                                use_global_config_base=use_global_config_base,
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
                    
                    last_vanilla_star = acc_test['vanilla_star'][-1]
                    last_prior = None
                    last_prior_star = None
                    if model_name in {'pieclam', 'pclam'}:
                        last_prior = acc_test['prior'][-1]
                        last_prior_star = acc_test['prior_star'][-1]

                    run_savers[i].update_file((last_vanilla_star, last_prior, last_prior_star), config_triplets)
                    

    except Exception as e:
        raise e
    
    finally:
        if trainer_anomaly is not None:
            del trainer_anomaly
        if ds is not None:
            del ds
        if ds_for_optimization is not None:
            del ds_for_optimization
        
        torch.cuda.empty_cache()
        printd('\n\nFinished CrossVal!\n\n')    