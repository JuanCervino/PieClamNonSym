
import torch
import pandas as pd
import json
import yaml
import os
from copy import deepcopy
import itertools
from torch_geometric.transforms import TwoHop
from torch_geometric import utils

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.join(current_dir, '..') not in sys.path:
    sys.path.insert(0, os.path.join(current_dir, '..'))
# if '..' not in sys.path:
#     sys.path.insert(0, '..')

from utils.printing_utils import printd
from utils import pyg_helpers as up
from utils import utils as my_utils
from utils.path_utils import get_project_root

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
    
    
# .dP"Y8    db    Yb    dP 888888 88""Yb 88   88 88b 88 
# `Ybo."   dPYb    Yb  dP  88__   88__dP 88   88 88Yb88 
# o.`Y8b  dP__Yb    YbdP   88""   88"Yb  Y8   8P 88 Y88 
# 8bodP' dP""""Yb    YP    888888 88  Yb `YbodP' 88  Y8 


class SaveRun:

    '''we save the a base config (either model specific or global) and change it with deltas. each experiment result is the config delta and the result of the experiment in a json file. to gather all of the results together there is an analysis.py in every results folder.'''
    
    def __init__(self, model_name, ds_name, task, metric=None, omitted_test_dyads=None, test_or_val=None, use_global_config_base=True, config_ranges=None):
        self.model_name = model_name
        self.task = task
        self.ds_name = ds_name
        self.use_global_config_base = use_global_config_base
    
        '''test as number is the binary edge_attr as a decimal number, which defines the test set so many experiments with the same test set can be saved in the same folder. edge attr is the omitted test/test and val mask'''
    
        timestamp = datetime.now().strftime('%H-%M_%d-%m-%y')
        # Find the directory where THIS python file (optimization_utils.py) is

        current_file_dir = os.path.dirname(os.path.realpath(__file__))

        # Now go to the results folder inside it
        base_results_dir = os.path.join(current_file_dir, 'results')

        # Build your final model path
        self.model_path = os.path.join(base_results_dir, task, metric, ds_name, model_name)
        os.makedirs(self.model_path, exist_ok=True)
        
        split_exists = False
        for split_name in os.listdir(self.model_path):
            if split_name.startswith('split'):
                split_omitted_test_dyads = torch.load(os.path.join(self.model_path, split_name, 'omitted_test_dyads.pt'))
                if omitted_test_dyads[0].shape == split_omitted_test_dyads[0].shape and omitted_test_dyads[1].shape == split_omitted_test_dyads[1].shape:     
                    if (omitted_test_dyads[0] == split_omitted_test_dyads[0]).all() and (omitted_test_dyads[1] == split_omitted_test_dyads[1]).all():
                        split_exists = True                    
                        break
        if not split_exists:
            split_name = f'split_{timestamp}'


        self.split_save_path = os.path.join(self.model_path, split_name)
        
        os.makedirs(self.split_save_path, exist_ok=True)
        
        omitted_test_dyads_path = os.path.join(self.split_save_path, 'omitted_test_dyads.pt')
        if not os.path.exists(omitted_test_dyads_path):
            torch.save(omitted_test_dyads, omitted_test_dyads_path)

        self.test_or_val_path = os.path.join(self.split_save_path, test_or_val)
        os.makedirs(self.test_or_val_path, exist_ok=True) 

        self.acc_configs_path = os.path.join(self.test_or_val_path, f"acc_configs{timestamp}.json")
        
        os.makedirs(os.path.dirname(self.acc_configs_path), exist_ok=True)
    

        first_entry = {'date_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'ds_name': ds_name, 'model_name': model_name, 'task': task, 'metric':metric }
        if config_ranges is not None:
            first_entry['config_ranges'] = config_ranges

        # Load your configuration. Either use global config for all datasets or use a dataset specific configuration
        curr_file_dir = os.path.dirname(os.path.abspath(__file__))
        hypers_path = os.path.join(curr_file_dir, '..', 'hypers', 'hypers_'+ task + '.yaml')
        with open(hypers_path, 'r') as hypers_file:
            params_dict = yaml.safe_load(hypers_file)
        if self.use_global_config_base:
            configs_dict = deepcopy(params_dict['GlobalConfigs' + '_' + model_name])
        else:
            configs_dict = deepcopy(params_dict[ds_name + '_' + model_name])
        self.configs_dict = configs_dict
        # Add the base config to the dictionary
        first_entry['base_config'] = configs_dict

        # Write everything to the file at once
        with open(self.acc_configs_path, 'w') as file:
            json.dump(first_entry, file, indent=4)


    
    def update_file(self, acc, config_triplets):
        with open(self.acc_configs_path, 'r') as file:
            loaded_acc_configs = json.load(file)

        loaded_acc_configs.update({str(acc): config_triplets})

        with open(self.acc_configs_path, 'w') as file:
            json.dump(loaded_acc_configs, file, indent=4)

    
    
    @staticmethod
    def load_saved(task, file_path, sort_by, metric='auc', print_base=False, print_config_ranges=False, print_date_time=False, return_base_config=True):

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
        if "ds_name" in data.keys():
            ds_name = data.pop("ds_name")
        if "model_name" in data.keys():
            model_name = data.pop("model_name")
        if "task" in data.keys():
            task = data.pop("task")
        if "metric" in data.keys():
            metric = data.pop("metric")

        base_config = data.pop("base_config")
        if data == {}:
            grouped = pd.DataFrame()

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
        if ('val_acc' in df.columns and df['val_acc'].notna().any()) or 'vanilla_star' in df.columns: # if there is validation score in the file 
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
        
        elif 'test_acc' in df.columns: # if there is only test accuracy
            if set(df.columns) == {'test_acc', 'val_acc'}:
                grouped = pd.DataFrame({'mean': df['test_acc'].mean(), 'std': df['test_acc'].std(), 'count':len(df['test_acc'])}, index=[0]) 
            else:
                grouped = df.groupby([col for col in df.columns if col not in {'test_acc', 'val_acc'}]).agg(
                    avg_acc=('test_acc', 'mean'),
                    std_acc=('test_acc', 'std'),
                    count=('test_acc', 'size')  # Add count of occurrences
                ).reset_index()
                
                # Rearrange columns so mean, std, and count are first
                cols = ['avg_acc', 'std_acc', 'count'] + [col for col in grouped.columns if col not in ['avg_acc', 'std_acc', 'count']]
                grouped = grouped[cols]

                # Sort by avg_acc
                grouped = grouped.sort_values(by='avg_acc', ascending=False).reset_index(drop=True)
        else:
            printd('no test or val accuracy scores in the file')

        if return_base_config:
            return grouped, base_config
        else:
            return grouped
       
   

def print_folder(ds_name, model_name, metric='auc', splits=None, test_or_val ='test',  num_to_plot=20, sort_by='val_acc', print_base_config=True):
    '''print the results of an experiment on a dataset with one of the Clam models. The results are arranged into a metric (auc or hits@20) and test/validation experiments.'''
    
    model_path = os.path.join(metric, ds_name, model_name)
    existing_splits = os.listdir(model_path)
    if splits is None:
        splits = existing_splits
    else:
        splits = list(set(splits).intersection(existing_splits))


    res_paths = []
    for split in splits: # all paths to test sets
        res_paths += [os.path.join(model_path, split, test_or_val)]



    for i, path in enumerate(res_paths):
        if os.path.exists(path):
            print(f'{splits[i]}')
            print(f'================== \n')
            for file_name in os.listdir(path):
                # the folder names should all be datetime %H%M_%d%m%y
                if file_name.endswith('.json'):
                    file_path = os.path.join(path, file_name)
                    grouped_tup = SaveRun.load_saved(
                        'link_prediction',
                        file_path, 
                        sort_by=sort_by,
                        return_base_config=print_base_config)
                    
                    if print_base_config:
                        grouped_df = grouped_tup[0]
                        base_config = grouped_tup[1]
                    else:
                        grouped_df = grouped_tup
                    print("    " + file_name + '\n    ==================')
                    if not grouped_df.empty:  
                        if print_base_config:
                            print('    Base config:')
                            print(json.dumps(base_config, indent=4))
                            print('    ==================') 
                        if num_to_plot == -1:
                            print(grouped_df)
                        else:
                            print(grouped_df.head(num_to_plot))
                        
                        print('\n')
                    else:
                        print(f'The file in {file_path} has no results, consider deleting.\n')
        else:
            print(f'Path {path} does not exist. Skipping.\n')



#  dP""b8 88""Yb  dP"Yb  .dP"Y8 .dP"Y8 88     88 88b 88 88  dP 
# dP   `" 88__dP dP   Yb `Ybo." `Ybo." 88     88 88Yb88 88odP  
# Yb      88"Yb  Yb   dP o.`Y8b o.`Y8b 88  .o 88 88 Y88 88"Yb  
#  YboodP 88  Yb  YbodP  8bodP' 8bodP' 88ood8 88 88  Y8 88  Yb 






def cross_val_link(
        ds_name, 
        model_name,
        range_triplets,
        n_reps,
        use_global_config_base,
        densify,
        device,
        test_p=0.0,
        val_p=0.0,
        test_dyads_to_omit=None,
        val_dyads_to_omit=None,
        test_dyads_path=None,
        val_dyads_path=None,
        test_only=False,
        metric='auc',
        attr_opt=False,
        acc_every=20,
        plot_every=10000,
        verbose=False,
        verbose_in_funcs=False):
    
    ds = None
    ds_test_omitted = None
    ds_test_val_omitted = None
    
    # ============ OMIT TEST =============
    '''The dyad omitting process for the algorithm is described in the paper. if a test set is provided it's used and if not the test set is taken randomly with the percentage given and 5X the number of negative samples. The same goes to the val set: if it is not given it is sampled from the dyad set for every parameter configuration.'''
    
    if val_p == 0:
        test_only = True

    try:
        
        curr_file_dir = os.path.dirname(os.path.abspath(__file__)) 
        test_or_val = 'test' if test_only else 'valid'
        
        # save run should configure the save paths 
        # if there is a test set folder (split. the number after split should be the number that is the test sets connected and turned into a number) like the test set we are using save n

        ds = import_dataset(ds_name, test_dyads_path=test_dyads_path, val_dyads_path=val_dyads_path)
        
        # if the dataset comes with dyads to omit use THEM
        if hasattr(ds, 'val_dyads_to_omit'):
            val_dyads_to_omit = ds.val_dyads_to_omit
        if hasattr(ds, 'test_dyads_to_omit'):
            test_dyads_to_omit = ds.test_dyads_to_omit

        # OMIT TEST
        ds_test_omitted = ds.clone()
        if test_dyads_to_omit is not None: # if the dataset comes with test dyads
            assert type(test_dyads_to_omit) == tuple
            assert utils.is_undirected(test_dyads_to_omit[0]) and utils.is_undirected(test_dyads_to_omit[1])
            
            ds_test_omitted.omitted_dyads_test, ds_test_omitted.edge_index, ds_test_omitted.edge_attr = lp.omit_dyads(ds_test_omitted.edge_index,
                                      ds_test_omitted.edge_attr,
                                      test_dyads_to_omit)
        else:

           ds_test_omitted.omitted_dyads_test, ds_test_omitted.edge_index, ds_test_omitted.edge_attr = lp.get_dyads_to_omit(
                                                ds.edge_index, 
                                                ds.edge_attr, 
                                                test_p)
             
        
        
        for triplet in range_triplets[:]:
            if triplet[2] == []:
                range_triplets.remove(triplet)
        
        run_saver = SaveRun(model_name, 
                            ds_name, 
                            'link_prediction', 
                            metric=metric,
                            omitted_test_dyads=ds_test_omitted.omitted_dyads_test, 
                            test_or_val=test_or_val, 
                            use_global_config_base=use_global_config_base, 
                            config_ranges=range_triplets)
        
        if val_dyads_to_omit is not None and not test_only:
            assert type(val_dyads_to_omit) == tuple
            assert utils.is_undirected(val_dyads_to_omit[0]) and utils.is_undirected(val_dyads_to_omit[1])

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

                if val_dyads_to_omit is None and not test_only: #sample random validation set
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



                trainer = Trainer(
                            dataset=ds_test_val_omitted,
                            model_name=model_name,
                            task='link_prediction',
                            config_triplets_to_change=config_triplets,
                            use_global_config_base=use_global_config_base,
                            attr_opt=False,
                            metric=metric,
                            device=device,
                )

                losses, acc_test, acc_val = trainer.train(
                            init_type='small_gaus',
                            init_feats=True,
                            acc_every=acc_every,
                            plot_every=plot_every,
                            verbose=verbose,
                            verbose_in_funcs=verbose_in_funcs
                        )
                
                # train returns acc test and acc val lists.
                if acc_test[metric]:
                    last_acc_test = acc_test[metric][-1]
                else:
                    last_acc_test = None
                
                if acc_val[metric]: # if val is not requested return an empty list
                    last_acc_val = acc_val[metric][-1]                    
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
        metric,
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
                                metric='auc',
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