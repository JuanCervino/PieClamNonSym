#!/usr/bin/env python3

import torch
from torch.autograd import grad
from torch_geometric.utils import subgraph
from torch_geometric.transforms import TwoHop
import numpy as np
import argparse
from datetime import datetime
import logging
from inspect import currentframe, getframeinfo
import os
import sys
from time import time
import json

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
grand_parent_dir = os.path.dirname(parent_dir)
grand_grand_parent_dir = os.path.dirname(grand_parent_dir)

# Add the parent directory to sys.path

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grand_parent_dir not in sys.path:
    sys.path.insert(0, grand_parent_dir)
if grand_grand_parent_dir not in sys.path:
    sys.path.insert(0, grand_grand_parent_dir)

from utils.printing_utils import printd, filename_n_line_str

from tests import tests
from utils import utils
from utils.plotting import *
import anomaly_detection as ad
from trainer import Trainer
from scripting_utils import print_prior_training_stats
from datasets.import_dataset import import_dataset



def main():    
    '''run big and ie on a chosen dataset to find the optimal number of communities and number of iterations.'''

    #todo: test run with small number of k and iters using srun
    #todo: find the optimal number of iterations from 1200 1500 ... 3000
    
    #           ARGS
    #=================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='name of the model')
    parser.add_argument('--ds_name', type=str, help='name of the dataset')
    parser.add_argument('--num_rounds', type=int, help='number of rounds')

    args = parser.parse_args()
    

    # ========= RESULTS FOLDERS =========
    if not torch.cuda.is_available():
        raise Exception('CUDA not available')
    device = torch.device('cuda')

    results_folder = f'results/attr/{args.model_name}_{args.ds_name}'
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    ds = import_dataset(args.ds_name)
    fat_ds = TwoHop()(ds)
    fat_ds.edge_attr = torch.ones(fat_ds.edge_index.shape[1]).bool()

    if args.ds_name in ['Flickr', 'ACM', 'BlogCatalog']:
        ds_to_use = ds
    elif args.ds_name in ['redditGGAD', 'photoGGAD', 'ellipticGGAD']:
        ds_to_use = fat_ds
    else:
        raise ValueError('ds_name not recognized')
    


    for i in range(args.num_rounds):
        round_num = i
        trainer = Trainer(
            model_name=args.model_name,
            device=device,
            dataset=ds_to_use.clone(),
            attr_opt=True,
            mighty_configs_dict=True,
            config_triplets_to_change=[]
        )

        losses_feats, losses_prior, accuracies_anomaly = trainer.train_model_on_params(
            init_type='small_gaus',
            init_feats=True,
            verbose=True,
            verbose_in_funcs=True
        )

        accuracies_folder = os.path.join(results_folder, f'accuracies_round_{round_num}')
        if not os.path.exists(accuracies_folder):
            os.makedirs(accuracies_folder)

        with open(os.path.join(accuracies_folder, 'accuracies.json'), 'w') as f:
            json.dump(accuracies_anomaly, f)

        del trainer.data


if __name__ == "__main__":
    main()

