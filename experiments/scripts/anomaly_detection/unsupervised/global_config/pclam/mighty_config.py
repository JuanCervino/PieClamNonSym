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
    parser.add_argument('--dim_feats', nargs='+', type=int, help='community dimension')
    parser.add_argument('--dim_attr', nargs='+', type=int, help='attribute dimension')
    parser.add_argument('--n_iters_feats', nargs='+', type=int, help='number of iterations in fit feats')
    parser.add_argument('--lr_feats', nargs='+', type=float, help='lr feats')
    parser.add_argument('--n_iters_prior', nargs='+', type=int, help='number of iterations in fit prior')
    parser.add_argument('--lr_prior', nargs='+', type=float, help='lr prior')
    parser.add_argument('--noise_amps', nargs='+', type=float, help='noise amplitudes')

    args = parser.parse_args()
    

    # ========= RESULTS FOLDERS =========
    if not torch.cuda.is_available():
        raise Exception('CUDA not available')
    device = torch.device('cuda')

    results_folder = 'results'
    
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
    
    
    if args.dim_attr is None:
        args.dim_attr = [-1]
    if args.n_iters_prior is None:
        args.n_iters_prior = [-1]
    if args.lr_prior is None:
        args.lr_prior = [-1]
    if args.noise_amps is None:
        args.noise_amps = [-1]


    round_num = 0

    for dim_feats in args.dim_feats:
        for dim_attr in args.dim_attr:
            for n_iters_feats in args.n_iters_feats:
                for n_iters_prior in args.n_iters_prior:
                    for lr_prior in args.lr_prior:
                        for noise_amp in args.noise_amps:
                            for lr_feats in args.lr_feats:
                                printd(f'\n {round_num= } \n\n')

                                config_triplets = [
                                    ['clamiter_init','dim_feats', dim_feats],
                                    ['feat_opt','n_iter', n_iters_feats],
                                    ['feat_opt','lr', lr_feats],
                                ]

                                if dim_attr > 0:
                                    config_triplets.append(['clamiter_init','dim_attr', dim_attr])
                                    config_triplets.append(['prior_opt','n_iter', n_iters_prior])
                                    config_triplets.append(['prior_opt','lr', lr_prior])
                                    config_triplets.append(['prior_opt', 'noise_amp', noise_amp])

                                trainer = Trainer(
                                    model_name=args.model_name,
                                    device=device,
                                    dataset=ds_to_use.clone(),
                                    attr_opt=True,
                                    use_global_config_base=True,
                                    config_triplets_to_change=config_triplets
                                )

                                trainer.train_model_on_params(
                                    init_type='small_gaus',
                                    init_feats=True,
                                    verbose=True,
                                    verbose_in_funcs=True
                                )
       
                                del trainer.data
        
                                round_num +=1

if __name__ == "__main__":
    main()

