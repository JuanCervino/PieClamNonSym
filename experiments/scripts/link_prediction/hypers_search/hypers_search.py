#!/usr/bin/env python3

import torch
import numpy as np
import argparse
from datetime import datetime
import logging
from inspect import currentframe, getframeinfo
import os
import sys
import json
# local imports

# Add the root directory of the project to the Python path
# root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# if root_dir not in sys.path:
#     sys.path.insert(0, root_dir)
# script_dir = os.path.dirname(os.path.realpath(__file__))
# parent_dir = os.path.dirname(script_dir)
# grand_parent_dir = os.path.dirname(parent_dir)
# grand_grand_parent_dir = os.path.dirname(grand_parent_dir)

# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)
# if grand_parent_dir not in sys.path:
#     sys.path.insert(0, grand_parent_dir)
# if grand_grand_parent_dir not in sys.path:
#     sys.path.insert(0, grand_grand_parent_dir)

script_dir = os.path.dirname(os.path.realpath(__file__))

# Traverse up 4 levels and add each directory to sys.path
for _ in range(4):
    script_dir = os.path.dirname(script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

from utils.printing_utils import printd, filename_n_line_str
import experiments.optimization_utils as ou

from tests import tests
from utils import utils
from utils.plotting import *
import anomaly_detection as ad
from trainer import Trainer
from scripting_utils import print_prior_training_stats
from datasets.import_dataset import import_dataset
import utils.link_prediction as lp


def main():    
    '''run big and ie on a chosen dataset to find the optimal number of communities and number of iterations.'''

    
    #           ARGS
    #=================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ieclam', help='name of the model')
    parser.add_argument('--ds_name', type=str, default='squirrel', help='name of the dataset')

    parser.add_argument('--dim_feats', nargs='+', type=int, default=[36, 40, 46, 50, 56], help='community dimension')
    parser.add_argument('--l1_regs', nargs='+', type=float, default=[1], help='l1 regularization')
    parser.add_argument('--s_regs', nargs='+', type=float, default=[0.0], help='s regularization')
    parser.add_argument('--n_iters_feats', nargs='+', type=int, default=[13000], help='number of iterations in fit feats')
    parser.add_argument('--lr_feats', nargs='+', type=float, default=[3e-6, 1e-5], help='lr feats')
    
    parser.add_argument('--dim_attr', nargs='+', type=int, default=[100], help='attribute dimension')
    parser.add_argument('--n_iters_prior', nargs='+', type=int, default=[1300], help='number of iterations in fit prior')
    parser.add_argument('--lr_prior', nargs='+', type=float, default=[0.000001], help='lr prior')
    parser.add_argument('--noise_amps', nargs='+', type=float, default=[0.1], help='noise amplitudes')
    parser.add_argument('--n_back_forth', nargs='+', type=int, default=[7], help='number of back and forth iterations')
    parser.add_argument('--first_funcs_in_fit', nargs='+', type=str, default=['fit_prior'], help='first function in alternation')

    parser.add_argument('--global_config_base', action='store_true', help='whether to use the global config base')
    parser.add_argument('--densify', action='store_true', help='whether to densify the data')
    parser.add_argument('--attr_opt', action='store_true', help='whether to optimize the attributes')
    parser.add_argument('--test_p', type=float, default=0.0, help='test proportion')
    parser.add_argument('--val_p', type=float, default=0.1, help='validation proportion')
    parser.add_argument('--n_reps', type=int, default=3, help='number of repetitions')

    args = parser.parse_args()
    

    # ========= RESULTS FOLDERS =========
    if not torch.cuda.is_available():
        raise Exception('CUDA not available')
    device = torch.device('cuda')
    printd(f'Using device: {device}')


    range_triplets = [
        ['clamiter_init','s_reg', args.s_regs],
        ['clamiter_init','l1_reg', args.l1_regs],
        ['clamiter_init', 'dim_feat', args.dim_feats],
        ['feat_opt','n_iter', args.n_iters_feats],
        ['feat_opt','lr', args.lr_feats],
    ]

    if args.model_name in ['pclam', 'pieclam']:
        range_triplets += [
            # ['clamiter_init','dim_attr', args.dim_attr],
            ['back_forth', 'first_func_in_fit', args.first_funcs_in_fit],
            ['prior_opt','n_iter', args.n_iters_prior],
            ['prior_opt','lr', args.lr_prior],
            ['prior_opt','noise_amp', args.noise_amps],
            ['back_forth','n_back_forth', args.n_back_forth]
        ]
    # Create the file if it doesn't exist
    #todo: test the datasets: photo, texas, facebook, squirrel and crocodile
    ou.cross_val_link(
        ds_name=args.ds_name,
        model_name=args.model_name,
        range_triplets=range_triplets,
        use_global_config_base=args.global_config_base,
        densify=args.densify,
        attr_opt=args.attr_opt,
        test_p=args.test_p,
        val_p=args.val_p,
        n_reps=args.n_reps,
        device=device
        )

if __name__ == "__main__":
    main()

