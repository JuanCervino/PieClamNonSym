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
import link_prediction as lp


def main():    
    '''run big and ie on a chosen dataset to find the optimal number of communities and number of iterations.'''

    
    #           ARGS
    #=================================
    #todo: this already has test set path. i need to do it in the batch file
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='iegam', help='name of the model')
    parser.add_argument('--ds_name', type=str, default='squirrel', help='name of the dataset')

    # feat config triplet range
    parser.add_argument('--dim_feats', nargs='+', type=int, default=[], help='community dimension')
    parser.add_argument('--l1_regs', nargs='+', type=float, default=[], help='l1 regularization')
    parser.add_argument('--s_regs', nargs='+', type=float, default=[], help='s regularization')
    parser.add_argument('--n_iters_feats', nargs='+', type=int, default=[], help='number of iterations in fit feats')
    parser.add_argument('--lr_feats', nargs='+', type=float, default=[], help='lr feats')
    
    #prior config triplet range
    parser.add_argument('--dim_attr', nargs='+', type=int, default=[], help='attribute dimension')
    parser.add_argument('--n_iters_prior', nargs='+', type=int, default=[], help='number of iterations in fit prior')
    parser.add_argument('--lr_prior', nargs='+', type=float, default=[], help='lr prior')
    parser.add_argument('--noise_amps', nargs='+', type=float, default=[], help='noise amplitudes')
    parser.add_argument('--n_back_forth', nargs='+', type=int, default=[], help='number of back and forth iterations')
    parser.add_argument('--first_funcs_in_fit', nargs='+', type=str, default=[], help='first function in alternation')

    parser.add_argument('--use_global_config_base', action='store_true', help='whether to use the global config base') # if not given, use_global_config base is false
    parser.add_argument('--densify', action='store_true', help='whether to densify the data')
    parser.add_argument('--attr_opt', action='store_true', help='whether to optimize the attributes')
    parser.add_argument('--test_p', type=float, default=0.1, help='test proportion')
    parser.add_argument('--val_p', type=float, default=0.0, help='validation proportion')
    parser.add_argument('--val_dyads_path', type=str, default=None, help='path to the validation dyads')
    parser.add_argument('--test_dyads_path', type=str, default=None, help='path to the test dyads')
    parser.add_argument('--random_search', action='store_true', help='whether to use random search')
    parser.add_argument('--num_draws_random', type=int, default=1, help='number of draws for random search')
    parser.add_argument('--test_only', action='store_true', help='whether to test only')
    parser.add_argument('--n_reps', type=int, default=3, help='number of repetitions')
    parser.add_argument('--reverse_test_set_order', action='store_true', help='whether to reverse the order of the features')
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

    if args.model_name in ['pclam', 'piegam']:
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
    printd(f'Running cross val link splits for {args.ds_name} with model {args.model_name}')
    printd(f'{args=}')
    ou.cross_val_link_splits(
        ds_name=args.ds_name,
        model_name=args.model_name,
        range_triplets=range_triplets,
        use_global_config_base=False,
        attr_opt=args.attr_opt,
        val_p=args.val_p,
        random_search=args.random_search,
        num_draws_random=args.num_draws_random,
        test_only=args.test_only,
        reverse_test_set_order=args.reverse_test_set_order,
        n_reps=args.n_reps,
        device=device,
        plot_every=100000,
        acc_every=-1
        )

if __name__ == "__main__":
    main()

