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
    parser.add_argument('--ds_name', type=str, help='name of the dataset')
    parser.add_argument('--ks', nargs='+', type=int, help='list of pairs: every even number is number of repetitions and every odd number is the number of communities')
    parser.add_argument('--n_iter', nargs='+', type=int, help='number of iterations')
    parser.add_argument('--model_name', type=str, help='name of the model')

    args = parser.parse_args()
    

    # ========= RESULTS FOLDERS =========
    if not torch.cuda.is_available():
        raise Exception('CUDA not available')
    device = torch.device('cuda')

    results_folder = 'results'
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    for n_iter in args.n_iter:

        aucs_folder = os.path.join(results_folder, args.ds_name, args.model_name, str(n_iter))
        
        if not os.path.exists(aucs_folder):
            os.makedirs(aucs_folder)
        
        # ====================================
        
        # ========== IMPORT DATA =============
        ds_with_anomalies = import_dataset(args.ds_name)

        if args.ds_name[-1] == 'D':
            ds_for_opt = TwoHop()(ds_with_anomalies)
        else:
            ds_for_opt = ds_with_anomalies
        # ====================================

        # ======== CROSSVAL AND SAVE =========
        assert len(args.ks) % 2 == 0, 'ks must be a list of pairs: every even number is number of repetitions and every odd number is the number of communities'

        num_k = args.ks[::2]
        num_communities = args.ks[1::2]

        ks = np.concatenate([num_k[i]*[num_communities[i]] for i in range(len(num_k))]).tolist() 
        ks_aucs_aps = ad.cross_val_communities_unsupervised(ds_for_opt, args.model_name, ks, n_iter, device=device, return_ap=True, verbose=False)

        ad.save_ks_aucs(args.model_name, ks_aucs_aps, save_dir=aucs_folder)
        
        # this will be a data collection method for the number of iterations, number of communities, and auc scores 
        

if __name__ == "__main__":
    main()
    