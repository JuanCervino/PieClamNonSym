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
    '''check to see if the losses convay a densification criterion'''
    printd('starting densification_criteria.py')

    #           ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='name of the model')
    parser.add_argument('--n_iter', type=int,default=200000, help='number of iterations')
    parser.add_argument('--lr', type=float, default=0.000001, help='learning rate')

    args = parser.parse_args()
    #====== end args ==========

    # SAVE
    
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    model_folder = os.path.join(results_folder, args.model_name)
        
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    printd(f'''starting densification_criteria.py on device: {device}''')
    # -- COMPARE LOSSES WITH WITHOUT DENSIFICATION  -----
    #? REDDIT
    config_triplets = [
        ['feat_opt', 'n_iter', args.n_iter],
        ['feat_opt', 'lr', args.lr],
        ]

    slim_reddit_iegam_losses, slim_reddit_iegam_anomaly_auc, slim_reddit_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'redditGGAD', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=False,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )

    fat_reddit_iegam_losses, fat_reddit_iegam_anomaly_auc, fat_reddit_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'redditGGAD', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=True,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )

    #? PHOTO
    slim_photo_iegam_losses, slim_photo_iegam_anomaly_auc, slim_photo_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'photoGGAD', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=False,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )

    fat_photo_iegam_losses, fat_photo_iegam_anomaly_auc, fat_photo_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'photoGGAD', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=True,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )

    #? ELLIPTIC
    slim_elliptic_iegam_losses, slim_elliptic_iegam_anomaly_auc, slim_elliptic_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'ellipticGGAD', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=False,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )

    fat_elliptic_losses, fat_elliptic_iegam_anomaly_auc, fat_elliptic_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'ellipticGGAD', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=True,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )

    #? BlogCatalog
    slim_blog_iegam_losses, slim_blog_iegam_anomaly_auc, slim_blog_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'BlogCatalog', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=False,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )

    fat_blog_iegam_losses, fat_blog_iegam_anomaly_auc, fat_blog_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'BlogCatalog', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=True,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )
    #? Flickr
    slim_flickr_iegam_losses, slim_flickr_iegam_anomaly_auc, slim_flickr_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'Flickr', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=False,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )

    fat_flickr_iegam_losses, fat_flickr_iegam_anomaly_auc, fat_flickr_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'Flickr', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=True,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )

    #? ACM
    slim_acm_iegam_losses, slim_acm_iegam_anomaly_auc, slim_acm_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'ACM', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=False,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )

    fat_acm_iegam_losses, fat_acm_iegam_anomaly_auc, fat_acm_iegam_link_auc = ad.classify_anomaly_link_earlystop(
        'ACM', 
        args.model_name, 
        config_triplets_clam=config_triplets,
        use_fat=True,
        percentage_of_dyads_to_omit=0.2,
        device=device,
        verbose=False
        )

    np.save(np.array(slim_reddit_iegam_losses), os.path.join(model_folder, 'slim_reddit_iegam_losses.npy'))
    np.save(np.array(slim_reddit_iegam_anomaly_auc), os.path.join(model_folder, 'slim_reddit_iegam_anomaly_auc.npy'))
    np.save(np.array(slim_reddit_iegam_link_auc), os.path.join(model_folder, 'slim_reddit_iegam_link_auc.npy'))

    np.save(np.array(fat_reddit_iegam_losses), os.path.join(model_folder, 'fat_reddit_iegam_losses.npy'))
    np.save(np.array(fat_reddit_iegam_anomaly_auc), os.path.join(model_folder, 'fat_reddit_iegam_anomaly_auc.npy'))
    np.save(np.array(fat_reddit_iegam_link_auc), os.path.join(model_folder, 'fat_reddit_iegam_link_auc.npy'))
    
    np.save(np.array(slim_photo_iegam_losses), os.path.join(model_folder, 'slim_photo_iegam_losses.npy'))
    np.save(np.array(slim_photo_iegam_anomaly_auc), os.path.join(model_folder, 'slim_photo_iegam_anomaly_auc.npy'))
    np.save(np.array(slim_photo_iegam_link_auc), os.path.join(model_folder, 'slim_photo_iegam_link_auc.npy'))
    np.save(np.array(fat_photo_iegam_losses), os.path.join(model_folder, 'fat_photo_iegam_losses.npy'))
    np.save(np.array(fat_photo_iegam_anomaly_auc), os.path.join(model_folder, 'fat_photo_iegam_anomaly_auc.npy'))
    np.save(np.array(fat_photo_iegam_link_auc), os.path.join(model_folder, 'fat_photo_iegam_link_auc.npy'))

    
    np.save(np.array(slim_elliptic_iegam_losses), os.path.join(model_folder, 'slim_elliptic_iegam_losses.npy'))
    np.save(np.array(slim_elliptic_iegam_anomaly_auc), os.path.join(model_folder, 'slim_elliptic_iegam_anomaly_auc.npy'))
    np.save(np.array(slim_elliptic_iegam_link_auc), os.path.join(model_folder, 'slim_elliptic_iegam_link_auc.npy'))
    np.save(np.array(fat_elliptic_losses), os.path.join(model_folder, 'fat_elliptic_losses.npy'))
    np.save(np.array(fat_elliptic_iegam_anomaly_auc), os.path.join(model_folder, 'fat_elliptic_iegam_anomaly_auc.npy'))
    np.save(np.array(fat_elliptic_iegam_link_auc), os.path.join(model_folder, 'fat_elliptic_iegam_link_auc.npy'))
    
    np.save(np.array(slim_blog_iegam_losses), os.path.join(model_folder, 'slim_blog_iegam_losses.npy'))
    np.save(np.array(slim_blog_iegam_anomaly_auc), os.path.join(model_folder, 'slim_blog_iegam_anomaly_auc.npy'))
    np.save(np.array(slim_blog_iegam_link_auc), os.path.join(model_folder, 'slim_blog_iegam_link_auc.npy'))
    np.save(np.array(fat_blog_iegam_losses), os.path.join(model_folder, 'fat_blog_iegam_losses.npy'))
    np.save(np.array(fat_blog_iegam_anomaly_auc), os.path.join(model_folder, 'fat_blog_iegam_anomaly_auc.npy'))
    np.save(np.array(fat_blog_iegam_link_auc), os.path.join(model_folder, 'fat_blog_iegam_link_auc.npy'))
    
    np.save(np.array(slim_flickr_iegam_losses), os.path.join(model_folder, 'slim_flickr_iegam_losses.npy'))
    np.save(np.array(slim_flickr_iegam_anomaly_auc), os.path.join(model_folder, 'slim_flickr_iegam_anomaly_auc.npy'))
    np.save(np.array(slim_flickr_iegam_link_auc), os.path.join(model_folder, 'slim_flickr_iegam_link_auc.npy'))
    np.save(np.array(fat_flickr_iegam_losses), os.path.join(model_folder, 'fat_flickr_iegam_losses.npy'))
    np.save(np.array(fat_flickr_iegam_anomaly_auc), os.path.join(model_folder, 'fat_flickr_iegam_anomaly_auc.npy'))
    np.save(np.array(fat_flickr_iegam_link_auc), os.path.join(model_folder, 'fat_flickr_iegam_link_auc.npy'))
    
    
    np.save(np.array(slim_acm_iegam_losses), os.path.join(model_folder, 'slim_acm_iegam_losses.npy'))
    np.save(np.array(slim_acm_iegam_anomaly_auc), os.path.join(model_folder, 'slim_acm_iegam_anomaly_auc.npy'))
    np.save(np.array(slim_acm_iegam_link_auc), os.path.join(model_folder, 'slim_acm_iegam_link_auc.npy'))
    np.save(np.array(fat_acm_iegam_losses), os.path.join(model_folder, 'fat_acm_iegam_losses.npy'))
    np.save(np.array(fat_acm_iegam_anomaly_auc), os.path.join(model_folder, 'fat_acm_iegam_anomaly_auc.npy'))
    np.save(np.array(fat_acm_iegam_link_auc), os.path.join(model_folder, 'fat_acm_iegam_link_auc.npy'))

    #todo: make the percentage of edges to omit a function of the average DEGREE or something. not the number of edges because a graph can have many nodes and many edges and little nodes and many edges...


if __name__ == "__main__":
    main()
    