#!/bin/bash

#SBATCH --job-name=mi-ie  # Job name
#SBATCH --output=out_ieclam.out # Name of stdout output file
#SBATCH --error=err_ieclam.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

##SBATCH --partition=euler             # Partition name
##SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate finegrain                  


python -u mighty_config.py --model_name 'ieclam' --ds_name 'redditGGAD' --dim_feats 10 20 24 30 --n_iters_feats 3000 --lr_feats 0.00005 0.000001

python -u mighty_config.py --model_name 'ieclam' --ds_name 'ellipticGGAD' --dim_feats 10 20 24 30 --n_iters_feats 3000 --lr_feats 0.000005 0.000001

python -u mighty_config.py --model_name 'ieclam' --ds_name 'photoGGAD' --dim_feats 10 20 24 30 --n_iters_feats 3000 --lr_feats 0.000005 0.000001

python -u mighty_config.py --model_name 'ieclam' --ds_name 'BlogCatalog' --dim_feats 10 20 24 30 --n_iters_feats 3000 --lr_feats 0.00001 0.000001

python -u mighty_config.py --model_name 'ieclam' --ds_name 'ACM' --dim_feats 10 20 24 30 --n_iters_feats 3000 --lr_feats 0.00001 0.000001

python -u mighty_config.py --model_name 'ieclam' --ds_name 'Flickr' --dim_feats 10 20 24 30 --n_iters_feats 3000 --lr_feats 0.00001 0.000001
