#!/bin/bash

#SBATCH --job-name=mighty_f  # Job name
#SBATCH --output=out_flickr.out # Name of stdout output file
#SBATCH --error=err_flickr.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate finegrain                  

# Run the Python script with n_iter set to 1700
python -u mighty_config.py --model_name 'piegam' --ds_name 'Flickr' --dim_feats 24 30 --dim_attr 50 100 --n_iters_feats 500 1000 2000 --lr_feats 0.00001 0.000001 --noise_amp 0.05 0.01 --n_iters_prior 800 1300 --lr_prior 0.0001 0.00001 0.000001
