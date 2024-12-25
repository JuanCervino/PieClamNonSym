#!/bin/bash

#SBATCH --job-name=red1  # Job name
#SBATCH --output=out_reddit1.out # Name of stdout output file
#SBATCH --error=err_reddit1.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate finegrain                  

# Run the Python script with n_iter set to 1700
python -u global_config.py --model_name 'pieclam' --ds_name 'redditGGAD' --n_iters_feats 500  --n_iters_prior 1300 --dim_feats 30 --noise_amp 0.01 0.05 --dim_attr 100 50 --lr_prior 0.00001 0.000001 0.0001 --lr_feats 0.0000025 0.000002 
