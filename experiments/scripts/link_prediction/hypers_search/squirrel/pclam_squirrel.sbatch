#!/bin/bash

#SBATCH --job-name=pc-s4  # Job name
#SBATCH --output=out_pc-s4.out # Name of stdout output file
#SBATCH --error=err_pc-s4.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ../hypers_search.py --model_name 'pclam' --ds_name 'squirrel' --dim_feat 56 70 --n_iters_feats 500 700 --n_iters_prior 1000 1300 2000 --lr_feats 0.000003 0.000005 --lr_prior 0.000003 0.000005 --noise_amps 0.1 0.05 --global_config_base --n_reps 3 --first_funcs_in_fit 'fit_prior' 'fit_feats'


