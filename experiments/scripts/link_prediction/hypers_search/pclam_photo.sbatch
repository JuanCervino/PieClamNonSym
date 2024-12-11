#!/bin/bash

#SBATCH --job-name=pc-p-f  # Job name
#SBATCH --output=out_pc-p-f.out # Name of stdout output file
#SBATCH --error=err_pc-p-f.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u hypers_search.py --model_name 'pclam' --ds_name 'photo' --dim_feat 110 150 120 --n_iters_feats 700 --n_iters_prior 2000 1300 --lr_feats 0.000005 --lr_prior 0.000003 0.000005 --noise_amps 0.1 --global_config_base --n_reps 10 --first_funcs_in_fit 'fit_feats'


