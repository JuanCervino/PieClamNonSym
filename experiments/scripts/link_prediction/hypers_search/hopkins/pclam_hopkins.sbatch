#!/bin/bash

#SBATCH --job-name=pc-h-f  # Job name
#SBATCH --output=out_pc-h-f.out # Name of stdout output file
#SBATCH --error=err_pc-h-f.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ../hypers_search.py --model_name 'pclam' --ds_name 'JohnsHopkins55' --dim_feat 200 170 150 --n_iters_feats  750 --n_iters_prior 1800 --lr_feats 0.0000055 --lr_prior 0.000003 --noise_amps 0.1 --global_config_base --n_reps 10 --first_funcs_in_fit 'fit_feats'


