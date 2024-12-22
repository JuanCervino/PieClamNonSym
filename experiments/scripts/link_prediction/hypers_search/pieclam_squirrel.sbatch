#!/bin/bash

#SBATCH --job-name=pie-s-f  # Job name
#SBATCH --output=out_pie_s-f.out # Name of stdout output file
#SBATCH --error=err_pie_s-f.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                 


python -u hypers_search.py --model_name 'pieclam' --ds_name 'squirrel' --dim_feat 80 90 --l1_regs 0.01 1.0 --s_regs 0.1 --n_iters_feats 700 --n_iters_prior 2000 --lr_feats 0.000005 --lr_prior 0.000005 --noise_amps 0.1 --global_config_base --n_reps 10 --first_funcs_in_fit 'fit_feats'
