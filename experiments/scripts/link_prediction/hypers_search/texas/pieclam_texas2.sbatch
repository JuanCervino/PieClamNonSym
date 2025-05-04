#!/bin/bash

#SBATCH --job-name=pie-t-2  # Job name
#SBATCH --output=out_pie_t-2.out # Name of stdout output file
#SBATCH --error=err_pie_t-2.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                 


python -u ../hypers_search.py --model_name 'piegam' --ds_name 'texas' --dim_feats 22 --n_iters_feats 2000 --n_iters_prior 1300 --lr_feats 0.000005 --lr_prior 0.000003 --noise_amps  0.1 --s_regs 0.0 --l1_regs 1.0 --global_config_base --n_reps 10 --first_funcs_in_fit 'fit_prior'
