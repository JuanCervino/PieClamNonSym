#!/bin/bash

#SBATCH --job-name=pie-t-d  # Job name
#SBATCH --output=out_pie_d.out # Name of stdout output file
#SBATCH --error=err_pie_d.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler2             # Partition name
#SBATCH --account=euler2               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                 

python -u ../hypers_search.py --model_name 'piegam' --ds_name 'ogbl-ddi' --dim_feats 22 24 --n_iters_feats 2300 2000 --n_iters_prior 1500 2000  --lr_feats 0.000004 0.000005 0.000003 --lr_prior 0.000004 0.000003 0.000005 --noise_amps  0.1 0.05 --l1_regs 1.0 --s_regs 0.0 --global_config_base --n_reps 10 --first_funcs_in_fit 'fit_feats' 'fit_prior' 
