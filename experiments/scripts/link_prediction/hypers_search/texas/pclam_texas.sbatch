#!/bin/bash

#SBATCH --job-name=pc-t  # Job name
#SBATCH --output=out_pc-t.out # Name of stdout output file
#SBATCH --error=err_pc-t.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

##SBATCH --partition=euler             # Partition name
##SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ../hypers_search.py --model_name 'pclam' --ds_name 'texas' --dim_feats 14 18 20 22 24 --n_iters_feats 500 1000 2000 --n_iters_prior 1300 2000 1000 --l1_regs 0.5 1.0 0.0  --lr_feats 0.000005 0.000003 --lr_prior 0.000003 0.000005 --noise_amps 0.1 0.05 --global_config_base --first_funcs_in_fit 'fit_prior' --n_reps 6

