#!/bin/bash

#SBATCH --job-name=ie-s  # Job name
#SBATCH --output=out_ie-s.out # Name of stdout output file
#SBATCH --error=err_ie-s.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

##SBATCH --partition=euler             # Partition name
##SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u hypers_search.py --model_name iegam --ds_name squirrel --dim_feats 56 60 66 70 76 80 90 100 --global_config_base --n_reps 10 --s_reg 0.1 0.01 0.0 --l1_regs 1 0.01 --n_iters_feats 4500


