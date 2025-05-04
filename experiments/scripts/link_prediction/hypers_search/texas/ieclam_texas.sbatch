#!/bin/bash

#SBATCH --job-name=ie-t  # Job name
#SBATCH --output=out_ie-t.out # Name of stdout output file
#SBATCH --error=err_ie-t.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

##SBATCH --partition=euler             # Partition name
##SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ../hypers_search.py --model_name 'iegam' --ds_name 'texas' --dim_feats 20 22 24 26 30 40 --global_config_base --l1_regs 1.0 0.01 --s_regs 0.1 0.01 --n_iters_feats 7500 --n_reps 10


