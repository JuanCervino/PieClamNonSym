#!/bin/bash

#SBATCH --job-name=bc-p  # Job name
#SBATCH --output=out_bc-p.out # Name of stdout output file
#SBATCH --error=err_bc-p.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

##SBATCH --partition=euler             # Partition name
##SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ../hypers_search.py --model_name 'bigclam' --ds_name 'photo' --dim_feats 100 110 120 130 --global_config_base --l1_regs 1.0 0.8 1.2 --n_iters_feats 7500 --n_reps 10


