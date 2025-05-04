#!/bin/bash

#SBATCH --job-name=bc-s  # Job name
#SBATCH --output=out_bc-s.out # Name of stdout output file
#SBATCH --error=err_bc-s.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ../hypers_search.py --model_name "bigclam" --ds_name "squirrel" --dim_feats 70 80 90 100 --l1_regs 1 --n_iters_feats 7500 --global_config_base --n_reps 10


