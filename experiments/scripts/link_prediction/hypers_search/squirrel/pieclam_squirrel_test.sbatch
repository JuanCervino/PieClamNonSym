#!/bin/bash

#SBATCH --job-name=pc-s-t  # Job name
#SBATCH --output=out/out_pc-s-t.out # Name of stdout output file
#SBATCH --error=err/err_pc-s-t.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

## SBATCH --partition=euler             # Partition name
## SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py \
 --model_name 'pieclam' \
 --ds_name 'squirrel' \
 --test_only \
 --n_reps 10

