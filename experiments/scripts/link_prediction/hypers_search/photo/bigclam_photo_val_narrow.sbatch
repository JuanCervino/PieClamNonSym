#!/bin/bash

#SBATCH --job-name=bc-p-vn  # Job name
#SBATCH --output=out/bc-p-vn.out # Name of stdout output file
#SBATCH --error=err/bc-p-vn.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'bigclam' --ds_name 'photo' --dim_feats 40 50 60 --lr_feats 0.000002 0.000003 0.000005 --n_iters_feats  7000  10000 13000 --n_reps 3 --val_p 0.1 --random_search --num_draws_random 50


