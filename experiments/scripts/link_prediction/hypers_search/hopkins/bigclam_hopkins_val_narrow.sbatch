#!/bin/bash

#SBATCH --job-name=bc-h-vn  # Job name
#SBATCH --output=out/bc-h-vn.out # Name of stdout output file
#SBATCH --error=err/bc-h-vn.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'bigclam' --ds_name 'JohnsHopkins55' --dim_feats  60 70 90 --lr_feats 0.000006 0.000007 0.000009 --n_iters_feats  6000 8000 10000 --n_reps 3 --val_p 0.1 --random_search --num_draws_random 50


