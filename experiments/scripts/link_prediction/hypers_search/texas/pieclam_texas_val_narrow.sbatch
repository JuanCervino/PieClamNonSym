#!/bin/bash

#SBATCH --job-name=pc-t-vn  # Job name
#SBATCH --output=out/out_pc-t-vn.out # Name of stdout output file
#SBATCH --error=err/err_pc-t-vn.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pieclam' --ds_name 'texas' --dim_feats 20 22 24 50 --s_reg 0.0 0.1 --lr_feats 0.000003 0.000004 0.000005 --n_iters_feats 1500 2000 3000 --lr_prior 0.000002 0.000005 0.000007 --n_iters_prior 1500 2000 3000  --n_reps 3 --val_p 0.1 --random_search


