#!/bin/bash

#SBATCH --job-name=pc-p-vn  # Job name
#SBATCH --output=out/out_pc-p-vn.out # Name of stdout output file
#SBATCH --error=err/err_pc-p-vn.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pclam' --ds_name 'photo' --dim_feats 130 150 170 --s_reg 0.0 0.1 --lr_feats 0.000003 0.000005 0.000006 --n_iters_feats 500 700 1000 --lr_prior 0.000002 0.000003 0.000005 --n_iters_prior 1000 1300 2000 --noise_amps 0.05 0.1  --n_reps 2 --val_p 0.1 --random_search


