#!/bin/bash

#SBATCH --job-name=all-ds  # Job name
#SBATCH --output=out/all-ds_%A_%a.out  # Output file (%A = job ID, %a = array index)
#SBATCH --error=err/all-ds_%A_%a.err   # Error file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --gres=gpu:1
#SBATCH --partition=euler
#SBATCH --account=euler
#SBATCH --array=0-10%4   # <<< ADD THIS: 0-x%y means create x tasks and run y at a time

# Set up environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clam

# Your command
# texas pieclam
#todo: make a list for every dataset of the test sets that it 
python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pieclam' --ds_name 'texas' --dim_feats 20 22 24 50 --s_reg 0.0 0.1 --lr_feats 0.000003 0.000004 0.000005 --n_iters_feats 1500 2000 3000 --lr_prior 0.000002 0.000005 0.000007 --n_iters_prior 1500 2000 3000  --n_reps 3 --val_p 0.1
# # texas ieclam
python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'ieclam' --ds_name 'texas' --dim_feats 40 50 60 --s_reg 0.0 0.1 --n_iters_feats 10000 13000 15000 --lr_feats 0.000002 0.000003 0.000005  --n_reps 3 --val_p 0.1

# squirrel pieclam
python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pieclam' --ds_name 'squirrel' --dim_feats 80 90 100 --s_reg 0.0 0.1 --lr_feats 0.000003 0.000005 0.000006 --n_iters_feats 500 700 1000 --lr_prior 0.000004 0.000005 0.000007 --n_iters_prior 1500 2000 3000 --noise_amps 0.05 0.1  --n_reps 3 --val_p 0.1 --random_search
# photo pclam
# squirrel pieclam
python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pclam' --ds_name 'photo' --dim_feats 130 150 170 --s_reg 0.0 0.1 --lr_feats 0.000003 0.000005 0.000006 --n_iters_feats 500 700 1000 --lr_prior 0.000002 0.000003 0.000005 --n_iters_prior 1000 1300 2000 --noise_amps 0.05 0.1  --n_reps 3 --val_p 0.1 --random_search
# hopkins pclam
python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pclam' --ds_name 'hopkins' --dim_feats 180 200 220 --s_reg 0.0 0.1 --lr_feats 0.000003 0.000005 0.000006 --n_iters_feats 700 750 800 --lr_prior 0.000002 0.000003 0.000005 --n_iters_prior 1500 1800 2000 --noise_amps 0.05 0.1  --n_reps 3 --val_p 0.1 --random_search


