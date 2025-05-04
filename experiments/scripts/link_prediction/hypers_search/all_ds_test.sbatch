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
python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search.py --model_name 'pieclam' --ds_name 'texas' --n_reps 5 --test_p 0.1 --test_only

python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search.py --model_name 'pieclam' --ds_name 'squirrel' --n_reps 5 --test_p 0.1 --test_only

python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search.py --model_name 'pclam' --ds_name 'photo' --n_reps 5 --test_p 0.1 --test_only

python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search.py --model_name 'pclam' --ds_name 'JohnsHopkins55' --n_reps 5 --test_p 0.1 --test_only
