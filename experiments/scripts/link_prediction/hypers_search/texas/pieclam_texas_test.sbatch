#!/bin/bash

#SBATCH --job-name=pie-t-e  # Job name
#SBATCH --output=out/out_pie_t-t_%A_%a.out  # Output file (%A = job ID, %a = array index)
#SBATCH --error=err/err_pie_t-t_%A_%a.err   # Error file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --gres=gpu:1
#SBATCH --partition=euler
#SBATCH --account=euler
#SBATCH --array=0-5%2   # <<< ADD THIS: 0-x%y means create x tasks and run y at a time

# Set up environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clam

# Your command
python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search.py --model_name 'pieclam' --ds_name 'texas' --n_reps 3 --test_p 0.1 --test_only