#!/bin/bash

#SBATCH --job-name=re_a  # Job name
#SBATCH --output=out_red_attr.out # Name of stdout output file
#SBATCH --error=err_red_attr.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate finegrain                  

# Run the Python script with n_iter set to 1700
python -u confidence_intervals_attr.py --model_name 'piegam' --ds_name 'redditGGAD' --num_rounds 10
