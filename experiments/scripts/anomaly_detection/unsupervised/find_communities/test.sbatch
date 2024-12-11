#!/bin/bash

#SBATCH --job-name=test  # Job name
#SBATCH --output=out.out_test # Name of stdout output file
#SBATCH --error=err.err_test  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=15            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate finegrain                  

# Run the Python script with n_iter set to 1700
python -u find_communities.py --model_name 'bigclam' --ds_name 'ellipticGGAD' --ks 2 5 --n_iter 170 