#!/bin/bash

#SBATCH --job-name=new_gam  # Job name
#SBATCH --output=out.out2 # Name of stdout output file
#SBATCH --error=err.err2  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=15            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate finegrain                  

# Run the Python script with n_iter set to 1700
python -u find_communities.py --model_name 'ieclam' --ds_name 'redditGGAD' --ks 10 5 10 10 10 15 10 16 10 17 10 18 10 19 10 20 10 21 10 22 10 23 10 24 --n_iter 600 800 1000 1200 1500 1700 2000 

python -u find_communities.py --model_name 'bigclam' --ds_name 'redditGGAD' --ks 10 5 10 10 10 15 10 16 10 17 10 18 10 19 10 20 10 21 10 22 10 23 10 24 --n_iter 600 800 1000 1200 1500 1700 2000 