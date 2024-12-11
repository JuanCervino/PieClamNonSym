#!/bin/bash

#SBATCH --job-name=new_clam  # Job name
#SBATCH --output=out.out1 # Name of stdout output file
#SBATCH --error=err.err1  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=15            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate finegrain                  

# Run the Python script with n_iter set to 1700
python -u find_communities.py --model_name 'bigclam' --ds_name 'ellipticGGAD' --ks 10 2 10 3 10 4 10 5 10 6 10 7 10 8 10 9 10 10 10 15 10 16 10 17 10 18 --n_iter 50 100 170 200 500 

# Run the Python script with n_iter set to 2000
python -u find_communities.py --model_name 'bigclam' --ds_name 'photoGGAD' --ks --ks 10 2 10 3 10 4 10 5 10 6 10 7 10 8 10 9 10 10 10 15 10 16 10 17 10 18 --n_iter 50 100 170 200 500 

# Run the Python script with n_iter set to 2500
python -u find_communities.py --model_name 'bigclam' --ds_name 'tfFinanceGGAD' --ks 10 2 10 3 10 4 10 5 10 6 10 7 10 8 10 9 10 10 10 15 10 16 10 17 10 18 --n_iter 50 100 170 200 500

python -u find_communities.py --model_name 'bigclam' --ds_name 'ACM' --ks 10 5 --ks 10 2 10 3 10 4 10 5 10 6 10 7 10 8 10 9 10 10 10 15 10 16 10 17 10 18 --n_iter 50 100 170 200 500


python -u find_communities.py --model_name 'bigclam' --ds_name 'Flickr' --ks 10 --ks 10 2 10 3 10 4 10 5 10 6 10 7 10 8 10 9 10 10 10 15 10 16 10 17 10 18 --n_iter 50 100 170 200 500

python -u find_communities.py --model_name 'bigclam' --ds_name 'BlogCatalog' --ks 10 2 10 3 10 4 10 5 10 6 10 7 10 8 10 9 10 10 10 15 10 16 10 17 10 18 --n_iter 50 100 170 200 500