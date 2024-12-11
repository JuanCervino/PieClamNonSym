#!/bin/bash

#SBATCH --job-name=ph3  # Job name
#SBATCH --output=out_phot3.out # Name of stdout output file
#SBATCH --error=err_phot3.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate finegrain                  

# Run the Python script with n_iter set to 1700
python -u mighty_config.py --model_name 'pclam' --ds_name 'photoGGAD' --n_back_forth 8 --scheduler_step_size 3 --n_iters_feats 500  --n_iters_prior 1300 --dim_feats 10 30 20  --noise_amp 0.005 0.05  0.01 --dim_attr 50 100 --lr_prior 0.000001 0.0001 0.00001 --lr_feats  0.000002 0.000001 0.0000025
## i think if all fails we know it's the noise amp
