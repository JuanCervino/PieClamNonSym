#!/bin/bash

#SBATCH --job-name=dens_ie  # Job name
#SBATCH --output=out.out2 # Name of stdout output file
#SBATCH --error=err.err2  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=15            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate finegrain                  

python -u densification_criteria.py --model_name 'ieclam'
