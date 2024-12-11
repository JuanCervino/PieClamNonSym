#!/bin/bash

#SBATCH --job-name=test_script      # Job name
#SBATCH --output=test_script.out    # Name of stdout output file
#SBATCH --error=test_script.err     # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=15            
#SBATCH --gres=gpu:1                  # Request one GPU

#SBATCH --partition=euler             # Partition name
#SBATCH --account=euler               # Account name

source ~/miniconda3/etc/profile.d/conda.sh                
conda activate finegrain                  

# Run the Python script
python test_script.py