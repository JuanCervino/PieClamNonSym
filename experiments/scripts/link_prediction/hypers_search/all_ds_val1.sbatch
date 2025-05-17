#!/bin/bash

#SBATCH --job-name=all-ds1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --gres=gpu:1
## SBATCH --partition=euler
## SBATCH --account=euler
#SBATCH --array=0-4%4  # 5 jobs total

source ~/miniconda3/etc/profile.d/conda.sh
conda activate clam

# Define job names and commands
declare -a NAMES=(
"texas_pieclam1"
"texas_ieclam1"
"squirrel_pieclam1"
"photo_pclam1"
"hopkins_pclam1"
)

declare -a JOBS=(

"python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pclam' --ds_name 'hopkins' --dim_feats 180 200 220 --s_reg 0.0 0.1 --lr_feats 0.000003 0.000005 0.000006 --n_iters_feats 700 750 800 --lr_prior 0.000002 0.000003 0.000005 --n_iters_prior 1500 1800 2000 --noise_amps 0.05 0.1 --n_reps 3 --val_p 0.1"

"python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pclam' --ds_name 'photo' --dim_feats 130 150 170 --s_reg 0.0 0.1 --lr_feats 0.000003 0.000005 0.000006 --n_iters_feats 500 700 1000 --lr_prior 0.000002 0.000003 0.000005 --n_iters_prior 1000 1300 2000 --noise_amps 0.05 0.1 --n_reps 3 --val_p 0.1"

"python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pieclam' --ds_name 'squirrel' --dim_feats 80 90 100 --s_reg 0.0 0.1 --lr_feats 0.000003 0.000005 0.000006 --n_iters_feats 500 700 1000 --lr_prior 0.000004 0.000005 0.000007 --n_iters_prior 1500 2000 3000 --noise_amps 0.05 0.1 --n_reps 3 --val_p 0.1"


"python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pieclam' --ds_name 'texas' --dim_feats 20 22 24 50 --s_reg 0.0 0.1 --lr_feats 0.000003 0.000004 0.000005 --n_iters_feats 1500 2000 3000 --lr_prior 0.000002 0.000005 0.000007 --n_iters_prior 1500 2000 3000 --n_reps 3 --val_p 0.1"

"python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'ieclam' --ds_name 'texas' --dim_feats 40 50 60 --s_reg 0.0 0.1 --n_iters_feats 10000 13000 15000 --lr_feats 0.000002 0.000003 0.000005 --n_reps 3 --val_p 0.1"
)

name=${NAMES[$SLURM_ARRAY_TASK_ID]}
cmd=${JOBS[$SLURM_ARRAY_TASK_ID]}

# Set custom log file paths
stdout="out/${name}.out"
stderr="err/${name}.err"

# Create directories if needed
mkdir -p out err logs

# Run the job and redirect output
echo "[$(date)] Starting job: $name"
echo "Command: $cmd"
eval "$cmd" > "$stdout" 2> "$stderr"