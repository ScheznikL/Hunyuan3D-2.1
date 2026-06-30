#!/bin/bash
#
#SBATCH --job-name=example-gpu  # Job name for tracking
#SBATCH --partition=gecko       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=12:00:00       # Job time limit set to 12 hours
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=jupiteroutput_%j.out # Standard out from your job
#SBATCH --error=jupiterboutput_%j.err  # Standard error from your job

## --mail-type=BEGIN # Get email when your allocation starts

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1

conda activate /dcs/pg24/u5745134/.conda/envs/hunyancondaenv

jupyter lab --no-browser --ip=0.0.0.0 --port=11934
