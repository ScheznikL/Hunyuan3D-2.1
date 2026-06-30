#!/bin/bash
#
#SBATCH --job-name=jupyter     # Job name for tracking
#SBATCH --partition=tiger      # Partition you wish to use
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48      # Number of CPU threads used by your job
#SBATCH --mem-per-cpu=3000     # Memory needed per thread (i.e. 3000MB * 6 threads = 60GB)
#SBATCH --time=12:00:00        # Job time limit set to 12 hours
#
#SBATCH --output=jupyter.log
#SBATCH --error=jupyter.err
#SBATCH --mail-type=BEGIN # Get email when your allocation starts

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1

conda activate /dcs/pg24/u5745134/.conda/envs/hunyancondaenv

jupyter lab --no-browser --ip=0.0.0.0 --port=11934
