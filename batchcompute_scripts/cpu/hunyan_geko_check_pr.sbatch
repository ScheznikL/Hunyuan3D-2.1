#!/bin/bash
#
#SBATCH --job-name=check-folder  # Job name for tracking
#SBATCH --partition=gecko       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # Number of CPU threads used by your job
#SBATCH --time=1-00:00:00       # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_cpu%j.out # Standard out from your job
#SBATCH --error=joboutput_cpu%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1

bash ./hy3dshape/tools/check.sh