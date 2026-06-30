#!/bin/bash
#
#SBATCH --job-name=img-gecko-test  # Job name for tracking
#SBATCH --partition=gecko       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=2-00:00:00       # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./out/log_100_joboutput_gecko_100_gpu%j.out # Standard out from your job
#SBATCH --error=./out/log_100_joboutput_gecko_100_gpu%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

module load Blender/4.2.1
module load CUDA/12.6.2

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models
export HF_HOME=/dcs/large/u5745134/cache/huggingface

cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1
conda activate hunyuan3D_torch28

#python ./experiments/utils_fantasia3d/render_normals.py
python ./experiments/exp_prior_init_img.py