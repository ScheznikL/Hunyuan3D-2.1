#!/bin/bash
#
#SBATCH --job-name=shardeagle  # Job name for tracking
#SBATCH --partition=eagle       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6       # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=1-00:00:00       #
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./joboutput_eagle_batch2_shard1%j.out # Standard out from your job
#SBATCH --error=./joboutput_eagle_batch2_shard1%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models

export HF_HOME=/dcs/large/u5745134/cache/huggingface

#cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/hy3dshape/tools/render
cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/
conda activate hunyuan3D_torch28

#module load Blender/4.2.1
module load CUDA
# OpenEXR support for OpenCV
# export OPENCV_IO_ENABLE_OPENEXR=1

#export CUDA_VISIBLE_DEVICES=1 #1
# export SHARD_ID=1 #0
# export NUM_SHARDS=2

# bash start_render_part.sh
python ./experiments/analyze_token_counts.py