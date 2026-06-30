#!/bin/bash
#
#SBATCH --job-name=hunuayn-cpu  # Job name for tracking
#SBATCH --partition=gecko       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # Number of CPU threads used by your job
#SBATCH --time=2-00:00:00       # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## conda activate /dcs/large/u5745134/hunyuanenv
## conda activate /dcs/pg24/u5745134/.conda/envs/hunyancondaenv
## pip install -r /tmp/old_freeze.txt

conda activate hunyan3d_clean

## conda install pytorch-cpu torchvision-cpu -c pytorch

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models
mkdir -p $HY3DGEN_MODELS

export PIP_CACHE_DIR=/dcs/large/u5745134/cache/pip
export HF_HOME=/dcs/large/u5745134/cache/huggingface

mkdir -p $PIP_CACHE_DIR $XDG_CACHE_HOME/hub

cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1

python -m democpuH2-1.py