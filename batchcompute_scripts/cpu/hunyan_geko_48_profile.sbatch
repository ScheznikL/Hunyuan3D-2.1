#!/bin/bash
#
#SBATCH --job-name=hunuayn-cpu-numa  # Job name for tracking
#SBATCH --partition=gecko       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48      # Number of CPU threads used by your job
#SBATCH --time=1-00:00:00       # Job time limit set to 1 days (24 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=TPOS_joboutput_48_n%j.out # Standard out from your job
#SBATCH --error=TPOS_joboutput_48_n%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

conda activate hunyan3d_clean

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models
mkdir -p $HY3DGEN_MODELS

export PIP_CACHE_DIR=/dcs/large/u5745134/cache/pip
export HF_HOME=/dcs/large/u5745134/cache/huggingface

mkdir -p $PIP_CACHE_DIR $XDG_CACHE_HOME/hub

cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1

#perf stat -e \
#  cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses \
#  python -m democpuH2-1.py

#perf record -g \
#  -e cycles \
#  python ./batchcompute_scripts/cpu/demotpos.py

##sudo perf report


### numactl --cpunodebind=0 --membind=0 python -m democpuH2-1.py