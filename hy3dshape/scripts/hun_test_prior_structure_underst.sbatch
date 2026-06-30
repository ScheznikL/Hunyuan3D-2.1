#!/bin/bash
#
#SBATCH --job-name=prior-test  # Job name for tracking
#SBATCH --partition=falcon       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=12:00:00       # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./out/prior_gpu%j.out # Standard out from your job
#SBATCH --error=./out/prior_gpu%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models
export HF_HOME=/dcs/large/u5745134/cache/huggingface

module load CUDA/12.6.2
cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1
conda activate hunyuan3D_torch28
#"/dcs/large/u5745134/dataset/raw/cover_prior/coverC_to_obj_pipe.obj"


# On SLURM headless — define splits manually after inspecting the mesh:
python /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/hy3dshape/scripts/interactive_part_selector.py \
    --mesh "/dcs/large/u5745134/dataset/raw/cover_prior/round_prior.obj" --mode headless \
    --slices 0.25 0.55 0.80 \
    --out ./part_outputs \
    --encode   # ← also runs ShapeVAE encode per part

# python /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/hy3dshape/scripts/test_part_visualization.py \
#     --mesh "/dcs/large/u5745134/dataset/raw/cover_prior/coverC_to_obj_pipe.obj" \
#     --out  ./part_vis_outputs \
#     --chunk 64