#!/bin/bash
#
#SBATCH --job-name=generate3d  # Job name for tracking
#SBATCH --partition=eagle       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6       # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=10:00:00       #
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=/dcs/pg24/u5745134/Desktop/dev/test_geneartion/log/3d-set%j.out # Standard out from your job
#SBATCH --error=/dcs/pg24/u5745134/Desktop/dev/test_geneartion/log/3d-set%j.err  # Standard error from your job
#!/bin/bash

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models
export HF_HOME=/dcs/large/u5745134/cache/huggingface

module load CUDA/12.6.2
cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1
conda activate hunyuan3D_torch28

pip install realesrgan==0.3.0

set -euo pipefail

echo "Working directory: $(pwd)"
echo


IMAGE=/dcs/pg24/u5745134/Desktop/dev/test_geneartion/images/config-abc123.png

OUTPUT_BASE="/dcs/pg24/u5745134/Desktop/dev/test_geneartion/adapted" 
OUTPUT_DIR="${OUTPUT_BASE}"
CHECKPOINT_NAME="INIT" 

#PRIOR="/dcs/large/u5745134/dataset/raw/cover_prior/coverC_to_obj_pipe.obj"


## ---------------- STEP 1: Generate 3D ---------------- ##
echo "========================================="
echo "STEP 1: Generating 3D model (Image Only)"
echo "========================================="

echo "Checkpoint: $CHECKPOINT_NAME"
echo "Image:      $IMAGE"
echo "Output dir: $OUTPUT_DIR"
echo

# Removed the --prompt flag entirely for image-only generation
python /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/hungred/generate_single_entry.py \
    --image "$IMAGE" \
    --out_dir "$OUTPUT_DIR" \
    --checkpoint_name "$CHECKPOINT_NAME"  \
    --refine \

echo
echo "========================================="
echo "SUCCESS: 3D model generated"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================="