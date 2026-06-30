#!/bin/bash
#
#SBATCH --job-name=testgen  # Job name for tracking
#SBATCH --partition=gecko       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=1:00:00       #
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./out/generate_single_model%j.out # Standard out from your job
#SBATCH --error=./out/generate_single_model%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models
export HF_HOME=/dcs/large/u5745134/cache/huggingface

module load CUDA/12.6.2
cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1
conda activate hunyuan3D_torch28

set -euo pipefail

echo "Working directory: $(pwd)"
echo

JSON_INPUT_ON_I=/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_MM_NO_NUMBERS.json


IMAGE_EXAMPLE=/dcs/large/u5745134/evaluation_test/testing_Hunuyuan3d/test_images_firered_v2/3e2a998b-7032-46c0-8fe4-77cabe34e430.png
IMAGE_GRAY=NONE

BASE_UUID=$(basename "$IMAGE_EXAMPLE" .png)
OUTPUT_BASE="/dcs/large/u5745134/evaluation_test/testing_Hunuyuan3d"
OUTPUT_DIR="${OUTPUT_BASE}/one_test_example_gray"
CHECKPOINT_NAME="CKPT00008000_dit_v2_mm_trained" 

PRIOR="/dcs/large/u5745134/dataset/raw/cover_prior/coverC_to_obj_pipe.obj"

#JSON_INPUT="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompt_final_text.json"

COUNT=100

for ((i=0; i<COUNT; i++))
do
    PROMPT=$(jq -r ".[$i].prompt" "$JSON_INPUT_ON_I")
    PROMPT=${PROMPT//Ergonomic semi-cylindrical U-shaped open channel/}

    UUID=$(jq -r ".[$i].uuid" "$JSON_INPUT_ON_I")

    if [ "$BASE_UUID" = "$UUID" ]; then
        echo "Match found at index $i"
        break
    fi
done


## ---------------- STEP 1: Generate 3D ---------------- ##
echo "========================================="
echo "STEP 1: Generating 3D model"
echo "========================================="

echo "Checkpoint: $CHECKPOINT_NAME"
echo "Image:      $IMAGE_GRAY"
echo "Prompt:     $PROMPT"
echo "Prior:      $PRIOR"
echo "Output dir: $OUTPUT_DIR"
echo



python ./experiments/100/generate_single_entry.py \
    --image "$IMAGE_GRAY" \
    --prompt "$PROMPT" \
    --out_dir "$OUTPUT_DIR" \
    --prior "$PRIOR"    \
    --checkpoint_name "$CHECKPOINT_NAME"

echo
echo "========================================="
echo "SUCCESS: 3D model generated"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================="