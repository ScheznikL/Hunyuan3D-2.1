#!/bin/bash
#
#SBATCH --job-name=one-model-gen  # Job name for tracking
#SBATCH --partition=gecko       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=1-00:00:00       #
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./out/one_model%j.out # Standard out from your job
#SBATCH --error=./out/one_model%j.err  # Standard error from your job

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

## ---------------- Config ---------------- ##
CHECKPOINT_NAME="CKPT00008000_dit_v2_mm_detailed"
INIT_MODEL_NAME="INIT_HUNYUYAN3D" # CHECK BELOW !!!!!!
JSON_INPUT="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompt_final_text.json"

OUTPUT_BASE="/dcs/large/u5745134/evaluation_test/testing_Hunuyuan3d"
OUTPUT_DIR="${OUTPUT_BASE}/one_model_${INIT_MODEL_NAME}_gen"

#INITIAL_IMAGE="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/assets/prior_imgs/img-1-min.png"
INITIAL_IMAGE=/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/assets/prior_imgs/img-3-min.png
IMAGE="${OUTPUT_DIR}/image_from_t2i.png"

PRIOR="/dcs/large/u5745134/dataset/raw/cover_prior/coverC_to_obj_pipe.obj"
SEED=43

mkdir -p "$OUTPUT_DIR"

## ---------------- Read JSON ---------------- ##
echo "========================================="
echo "Reading prompts from JSON"
echo "========================================="

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required but not installed."
    exit 1
fi
JSON_INPUT="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompt_final_text.json"

PROMPT_T2I=$(jq -r '.prompt_t2i' "$JSON_INPUT")
PROMPT_I23D=$(jq -r '.prompt' "$JSON_INPUT")

echo "T2I prompt:   $PROMPT_T2I"
echo "I2→3D prompt: $PROMPT_I23D"
echo

## ---------------- STEP 0: Generate Image ---------------- ##
echo "========================================="
echo "STEP 0: Generating image"
echo "========================================="

if [ ! -f "$IMAGE" ]; then
    python /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/image_editor/FireRed-Image-Edit/inference.py \
        --input_image "$INITIAL_IMAGE" \
        --prompt "$PROMPT_T2I" \
        --output_image "$IMAGE" \
        --seed "$SEED" \
        --num_inference_steps 30 \
        --optimized True  # NEW
    echo "Image generated: $IMAGE"
    echo

else
    echo "$IMAGE already exists"
fi



## ---------------- STEP 1: Generate 3D ---------------- ##
echo "========================================="
echo "STEP 1: Generating 3D model"
echo "========================================="

echo "Checkpoint: $INIT_MODEL_NAME"
echo "Image:      $IMAGE"
echo "Prompt:     $PROMPT_I23D"
echo "Prior:      $PRIOR"
echo "Output dir: $OUTPUT_DIR"
echo

python ./experiments/100/generate_single_entry.py \
    --image "$IMAGE" \
    --prompt "$PROMPT_I23D" \
    --out_dir "$OUTPUT_DIR" \
    --prior "$PRIOR"    \
    --checkpoint_name "$INIT_MODEL_NAME"

echo
echo "========================================="
echo "SUCCESS: 3D model generated"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================="