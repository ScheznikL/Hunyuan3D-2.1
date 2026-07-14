#!/bin/bash
#
#SBATCH --job-name=image-set  # Job name for tracking
#SBATCH --partition=gecko       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # Number of CPU threads used by your job
#SBATCH --gres=gpu:2            # Number of GPUs to use 
#SBATCH --time=1-00:00:00       #
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=/dcs/pg24/u5745134/Desktop/dev/test_geneartion/log/image-set%j.out # Standard out from your job
#SBATCH --error=/dcs/pg24/u5745134/Desktop/dev/test_geneartion/log/image-set%j.err  # Standard error from your job
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

JSON_INPUT=/dcs/pg24/u5745134/Desktop/dev/test_geneartion/tryon_request.json
OUTPUT_DIR="/dcs/pg24/u5745134/Desktop/dev/test_geneartion/tryon_results"
SEED=43

mkdir -p "$OUTPUT_DIR"

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required but not installed."
    exit 1
fi

echo "========================================="
echo "Reading try-on config from JSON"
echo "========================================="

CONFIG_ID=$(jq -r ".config_id"                    "$JSON_INPUT")
PROMPT=$(jq -r ".prompt"                          "$JSON_INPUT")
USER_PHOTO=$(jq -r ".user_photo_url"              "$JSON_INPUT")
GENERATED_MODEL=$(jq -r ".generated_model_image_url" "$JSON_INPUT")
MATERIAL=$(jq -r ".material"                      "$JSON_INPUT")
COLOR_NAME=$(jq -r ".color_name"                  "$JSON_INPUT")
INSCRIPTION=$(jq -r ".inscription // empty"       "$JSON_INPUT")
# "/dcs/pg24/u5745134/Desktop/dev/test_geneartion/images/config-abc123.png", OLD
echo "Config ID      : $CONFIG_ID"
echo "Material       : $MATERIAL"
echo "Color          : $COLOR_NAME"
echo "Inscription    : ${INSCRIPTION:-"(none)"}"
echo "User photo     : $USER_PHOTO"
echo "Generated model: $GENERATED_MODEL"
echo "Prompt         : $PROMPT"

# ── Validate required fields ─────────────────────────────────────────────────
if [ -z "$USER_PHOTO" ]; then
    echo "ERROR: user_photo_url is required for try-on."
    exit 1
fi

if [ -z "$GENERATED_MODEL" ]; then
    echo "ERROR: generated_model_image_url is required for try-on."
    exit 1
fi

OUT_IMAGE="${OUTPUT_DIR}/${CONFIG_ID}_tryon.png"

echo "========================================="
echo "Running try-on generation"
echo "========================================="

if [ ! -f "$OUT_IMAGE" ]; then
    echo "Combining: user photo + generated model image"

    CUDA_VISIBLE_DEVICES=0 python /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/image_editor/FireRed-Image-Edit/inference.py \
        --input_image "$USER_PHOTO" "$GENERATED_MODEL" \
        --prompt "$PROMPT" \
        --output_image "$OUT_IMAGE" \
        --seed "$SEED" \
        --num_inference_steps 30

else
    echo "$OUT_IMAGE already exists, skipping."
fi

echo "========================================="
echo "Done: $OUT_IMAGE"
echo "========================================="