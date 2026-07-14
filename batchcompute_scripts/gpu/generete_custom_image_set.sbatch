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

JSON_INPUT=/dcs/pg24/u5745134/Desktop/dev/test_geneartion/request-t-shirt.json
OUTPUT_DIR="/dcs/pg24/u5745134/Desktop/dev/test_geneartion/images"
SEED=43

mkdir -p "$OUTPUT_DIR"

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required but not installed."
    exit 1
fi

echo "========================================="
echo "Reading config from JSON"
echo "========================================="

CONFIG_ID=$(jq -r ".config_id"                "$JSON_INPUT")
PROMPT=$(jq -r ".prompt"                      "$JSON_INPUT")
REFERENCE_IMAGE=$(jq -r ".reference_photo_url" "$JSON_INPUT")

TEXTURE_URL=$(jq -r ".texture_url // empty"   "$JSON_INPUT")  # empty if null
# if [ -z "$TEXTURE_URL" ]; then
#     TEXTURE_NAME=$(jq -r '.texture_name' "$JSON_INPUT")
#     PROMPT="$PROMPT Texture: $TEXTURE_NAME."
# fi

echo "Config ID : $CONFIG_ID"
echo "Prompt    : $PROMPT"
echo "Reference : $REFERENCE_IMAGE"
echo "Texture   : ${TEXTURE_URL:-"(none)"}"

OUT_IMAGE="${OUTPUT_DIR}/${CONFIG_ID}.png"

echo "========================================="
echo "Generating image"
echo "========================================="

if [ ! -f "$OUT_IMAGE" ]; then

    # ── Build --input_image args conditionally ──────────────────────────────
    INPUT_ARGS="--input_image $REFERENCE_IMAGE"

    if [ -n "$TEXTURE_URL" ]; then
        echo "Texture provided — combining reference + texture"
        INPUT_ARGS="$INPUT_ARGS $TEXTURE_URL"   # pipeline receives 2 images
    else
        echo "No texture — using reference image only"
    fi

    echo "Running generation for config $CONFIG_ID"

    CUDA_VISIBLE_DEVICES=0 python /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/image_editor/FireRed-Image-Edit/inference.py \
        $INPUT_ARGS \
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