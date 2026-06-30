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

JSON_INPUT=/dcs/pg24/u5745134/Desktop/dev/test_geneartion/prompt.json # /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_MM_NO_NUMBERS.json

OUTPUT_DIR="/dcs/pg24/u5745134/Desktop/dev/test_geneartion/images"

INITIAL_IMAGE=https://techneek.ie/cdn/shop/products/76.png

SEED=43

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "Reading prompts from JSON"
echo "========================================="

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required but not installed."
    exit 1
fi

COUNT=$(jq length "$JSON_INPUT")

echo "Found $COUNT prompts"

echo "========================================="
echo "Generating images"
echo "========================================="

for ((i=0; i<COUNT; i++))
do
    PROMPT_T2I=$(jq -r ".[$i].prompt" "$JSON_INPUT")
    PROMPT_T2I=${PROMPT_T2I//Ergonomic semi-cylindrical U-shaped open channel/}

    UUID=$(jq -r ".[$i].uuid" "$JSON_INPUT")

    OUT_IMAGE="${OUTPUT_DIR}/${UUID}.png"

    if [ ! -f "$OUT_IMAGE" ]; then

        GPU=$((i % 2))

        echo "Running prompt $i on GPU $GPU"

        CUDA_VISIBLE_DEVICES=$GPU python /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/image_editor/FireRed-Image-Edit/inference.py \
            --input_image "$INITIAL_IMAGE" \
            --prompt "Edit the shape of cover to match $PROMPT_T2I" \
            --output_image "$OUT_IMAGE" \
            --seed "$SEED" \
            --num_inference_steps 30 &

        if (( i % 2 == 1 )); then
            wait
        fi

    else
        echo "$OUT_IMAGE already exists"
    fi
done

wait
echo "All images generated."

# for ((i=0; i<COUNT; i++))
# do
#     PROMPT_T2I=$(jq -r ".[$i].prompt" "$JSON_INPUT")
#     UUID=$(jq -r ".[$i].uuid" "$JSON_INPUT")

#     OUT_IMAGE="${OUTPUT_DIR}/${UUID}.png"

#     if [ ! -f "$OUT_IMAGE" ]; then
#         GPU=$((i % 2))

#         CUDA_VISIBLE_DEVICES=$GPU python inference.py \
#             --input_image "$INITIAL_IMAGE" \
#             --prompt "Edit the shape of cover to match $PROMPT_T2I" \
#             --output_image "$OUT_IMAGE" \
#             --seed "$SEED" \
#             --num_inference_steps 30 \
#             --optimized True &
        
#         if (( i % 2 == 1 )); then
#             wait
#         fi
            
#         # python /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/image_editor/FireRed-Image-Edit/inference.py \
#         #     --input_image "$INITIAL_IMAGE" \
#         #     --prompt "Edit the shape of cover to match $PROMPT_T2I" \
#         #     --output_image "$OUT_IMAGE" \
#         #     --seed "$SEED" \
#         #     --num_inference_steps 30 \
#         #     --optimized True

#         echo "Image generated: $OUT_IMAGE"
#         echo

#     else
#         echo "$OUT_IMAGE already exists"
#     fi
# done