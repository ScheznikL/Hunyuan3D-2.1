#!/bin/bash
#
#SBATCH --job-name=100-eagle-test  # Job name for tracking
#SBATCH --partition=eagle       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6       # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=10:00:00       #
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./out/test_OUTPUT_100%j.out # Standard out from your job
#SBATCH --error=./out/test_ERROR_100%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models
export HF_HOME=/dcs/large/u5745134/cache/huggingface

module load CUDA/12.6.2
cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1
conda activate hunyuan3D_torch28

echo "========================================="
echo "STEP 1: Generating 3D models"
echo "========================================="

CHECKPOINT_NAME="CKPT00030000" 
#CHECKPOINT_NAME="INIT_mm_detailed"
#CHECKPOINT_NAME="INIT_mm_trained" 

#python ./experiments/test_gen_100_from_ckpt.py
python ./experiments/100/generate_batch_from_text_initial_model.py

# Check if generation succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Model generation failed!"
    exit 1
fi

#OUTPUT_DIR="/dcs/pg24/u5745134/Desktop/dev/evaluation_test/testing_Hunuyuan3d/batch_${CHECKPOINT_NAME}_100_res"
OUTPUT_BASE="/dcs/pg24/u5745134/Desktop/dev/evaluation_test/testing_Hunuyuan3d"
OUTPUT_DIR="${OUTPUT_BASE}/batch_${CHECKPOINT_NAME}_100_res"
CLIP_PROMPTS_DIR="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_MM_shortened_by_sentance.json"
CLIP_PROMPTS_DIR_TR="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_TRAINING_ALIGNED.json"

# Check if output directories exist and contain files
if [ ! -d "${OUTPUT_DIR}/100_test_models" ] || [ ! -d "${OUTPUT_DIR}/100_test_images" ]; then
    echo "ERROR: Output directories not created!"
    echo "Expected: ${OUTPUT_DIR}/100_test_models and ${OUTPUT_DIR}/100_test_images"
    exit 1
fi

# Count generated models and images
MODEL_COUNT=$(find "${OUTPUT_DIR}/100_test_models" -name "*.glb" 2>/dev/null | wc -l)
IMAGE_COUNT=$(find "${OUTPUT_DIR}/100_test_images" -name "*.png" 2>/dev/null | wc -l)

echo "Generated files:"
echo "  Models: ${MODEL_COUNT}"
echo "  Images: ${IMAGE_COUNT}"

# Abort if no models were generated
if [ ${MODEL_COUNT} -eq 0 ]; then
    echo "ERROR: No models generated! Aborting evaluation."
    exit 1
fi

if [ ${IMAGE_COUNT} -eq 0 ]; then
    echo "WARNING: No images found, but models exist. Continuing..."
fi

echo "Generation completed successfully with ${MODEL_COUNT} models."

echo "========================================="
echo "STEP 2: Running evaluation"
echo "========================================="

conda deactivate 
conda activate hunyuan3d_eval

# Run your evaluation script here
python /dcs/pg24/u5745134/Desktop/dev/evaluation_test/testing_Hunuyuan3d/evaluate_with_renders.py \
    --base_dir ${OUTPUT_DIR} \
    --models_dir ${OUTPUT_DIR}/100_test_models \
    --output_metrics ${OUTPUT_DIR}/evaluation_results.json \
    --batch_name ${CHECgenerateKPOINT_NAME} \
    #--clip_prompts_dir ${CLIP_PROMPTS_DIR_TR}

if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Pipeline completed successfully!"
    echo "Results in: ${OUTPUT_DIR}"
    echo "========================================="
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi