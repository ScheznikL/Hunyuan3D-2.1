#!/bin/bash
#
#SBATCH --job-name=batch-mm-ckpt-prompts  # Job name for tracking
#SBATCH --partition=falcon       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=12:00:00       # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./out/batch_gpu%j.out # Standard out from your job
#SBATCH --error=./out/batch_gpu%j.err  # Standard error from your job

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

CHECKPOINT_NAME="CKPT00008000_dit_v2_mm_detailed" 

CLIP_PROMPTS_DIR="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_MM_SHORT.json"
ORIG_PROMPTS="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/input_prompts.json"
PROMPTS_MM="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_MM.json"
CLIP_PROMPTS_DIR_TR="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_TRAINING_ALIGNED.json"
CLIP_PROMPTS_CHOP="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_MM_AUTO_CHOPPED.json"

#CHECKPOINT_NAME="batch_generation_results_with_less_alph_normals"
#CHECKPOINT_NAME="INIT_mm_trained" 

#python ./experiments/test_gen_100_from_ckpt.py
# python ./experiments/100/generate_batch_from_text_initial_model.py \
#     --prompt_dir ${PROMPTS_MM}

#python ./experiments/100/gen_from_prompt_100_with_ControlNet.py

# Check if generation succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Model generation failed!"
    exit 1
fi

OUTPUT_BASE="/dcs/large/u5745134/evaluation_test/testing_Hunuyuan3d/"

#OUTPUT_DIR="${OUTPUT_BASE}/${CHECKPOINT_NAME}"

OUTPUT_DIR="${OUTPUT_BASE}/batch_${CHECKPOINT_NAME}_100_res"

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
python /dcs/large/u5745134/evaluation_test/testing_Hunuyuan3d/evaluate_with_renders.py \
    --base_dir ${OUTPUT_DIR} \
    --models_dir ${OUTPUT_DIR}/100_test_models \
    --output_metrics ${OUTPUT_DIR}/evaluation_results.json \
    --batch_name ${CHECKPOINT_NAME} \
    --clip_prompts_dir ${CLIP_PROMPTS_CHOP}

if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Pipeline completed successfully!"
    echo "Results in: ${OUTPUT_DIR}"
    echo "========================================="
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi

echo "========================================="
echo "STEP 3: Cheking sizes"
echo "========================================="

# ---- CHeking sizes right away
ZIP_PATH="/dcs/large/u5745134/batch_100/batch_${CHECKPOINT_NAME}_100_res.zip"


cd "${OUTPUT_DIR}"

rm -rf "./_renders"
zip -r "${ZIP_PATH}" .


python /dcs/large/u5745134/evaluation_test/testing_Hunuyuan3d/test_sizes.py \
    --zip_path ${ZIP_PATH} \
    --output_metrics_dir ${OUTPUT_DIR} \
    --batch_name ${CHECKPOINT_NAME} \
    --prompts_json_file ${PROMPTS_MM} \
    --check_format 'simple_mm'

if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Pipeline completed successfully!"
    echo "Results in: ${OUTPUT_DIR}"
    echo "========================================="
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi


#---ANOTHER


echo "========================================="
echo "STEP 2.1: Evaluation 2"
echo "========================================="

ANOTHER_CH_DIR="/dcs/large/u5745134/evaluation_test/testing_Hunuyuan3d/batch_CKPT00008000_dit_v2_cm_detailed_100_res"


# Run your evaluation script here
python /dcs/large/u5745134/evaluation_test/testing_Hunuyuan3d/evaluate_with_renders.py \
    --base_dir ${ANOTHER_CH_DIR} \
    --models_dir ${ANOTHER_CH_DIR}/100_test_models \
    --output_metrics ${ANOTHER_CH_DIR}/evaluation_results.json \
    --batch_name "batch_CKPT00008000_dit_v2_cm_detailed_100_res" \
    --clip_prompts_dir ${CLIP_PROMPTS_CHOP}

if [ $? -eq 0 ]; then
    echo "========================================="
    echo "ANOTHER Pipeline completed successfully!"
    echo "Results in: ${ANOTHER_CH_DIR}"
    echo "========================================="
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi

echo "========================================="
echo "STEP 3.1: Cheking sizes"
echo "========================================="

cd "${ANOTHER_CH_DIR}"
ZIP_PATH_2="/dcs/large/u5745134/batch_100/batch_CKPT00008000_dit_v2_cm_detailed_100_res.zip"

#rm -rf "./_renders"
zip -r "${ZIP_PATH_2}" .

python /dcs/large/u5745134/evaluation_test/testing_Hunuyuan3d/test_sizes.py \
    --zip_path ${ZIP_PATH_2} \
    --output_metrics_dir ${ANOTHER_CH_DIR} \
    --batch_name "CKPT00008000_dit_v2_cm_detailed" \
    --prompts_json_file ${PROMPTS_MM} \
    --check_format 'simple_mm'

if [ $? -eq 0 ]; then
    echo "========================================="
    echo "ANOTHER Pipeline completed successfully!"
    echo "Results in: ${ANOTHER_CH_DIR}"
    echo "========================================="
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi
