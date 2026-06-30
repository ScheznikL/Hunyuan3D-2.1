#!/bin/bash
#
#SBATCH --job-name=100-size-test  # Job name for tracking
#SBATCH --partition=eagle       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6       # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=4:00:00       #
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./out/test_OUT_100_%j.out # Standard out from your job
#SBATCH --error=./out/test_ERROR_100_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models
export HF_HOME=/dcs/large/u5745134/cache/huggingface

module load CUDA/12.6.2
#cd /dcs/pg24/u5745134/Desktop/dev/evaluation_test/testing_Hunuyuan3d/thin
cd /dcs/pg24/u5745134/Desktop/dev/evaluation_test/testing_Hunuyuan3d

echo "========================================="
echo "Begin check 3D models"
echo "========================================="
 
#CHECKPOINT_NAME="CKPT00008000_dit_v2_mm_trained" 
#CHECKPOINT_NAME="INIT_mm_trained"
#BATCH_NAME='batch_INIT_mm_trained_100_res'
BATCH_NAME="batch_generation_results_with_depth_on_normals_less_alpa"

#OUTPUT_DIR="/dcs/pg24/u5745134/Desktop/dev/evaluation_test/testing_Hunuyuan3d/batch_${CHECKPOINT_NAME}_100_res"

OUTPUT_DIR="/dcs/pg24/u5745134/Desktop/dev/evaluation_test/testing_Hunuyuan3d/${BATCH_NAME}"

ZIP_PATH_21='/dcs/large/u5745134/batch_100/batch_CKPT00021000_dit_v2_mm_trained_100_res.zip'
ZIP_PATH_8='/dcs/large/u5745134/batch_100/batch_CKPT00008000_dit_v2_mm_trained_100_res.zip'

PROMPTS_JSON="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_TRAINING_ALIGNED.json"
PROMPTS_JSON_SHORT="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_MM_SHORT.json"
#prompts_path = "/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_MM.json"

ZIP_PATH="/dcs/large/u5745134/batch_100/${BATCH_NAME}.zip"

#SCRIPT=measure_batch_thin


conda activate hunyuan3d_eval

python ./test_sizes.py \
    --zip_path ${ZIP_PATH} \
    --output_metrics_dir ${OUTPUT_DIR} \
    --batch_name ${BATCH_NAME} \
    --prompts_json_file ${PROMPTS_JSON_SHORT} \
    --check_format 'simple_mm'

# --check_format 'training_aligned' #'training_aligned' 'detailed' # ${PROMPTS_JSON}  --prompts_json_file 'None' \  

if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Pipeline completed successfully!"
    echo "Results in: ${OUTPUT_DIR}"
    echo "========================================="
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi

#--models_dir ${OUTPUT_DIR}/100_test_models \

# Basic usage (same as before - backward compatible)
# python "./${SCRIPT}.py" \
#     --zip_path ${ZIP_PATH} \
#     --output_metrics_dir ${OUTPUT_DIR} \
#     --batch_name ${CHECKPOINT_NAME} \
#     --prompts_json_file ${PROMPTS_JSON} \
#     --check_format 'training_aligned'  # 'training_aligned' or 'detailed'

# # NEW: With normalization (reorient + uniform scaling)
# python "./${SCRIPT}.py" \
#     --zip_path ${ZIP_PATH} \
#     --output_metrics_dir ${OUTPUT_DIR} \
#     --batch_name ${CHECKPOINT_NAME} \
#     --prompts_json_file ${PROMPTS_JSON} \
#     --check_format 'training_aligned' \
#     --normalize  # Enables smart normalization

# # NEW: With custom 3D view angles
# python "./${SCRIPT}.py" \
#     --zip_path ${ZIP_PATH} \
#     --output_metrics_dir ${OUTPUT_DIR} \
#     --batch_name ${CHECKPOINT_NAME} \
#     --prompts_json_file ${PROMPTS_JSON} \
#     --check_format 'training_aligned' \
#     --view_elev 30 \
#     --view_azim 135

# NEW: All features combined
# python "./${SCRIPT}.py" \
#     --zip_path ${ZIP_PATH} \
#     --output_metrics_dir ${OUTPUT_DIR} \
#     --batch_name ${CHECKPOINT_NAME} \
#     --prompts_json_file ${PROMPTS_JSON} \
#     --check_format 'training_aligned' \
#     --normalize \
#     --view_elev 25 \
#     --view_azim 90

# if [ $? -eq 0 ]; then
#     echo "========================================="
#     echo "Pipeline completed successfully!"
#     echo "Results in: ${OUTPUT_DIR}"
#     echo "========================================="
# else
#     echo "ERROR: Evaluation failed!"
#     exit 1
# fi

