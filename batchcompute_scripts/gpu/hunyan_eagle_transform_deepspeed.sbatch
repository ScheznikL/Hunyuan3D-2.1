#!/bin/bash
#
#SBATCH --job-name=transform  # Job name for tracking
#SBATCH --partition=eagle       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6       # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=1-00:00:00       #
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./out/transform_joboutput_eagle%j.out # Standard out from your job
#SBATCH --error=./out/transform_joboutput_eagle%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models
export HF_HOME=/dcs/large/u5745134/cache/huggingface

module load CUDA/12.6.2
cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1
conda activate hunyuan3D_torch28

set -euo pipefail

BASE="/dcs/large/u5745134/train_results/finetune_output_folder_1/dit_v3/ckpt"
SCRIPT="zero_to_fp32.py"

for CKPT in "$BASE"/ckpt-step=*.ckpt; do
    # skip if not a directory
    [ -d "$CKPT" ] || continue

    OUT_DIR="$CKPT/out"

    # skip if already converted (out dir exists and not empty)
    if [ -d "$OUT_DIR" ] && [ "$(ls -A "$OUT_DIR")" ]; then
        echo "Skipping (already converted): $CKPT"
        continue
    fi

    echo "Converting: $CKPT"
    (
        cd "$CKPT"
        python3 "$SCRIPT" ./ ./out --max_shard_size 30GB
    )
done

echo "Done."

# #if DeepSpeed
# cd /dcs/large/u5745134/train_results/finetune_output_folder_1/dit_v2/ckpt/ckpt-step=00008000.ckpt
# python3 zero_to_fp32.py ./ ./out --max_shard_size 30GB
# then you can get output_folder/dit/overfitting_depth_16_token_4096_lr1e4/ckpt/ckpt-step=00030000.ckpt/out/pytorch_model.bin

# cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1
# python ./experiments/text_demo_form_ckpt.py