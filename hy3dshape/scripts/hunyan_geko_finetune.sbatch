#!/bin/bash
#
#SBATCH --job-name=finetune-hun  # Job name for tracking
#SBATCH --partition=gecko       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # Number of CPU threads used by your job
#SBATCH --gres=gpu:2            # Number of GPUs to use 
#SBATCH --time=1-12:00:00       #
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./log/finetune-hun_gecko%j.out # Standard out from your job
#SBATCH --error=./log/finetune-hun_gecko%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models
export HF_HOME=/dcs/large/u5745134/cache/huggingface

module load CUDA/12.6.2
cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/hy3dshape
conda activate hunyuan3D_torch28

export CUDA_VISIBLE_DEVICES=0,1
export num_gpu_per_node=2

export node_num=1
export node_rank=0
export master_ip=0.0.0.0 # set your master_ip

export config=configs/finetunung-flowmatching-with-text.yaml
export output_dir=/dcs/large/u5745134/train_results/finetune_output_folder_1/dit_v3/
export ckpt_path=/dcs/large/u5745134/train_results/finetune_output_folder_1/dit_v3/ckpt/ckpt-step=00020000.ckpt

bash scripts/train_deepspeed_fine.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir $ckpt_path