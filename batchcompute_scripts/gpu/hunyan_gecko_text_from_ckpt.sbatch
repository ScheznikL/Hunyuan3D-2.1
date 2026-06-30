#!/bin/bash
#
#SBATCH --job-name=text_demo_form_ckpt
#SBATCH --partition=gecko       
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      
#SBATCH --gres=gpu:1            
#SBATCH --time=1-00:00:00       
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 
#SBATCH --output=./out/text_demo_form_ckpt_joboutput_gecko%j.out 
#SBATCH --error=./out/text_demo_form_ckpt_joboutput_gecko%j.err  

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HY3DGEN_MODELS=/dcs/large/u5745134/hy3dgen_models
export HF_HOME=/dcs/large/u5745134/cache/huggingface

module load CUDA/12.6.2
cd /dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1
conda activate hunyuan3D_torch28

python ./experiments/text_demo_form_ckpt.py