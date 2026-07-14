import argparse
import glob
import os
import sys
import time
import json
import random
import logging
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForText2Image
from enum import Enum



# --- Local Imports (Adjust paths if needed) ---
sys.path.insert(0, './hy3dshape')
#sys.path.insert(0, './hy3dpaint')
# Add the parent directory of the current script (experiments/)
current_dir = os.path.dirname(__file__)
experiments_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, experiments_dir)

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, export_to_trimesh, HunyuanDiTPipeline
from custom_utils import prepare_directories
from test_gen_100_from_ckpt_with_zip_data import load_image_from_zip

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_prompts_data(input_file):        
        # 1. Load the data  
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    with open(input_file, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    return prompts_data

def get_seed(path, uuid):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        raise ValueError("No path")
    
    json_file = os.path.join(path,f"{uuid}_meta.json")
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
    else:
        json_file = os.path.join(path,f"{uuid}.json")
    
    if not os.path.exists(json_file):
        raise ValueError(f"Error: {json_file} not found.")
        
    with open(json_file, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d['seed']    

# class syntax
class Modes(Enum):
    TextAligned = 1
    ImageAligned = 2
    TextImageAligned = 3
    PriorInitImageAligned = 4 
    
def generate_single_entry_(shape_pipeline, 
                          prompt, 
                          iteration_idx,
                          unique_id, 
                          num_models,
                          models_dir,
                          mode: Modes,
                          prior_path = None,
                          alpha = None,
                          seed = None, 
                          t2i_pipeline = None, raw_image = None, imref_dir = None, object_name = None):
    """
    Generates one complete entry: prompt -> Image -> Background Removal -> 3D Mesh.
    Saves files with UUID.
    """
        
        
    logger.info(f"[{iteration_idx}/{num_models}] Processing UUID: {unique_id} | Seed: {seed}")
     # 6. Save Model
    model_filename = f"{object_name}.glb" if object_name is not None else f"{unique_id}.glb"
    model_path = os.path.join(models_dir, model_filename)
    meta_path = os.path.join(models_dir, f"{unique_id}_meta.json")
    
    if not os.path.exists(model_path) and not os.path.exists(meta_path):
        # 2. Text-to-Image Generation        
        # 4. Save Image (Reference)
        image_filename = f"{unique_id}.png"
        image_path = os.path.join(imref_dir, image_filename)
        if not os.path.exists(image_path):
        # 2. Text-to-Image Generation
            if t2i_pipeline is not None:
                try:
                    start_t = time.time()
                    raw_image = t2i_pipeline(prompt, seed=seed)
                    t2i_time = time.time() - start_t
                except Exception as e:
                    logger.error(f"Failed T2I generation: {e}")
                    return
            elif raw_image is None: 
                raw_image = load_image_from_zip(IM_ZIP, uuid = unique_id)
                if raw_image is None:
                    raise ValueError(f"raw_image is {raw_image}")

            # 3. Background Removal
            if raw_image.mode != "RGBA":
                rembg = BackgroundRemover()
                clean_image = rembg(raw_image.convert("RGB"))
            else:
                clean_image = raw_image
            clean_image.save(image_path)
            
        else:
            clean_image = image_path
            print(f"[INFO] skipping image generation for {unique_id}")
            
        generator = torch.Generator().manual_seed(seed)
        try:
            
            match mode:
                case Modes.TextAligned:
                    start_t = time.time()
                    # TODO
                    raise NotImplementedError
                        
                case Modes.ImageAligned:
                    start_t = time.time()
                    
                    outputs = shape_pipeline(
                        image=clean_image,
                        generator=generator,
                        output_type='mesh'
                        )
                    
                case Modes.TextImageAligned:
                    start_t = time.time()
                    if prior_path and alpha is not None:
                        outputs = shape_pipeline(
                        image=clean_image,
                        prompt=prompt,
                        prior=prior_path,
                        alpha=alpha,
                        generator=generator,
                        output_type='mesh'
                        )
                    else:
                        logger.info("Proceeding withour Prior")
                        outputs = shape_pipeline(
                        image=clean_image,
                        prompt=prompt,
                        generator=generator,
                        output_type='mesh'
                        )
                        
                case Modes.PriorInitImageAligned:
                    start_t = time.time()
                    if prior_path and alpha is not None:
                        outputs = shape_pipeline(
                        image=clean_image,
                        prompt=prompt,
                        prior=prior_path,
                        alpha=alpha,
                        generator=generator,
                        output_type='mesh'
                        )
                    else:
                        raise ValueError("No prior or alpha")
                case _:
                    raise ValueError("Error mode is {mode}")   
            
            # Export logic
            mesh = export_to_trimesh(outputs)[0]
            shape_time = time.time() - start_t
            
            mesh.export(model_path)
        except Exception as e:
            logger.error(f"Failed 3D generation for {unique_id}: {e}")
            return
    else:
        logger.info(f"{model_path} and its metadata exist, continue further")
        return
    # 7. Save Metadata for this specific run
    metadata = {
        "uuid": unique_id,
        "seed": seed,
        "prompt": prompt,
        "image_path": image_path,
        "model_path": model_path,
        "metrics": {
            "time_t2i": round(t2i_time, 2) if t2i_pipeline is not None else 0,
            "time_shape": round(shape_time, 2)
        }
    }
    
    # Save per-model metadata json
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dir", required=True, type=str,)
    return parser.parse_args()
    
    
if __name__ == '__main__':
    
    args = get_args()
    prompts_path_init = "/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/input_prompts.json" #"/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_MM.json"
    prompt_training = "/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_TRAINING_ALIGNED.json"
    prompts_path  = "/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_MM.json"
    
    if args.prompt_dir is not None:
        prompts_path = args.prompt_dir
    else:
        print(f"[INFO prompt path] Using {prompts_path}")
    
    from pipe_utils import setup_pipeline_with_custom_weights

    INITIAL_GENERATION = True 
    ckpt = "CKPT00008000"
    config = '/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/hy3dshape/configs/finetunung-flowmatching-with-text.yaml'

   # ckpt_name = "INIT_mm_detailed"  #CKPT00030000_mm # TODO version 2 with trained 
    #ckpt_name = "INIT_mm_trained"  #CKPT00030000_mm # TODO version 2 with trained 
    ckpt_name = f"{ckpt}_dit_v2_mm_detailed" # TODO version 2 with trained 
    
    num_models = 100
    
    ckpt_path = "/dcs/large/u5745134/train_results/finetune_output_folder_1/dit/ckpt/ckpt-step=00030000.ckpt/out/pytorch_model.bin"
    ckpt_path_dit_v2_21 = "/dcs/large/u5745134/train_results/finetune_output_folder_1/dit_v2/ckpt/ckpt-step=00021000.ckpt/out/pytorch_model.bin"
    ckpt_path_dit_v2_8 = "/dcs/large/u5745134/train_results/finetune_output_folder_1/dit_v2/ckpt/ckpt-step=00008000.ckpt/out/pytorch_model.bin"
    
    OUTPUT_BASE = "/dcs/large/u5745134/evaluation_test/testing_Hunuyuan3d"
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"batch_{ckpt_name}_100_res")
    IM_ZIP = "/dcs/large/u5745134/batch_100/batch_generation_results.zip" #TODO set to params
    
    models_dir = os.path.join(OUTPUT_DIR, "100_test_models")
    images_dir = os.path.join(OUTPUT_DIR, "100_test_images")
    prior = "/dcs/large/u5745134/dataset/raw/cover_prior/coverC_to_obj_pipe.obj"
    

    num_inference_steps = 50
    # 1. Compatibility Fixes
    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except ImportError:
        logger.warning("torchvision_fix module not found.")

    # 2. Initialize 3D Pipeline
    logger.info("Loading Image-to-3D Pipeline...")
    if "CKPT" in ckpt_name:
        shape_pipe= setup_pipeline_with_custom_weights(
            ckpt_path= ckpt_path_dit_v2_8,
            config_path= config,
        )
    else:
        shape_model_path = 'tencent/Hunyuan3D-2mini'
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            shape_model_path,
            subfolder='hunyuan3d-dit-v2-mini'
        )
        
    if INITIAL_GENERATION:
        train_like_prompt = False
        mode=Modes.TextImageAligned
        
        prepare_directories(models_dir, images_dir)
        prompts_data = get_prompts_data(prompts_path) #LOOK ❗️❗️❗️
        
        for i in range(num_models):
            if i < len(prompts_data):
                item = prompts_data[i]
                if not train_like_prompt and mode == Modes.ImageAligned:
                    generate_single_entry_(shape_pipeline=shape_pipe, 
                                        prompt=item['prompt'],
                                        iteration_idx=i+1,
                                        unique_id=item['uuid'],
                                        seed=random.randint(1000, 999999),
                                        num_models=num_models,
                                        mode=Modes.ImageAligned,  #TODO TExtIm
                                        imref_dir=images_dir,
                                        models_dir=models_dir)
                if train_like_prompt or mode == Modes.TextImageAligned:
                    # first_model_run_json = "batch_INIT_mm_detailed_100_res"
                    # first_model_run_json  = "/dcs/pg24/u5745134/Desktop/dev/evaluation_test/testing_Hunuyuan3d/batch_INIT_mm_detailed_100_res"
                    #seed = get_seed(os.path.join(first_model_run_json, "100_test_models"), item['uuid'])
                    # t2i_pipe = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
                   
                    seed=random.randint(1000, 999999)                   
                    generate_single_entry(
                                    shape_pipeline=shape_pipe, 
                                    prompt=item['prompt'],
                                    prior_path=prior,
                                    alpha=0.5, 
                                    iteration_idx=i+1,
                                    unique_id=item['uuid'],
                                    seed=seed,
                                    num_models=num_models,
                                    mode=Modes.TextImageAligned, 
                                    imref_dir=images_dir,
                                    models_dir=models_dir)

    logger.info("Processing complete.")