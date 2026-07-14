import uuid
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
sys.path.insert(0, './hy3dpaint')
# Add the parent directory of the current script (experiments/)
current_dir = os.path.dirname(__file__)
experiments_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, experiments_dir)

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, export_to_trimesh, HunyuanDiTPipeline
from custom_utils import prepare_directories
#from test_gen_100_from_ckpt_with_zip_data import load_image_from_zip

from textureGenPipeline import Hunyuan3DPaintPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

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
def generate_single_entry_no_image(
                          shape_pipeline,                            
                          iteration_idx,
                          unique_id, 
                          num_models,
                          out_models_dir,
                          prompt = None,
                          prior_path = None,
                          alpha = None,
                          seed = None, 
                          ):
    """
    Generates one complete entry: prompt -> Image -> Background Removal -> 3D Mesh.
    Saves files with UUID.
    """
        
        
    logger.info(f"[{iteration_idx}/{num_models}] Processing UUID: {unique_id} | Seed: {seed}")
    model_filename = f"no_image_{unique_id}.glb"
    model_path = os.path.join(out_models_dir, model_filename)
    meta_path = os.path.join(out_models_dir, f"{unique_id}_meta.json")
    
    if not os.path.exists(model_path) and not os.path.exists(meta_path):  
            
        generator = torch.Generator().manual_seed(seed)
        try:
                
            start_t = time.time()
            outputs = shape_pipeline(
            prompt=prompt,
            prior=prior_path,
            alpha=alpha,
            generator=generator,
            output_type='mesh'
            )

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

    metadata = {
        "uuid": unique_id,
        "seed": seed,
        "prompt": prompt,
        "model_path": model_path,
        "metrics": {
            "time_shape": round(shape_time, 2)
        }
    }
    
    # Save per-model metadata json
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def generate_single_entry(shape_pipeline,                            
                          iteration_idx,
                          unique_id, 
                          num_models,
                          out_models_dir,
                          mode: Modes,
                          prompt = None,
                          prior_path = None,
                          alpha = None,
                          seed = None, 
                          raw_image = None,
                          refine = None,
                          paint_pipeline = None,
                          im_zip_dir =None, remove_background = True, object_name = None
                          ):
    """
    Generates one complete entry: prompt -> Image -> Background Removal -> 3D Mesh.
    Saves files with UUID.
    """
        
        
    logger.info(f"[{iteration_idx}/{num_models}] Processing UUID: {unique_id} | Seed: {seed}")
     # 6. Save Model
    model_filename = f"{object_name}.glb" if object_name is not None else f"{unique_id}.glb"
    model_path = os.path.join(out_models_dir, model_filename)
    meta_path = os.path.join(out_models_dir, f"{object_name}_meta.json" if object_name is not None else f"{unique_id}_meta.json" )
    
    
    if not os.path.exists(model_path) and not os.path.exists(meta_path):

        raw_image = Image.open(raw_image) if raw_image is not None else raw_image
         
        if remove_background and raw_image is not None and raw_image.mode != "RGBA":
            rembg = BackgroundRemover()
            clean_image = rembg(raw_image.convert("RGB"))
        else:
            clean_image = raw_image       
            
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
            
            if refine and paint_pipeline:
                mesh_textured = paint_pipeline(model_path, image_path=image)
                mesh_textured.export(model_path)

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
        "model_path": model_path,
        "metrics": {
            "time_shape": round(shape_time, 2)
        }
    }
    
    # Save per-model metadata json
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str,)
    parser.add_argument("--prompt", type=str,)
    parser.add_argument("--out_dir", required=True, type=str,)
    parser.add_argument("--prior", type=str,)
    parser.add_argument("--checkpoint_name", required=True, type=str,)
    parser.add_argument("--refine", action="store_true")
    #prior = "/dcs/large/u5745134/dataset/raw/cover_prior/coverC_to_obj_pipe.obj"
    #OUTPUT_BASE = "/dcs/large/u5745134/evaluation_test/testing_Hunuyuan3d"
    return parser.parse_args()
    
    
if __name__ == '__main__':
    from pipe_utils import setup_pipeline_with_custom_weights
    
    args = get_args()
    
    image = args.image if args.image != "NONE" else None
    prompt = args.prompt if args.image != "NONE" else None
    out_dir= args.out_dir
    prior = args.prior if args.image != "NONE" else None
    refine = args.refine if args.refine != "NONE" else None
    num_inference_steps = 50  
    
    mode = Modes.ImageAligned # FIXME
    seed=random.randint(1000, 999999)   

    #ckpt = "CKPT00008000" # FIXME
    config = '/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/hy3dshape/configs/finetunung-flowmatching-with-text.yaml'

    #ckpt_name = "INIT_mm_detailed"  #CKPT00030000_mm # TODO version 2 with trained 
    #ckpt_name = "INIT_mm_trained"  #CKPT00030000_mm # TODO version 2 with trained 
    #ckpt_name = f"{ckpt}_dit_v2_mm_detailed" # TODO version 2 with trained 

    ckpt_name = args.checkpoint_name # FIXME
    
    ckpt_path = "/dcs/large/u5745134/train_results/finetune_output_folder_1/dit/ckpt/ckpt-step=00030000.ckpt/out/pytorch_model.bin"
    ckpt_path_dit_v2_21 = "/dcs/large/u5745134/train_results/finetune_output_folder_1/dit_v2/ckpt/ckpt-step=00021000.ckpt/out/pytorch_model.bin"
    ckpt_path_dit_v2_8 = "/dcs/large/u5745134/train_results/finetune_output_folder_1/dit_v2/ckpt/ckpt-step=00008000.ckpt/out/pytorch_model.bin"

    unique_id = str(uuid.uuid4())
    
    prepare_directories(out_dir)
     
    if "CKPT" in ckpt_name:        
        unique_id =f"from_ckpt_{unique_id}"
        shape_pipe= setup_pipeline_with_custom_weights(
            ckpt_path= ckpt_path_dit_v2_8,
            config_path= config,
        )
    elif not refine:
        unique_id =f"from_Hunyuan3D_{unique_id}"
        shape_model_path = 'tencent/Hunyuan3D-2mini'
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            shape_model_path,
            subfolder='hunyuan3d-dit-v2-mini'
        )
    else:
        unique_id =f"refined_from_Hunyuan3D_{unique_id}"
        
        paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
               
        shape_model_path = 'tencent/Hunyuan3D-2'
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            shape_model_path,
            subfolder='hunyuan3d-dit-v2-0'
        )

        
    if mode == Modes.ImageAligned:
        generate_single_entry(
                shape_pipeline=shape_pipe, 
                iteration_idx=1,
                unique_id=unique_id,
                seed=seed,
                num_models=1,
                mode=Modes.ImageAligned,  #TODO TExtIm
                raw_image=image,
                out_models_dir=out_dir,
                remove_background = True,
                refine = refine,
                paint_pipeline=paint_pipeline)
        
    if mode == Modes.TextImageAligned:                         
            generate_single_entry(
                shape_pipeline=shape_pipe, 
                prompt=prompt,
                prior_path=prior,
                alpha=0.5, 
                iteration_idx=1,
                unique_id=unique_id,
                seed=seed,
                num_models=1,
                mode=Modes.TextImageAligned, 
                raw_image=image,
                out_models_dir=out_dir,
                remove_background = True)
    if mode == Modes.TextAligned:                         
        generate_single_entry_no_image(
            shape_pipeline=shape_pipe, 
            prompt=prompt,
            prior_path=prior,
            alpha=0.5, 
            iteration_idx=1,
            unique_id=unique_id,
            seed=seed,
            num_models=1,
            mode=Modes.TextImageAligned, 
            out_models_dir=out_dir,
            remove_background = True)