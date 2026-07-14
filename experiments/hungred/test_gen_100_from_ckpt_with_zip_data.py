import os
import json
import time
from PIL import Image
import torch
import sys
import zipfile
from io import BytesIO

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipelineWithPriorInitialization
from hy3dshape.pipelines import export_to_trimesh
#from ..exp_prior_init_img import edit_image_nm
from pipe_utils import setup_pipeline_with_custom_weights

PRIOR_PATH = "/dcs/large/u5745134/dataset/raw/cover_prior/coverC_to_obj_pipe.obj"

# Configuration
ZIP_PATH = "/dcs/large/u5745134/batch_100/batch_generation_results.zip"  # Path to your zip file with test prompts
CHECKPOINT_NAME = "CKPT00030000"  # Change this to match your checkpoint
#CHECKPOINT_PATH = "/dcs/large/u5745134/train_results/finetune_output_folder_1/dit/ckpt/ckpt-step=00154000.ckpt"
#CHECKPOINT_PATH = "/dcs/large/u5745134/train_results/finetune_output_folder_1/dit/ckpt/ckpt-step=00014000.ckpt/out/pytorch_model.bin"
CHECKPOINT_PATH ='/dcs/large/u5745134/train_results/finetune_output_folder_1/dit/ckpt/ckpt-step=00030000.ckpt/out/pytorch_model.bin'
# Input - existing images from initial batch (also in a zip)
IMAGES_ZIP_PATH = "/dcs/large/u5745134/batch_100/batch_generation_results.zip"
IMAGES_FOLDER_IN_ZIP = "100_test_images"  # Folder name inside the zip


# Output
OUTPUT_BASE = "/dcs/pg24/u5745134/Desktop/dev/evaluation_test/testing_Hunuyuan3d"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"batch_{CHECKPOINT_NAME}_100_res")
IMAGE_OUT_DIR = os.path.join(OUTPUT_DIR, "100_test_images")
MODEL_OUT_DIR = os.path.join(OUTPUT_DIR, "100_test_models")

TEMP_RESULTS = "/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/tmp_results"

# Create output directories
for d in [IMAGE_OUT_DIR, MODEL_OUT_DIR]:
    os.makedirs(d, exist_ok=True)



def load_json_from_zip(zip_path):
    """
    Load all JSON files from zip without extracting.
    Returns list of (filename, json_data) tuples.
    """
    json_data_list = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get all JSON files in the zip
        json_files = [f for f in zip_ref.namelist() 
                     if f.endswith('_meta.json') and not f.endswith('_compressed_meta.json')]
        
        for json_file in sorted(json_files):
            # Read JSON directly from zip
            with zip_ref.open(json_file) as f:
                data = json.load(f)
                json_data_list.append((json_file, data))
    
    return json_data_list

def load_image_from_zip(images_zip_path, uuid, folder_in_zip="100_test_images"):
    """
    Load a single image from a zip file by UUID.
    
    Args:
        images_zip_path: Path to the zip file containing images
        uuid: UUID of the image to load
        folder_in_zip: Folder name inside the zip where images are stored
    
    Returns:
        PIL Image object or None if not found
    """
    # Possible paths inside zip (with or without folder prefix)
    possible_paths = [
        f"{folder_in_zip}/{uuid}.png",
        f"{uuid}.png",
        f"{folder_in_zip}/{uuid}.jpg",
        f"{uuid}.jpg"
    ]
    
    with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
        # Try each possible path
        for img_path in possible_paths:
            if img_path in zip_ref.namelist():
                # Read image directly from zip
                with zip_ref.open(img_path) as f:
                    image_data = f.read()
                    image = Image.open(BytesIO(image_data))
                    return image
    
    return None

def process_batch_from_zip(prompts_zip_path, images_zip_path, pipeline, prior_path, images_folder_in_zip):
    """
    Process all prompts from a zip file.
    """
    print(f"Loading test prompts from: {prompts_zip_path}")
    json_data_list = load_json_from_zip(prompts_zip_path)
    print(f"Found {len(json_data_list)} test prompts")
    
    for idx, (json_filename, data) in enumerate(json_data_list, 1):
        start_batch_time = time.time()
        
        # Get data from JSON - using 'prompt' NOT 'prompt_clip'
        uid = data.get('uuid')
        prompt = data.get('prompt')  # Changed from prompt_clip
        seed = data.get('seed')
        # Skip if GLB already exists
        model_path = os.path.join(MODEL_OUT_DIR, f"{uid}.glb")
        if os.path.exists(model_path):
            print(f"[SKIP] UUID {uid} already has GLB, skipping...")
            continue

        if not prompt:
            print(f"[WARNING] No 'prompt' field in {json_filename}, skipping...")
            print(f"Available keys: {data.keys()}")
            continue
        
        print(f"\n[{idx}/{len(json_data_list)}] Processing UUID: {uid}")
        print(f"Prompt: {prompt}")
        metrics = {}

         # 1. Load existing image from zip file
        try:
            image = load_image_from_zip(images_zip_path, uid, images_folder_in_zip)
            
            if image is None:
                print(f"[ERROR] Image not found for UUID {uid} in zip, skipping...")
                continue
            
            print(f"[INFO] Loaded existing image for {uid} from zip")
        except Exception as e:
            print(f"[ERROR] Failed to load image: {e}")
            continue
        
        # Copy image to output directory
        img_path = os.path.join(IMAGE_OUT_DIR, f"{uid}.png")
        image.save(img_path)

        # 2. Shape Generation (3D model)
        shape_start = time.time()
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        
        try:
            # outputs = pipeline(
            #     prompt=prompt,
            #     image=image,
            #     prior=prior_path,
            #     alpha=alpha,
            #     generator=generator,
            #     output_type='mesh',
            #     num_inference_steps=num_inference_steps,
            #     # octree_resolution=380,
            # )
            outputs = pipeline(
                image=image,
                prompt=prompt,
                prior=prior_path,
                alpha=0.7,
                generator=generator,
                output_type='mesh'
            )
            metrics['time_shape'] = round(time.time() - shape_start, 2)
        except Exception as e:
            print(f"[ERROR] Failed to generate shape: {e}")
            continue

        # 3. Export mesh
        try:
            mesh = export_to_trimesh(outputs)[0]
            model_path = os.path.join(MODEL_OUT_DIR, f"{uid}.glb")
            mesh.export(model_path)
        except Exception as e:
            print(f"[ERROR] Failed to export mesh: {e}")
            continue

        # 4. Save result JSON
        result_json = {
            "uuid": uid,
            "seed": seed,
            "prompt": prompt,
            "image_path": img_path,
            "model_path": model_path,
            "metrics": metrics,
            "checkpoint": CHECKPOINT_NAME
        }

        final_json_path = os.path.join(MODEL_OUT_DIR, f"{uid}.json")
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=4)
            
        total_time = time.time() - start_batch_time
        print(f"[DONE] UUID: {uid} | Total: {total_time:.2f}s | "
              f"T2I: {metrics.get('time_t2i', 0)}s | Shape: {metrics.get('time_shape', 0)}s")
    
    print(f"\n✅ Processing complete! Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    # Enable TF32 for faster inference
    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except ImportError:
        print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
    except Exception as e:
        print(f"Warning: Failed to apply torchvision fix: {e}")

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    print(f"Testing checkpoint: {CHECKPOINT_NAME}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Initialize pipeline with your fine-tuned checkpoint
    print("\n[INFO] Loading model...")
    config = '/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/hy3dshape/configs/finetunung-flowmatching-with-text.yaml'

    pipeline_shapegen = setup_pipeline_with_custom_weights(ckpt_path=CHECKPOINT_PATH, config_path=config)
    # # Load base pipeline
    print("[INFO] Model loaded successfully!")
    
    # Process the batch
    process_batch_from_zip(
        prompts_zip_path=ZIP_PATH,
        images_zip_path=IMAGES_ZIP_PATH,
        pipeline=pipeline_shapegen,
        prior_path=PRIOR_PATH,
        images_folder_in_zip=IMAGES_FOLDER_IN_ZIP
    )