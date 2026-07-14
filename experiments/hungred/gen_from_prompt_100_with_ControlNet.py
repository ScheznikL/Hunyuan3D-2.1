import os
import json
from pathlib import Path
import time
import torch
import shutil
import sys
from venv import logger
sys.path.insert(0, './hy3dshape')

current_dir = os.path.dirname(__file__)
experiments_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, experiments_dir)


from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipelineWithPriorInitialization
from hy3dshape.pipelines import export_to_trimesh
from exp_prior_init_img import edit_image_controlnet, edit_image_pix2pix, edit_image_nm
#from test_gen_100_from_ckpt_with_zip_data import load_image_from_zip

PRIOR_PATH = "/dcs/large/u5745134/dataset/raw/cover_prior/coverC_to_obj_pipe.obj"


def process_batch_with_uuid(json_folder_init, pipeline, prior_path, im_path = None,
                            tmp_results_renders = "/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/tmp_results",
                            image_out_dir = None,
                            model_out_dir = None,
                            alpha = 0.7,
                            prompt_prefix = None,
                            is_normals = None,
                            ):
    
    is_prompt_from_file = None
    json_data = None
    json_folder_init = Path(json_folder_init)
    
    if json_folder_init.is_dir():
        # batch mode (many meta files)
        json_files = sorted(
            f for f in os.listdir(json_folder_init)
            if f.endswith("_meta.json") and not f.endswith("_compressed_meta.json")
        )
        is_prompt_from_file = False
    else:
          # single JSON mode
        with open(json_folder_init, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        if isinstance(json_data, dict): # one obj
            json_files = [json_folder_init.name]
            json_data = [json_data]
        elif isinstance(json_data, list): 
            json_files = [f"entry_{i}" for i in range(len(json_data))]
        else:
            raise ValueError("Unsupported JSON structure")
        is_prompt_from_file = True
  
    # === image time ===
    #render_normals_blender(prior_path, norm_out_dir)

    
    # ---------- MAIN LOOP ----------
    for idx, json_file in enumerate(json_files):
        start_batch_time = time.time()

        # ----- LOAD JSON ENTRY -----
        if is_prompt_from_file:
            
            data = json_data[idx]
            prompt = data.get("prompt_original")
        else:
            json_path = os.path.join(json_folder_init, json_file)
            compressed_filename = json_file.replace("_meta.json", "_compressed_meta.json")
            json_path_compressed = os.path.join(json_folder_init, compressed_filename)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            with open(json_path_compressed, "r", encoding="utf-8") as f:
                data_clip = json.load(f)
            prompt = data_clip.get("prompt_clip") if data_clip else None
            
        # ----- EXTRACT FIELDS -----
        uid = data.get("uuid")
        seed = data.get("seed")
        if seed is None:
            seed = torch.seed()
       
        short_prompt = data.get("prompt")
        
        if not prompt or not short_prompt:
            raise KeyError(f"Missing some prompts for UUID {uid}")
        
        #----- IMAGE SFTE PROMPT ----
        t2i_start = time.time()
        print("[INFO] Getting images ...")
        if not im_path:
            norm_out_dir = os.path.join(tmp_results_renders, "normals")
            print(f"[INFO] From {norm_out_dir}")
            # image, filename = edit_image_controlnet(normal_map_dir = norm_out_dir,
            #                                         prompt=short_prompt, 
            #                                         is_normal_mode = is_normals,
            #                                         usereal=False)
            image =   edit_image_nm(normal_map_dir = norm_out_dir, prompt=short_prompt)
        else:
            print(f"[INFO] From {im_path}")
            # image, filename= edit_image_pix2pix(im_path=im_path, prompt=short_prompt, 
            #                                     usereal=True, 
            #                                     prompt_prefix=prompt_prefix)
            image, filename= edit_image_controlnet(im_path=im_path, prompt=short_prompt, usereal=True) 
            
        rembg = BackgroundRemover("isnet-general-use")
        image = rembg(image)
        time_t2i = round(time.time() - t2i_start, 2)
        img_path = os.path.join(image_out_dir, f'{uid}.png')    
        image.save(img_path)
        
        print(f"[INFO] Edited image saved to {img_path}")
        
        metrics = {"time_t2i": time_t2i}
        print(f"\n[INFO] Processing UUID: {uid} with alpha: {alpha}")

        # ---------- SHAPE GENERATION ----------
        shape_start = time.time()
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

        outputs = pipeline(
            image=image,
            prior=prior_path,
            alpha=alpha,
            generator=generator,
            output_type="mesh",
        )

        metrics["time_shape"] = round(time.time() - shape_start, 2)

        # ---------- EXPORT ----------
        mesh = export_to_trimesh(outputs)[0]
        model_path = os.path.join(model_out_dir, f"{uid}.glb")
        mesh.export(model_path)

        # ---------- RESULT JSON ----------
        result_json = {
            "uuid": uid,
            "seed": seed,
            "prompt": prompt,
            "image_path": img_path,
            "model_path": model_path,
            "metrics": metrics,
        }

        final_json_path = os.path.join(model_out_dir, f"{uid}.json")
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=4)

        print(f"[DONE] Saved UUID: {uid} | Total time: {time.time() - start_batch_time:.2f}s")
     
     
     
     
if __name__ == '__main__':
 # Enable TF32 for faster inference
    #JSON_INPUT_DIR = "./prompts_json" 
    #INIT_BASE = "/dcs/pg24/u5745134/Desktop/dev/evaluation_test/testing_Hunuyuan3d/batch_generation_results"
    
    destination_name = "batch_generation_results_with_less_alph_normals"
    output_base = f"/dcs/pg24/u5745134/Desktop/dev/evaluation_test/testing_Hunuyuan3d/{destination_name}"
    image_out_dir = os.path.join(output_base, "100_test_images")
    model_out_dir = os.path.join(output_base, "100_test_models")
    
    # IMAGE_INIT_DIR = os.path.join(INIT_BASE, "100_test_images")
    # mode_init_dir = os.path.join(INIT_BASE, "100_test_models")

    json_file_prompt = "/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/experiments/in_prompts_MM_SHORT.json"
    
    for d in [image_out_dir, model_out_dir]:
        os.makedirs(d, exist_ok=True)

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    
    # Initialize pipeline with your fine-tuned checkpoint
    print("\n[INFO] Loading model...")
    
    model_path = 'tencent/Hunyuan3D-2mini'
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipelineWithPriorInitialization.from_pretrained(
        model_path,
        subfolder='hunyuan3d-dit-v2-mini',
    )
    #prompt_prefix = "Turn given one part of prosthetic cover into same structured one "
    im_path = '/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/assets/image_prior_part.png'
    full_prior_path= "/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/assets/image_prior.png"
    alpha = 0.5
    is_normals = True
    process_batch_with_uuid(json_folder_init=json_file_prompt, pipeline=pipeline_shapegen, 
                            prior_path=PRIOR_PATH, alpha=alpha, image_out_dir=image_out_dir, model_out_dir= model_out_dir, 
                            #im_path=im_path,
                            is_normals=is_normals)