# -*- coding: utf-8 -*-

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import os
import io
import sys
import time
import random
import traceback
from typing import Optional, Union, List, Tuple, Dict

import json
import glob
import cv2
import numpy as np
import trimesh

import unicodedata
import re


import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_info

from .utils import worker_init_fn, pytorch_worker_seed, make_seed


class ResampledShards(torch.utils.data.dataset.IterableDataset):
    def __init__(self, datalist, nshards=sys.maxsize, worker_seed=None, deterministic=False):
        super().__init__()
        self.datalist = datalist
        self.nshards = nshards
        # If no worker_seed provided, use pytorch_worker_seed function; else use given seed
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = -1

    def __iter__(self):
        self.epoch += 1
        if self.deterministic:
            seed = make_seed(self.worker_seed(), self.epoch)
        else:
            seed = make_seed(self.worker_seed(), self.epoch, 
                             os.getpid(), time.time_ns(), os.urandom(4))
        self.rng = random.Random(seed)
        for _ in range(self.nshards):
            index = self.rng.randint(0, len(self.datalist) - 1)
            yield self.datalist[index]


class ResampledShardsOnce(torch.utils.data.dataset.IterableDataset):
    '''
    Modified ResampledShards
    '''
    def __init__(self, datalist, nshards=sys.maxsize, worker_seed=None, deterministic=False):
        super().__init__()
        self.datalist = datalist
        self.nshards = nshards
        # If no worker_seed provided, use pytorch_worker_seed function; else use given seed
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = -1

    def __iter__(self):
        self.epoch += 1
        
        indices = list(range(len(self.datalist)))
        #random.Random(self.epoch).shuffle(indices) 
        
        for idx in indices:
            yield self.datalist[idx]

            
def read_npz(data):
    # Load a numpy .npz file from a file path or file-like object
    # The commented line shows how to load from bytes in memory
    # return np.load(io.BytesIO(data))
    return np.load(data)


def read_json(path):
    # Read and parse a JSON file from the given file path
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def padding(image, mask, center=True, padding_ratio_range=[1.15, 1.15]):
    """
    Pad the input image and mask to a square shape with padding ratio.

    Args:
        image (np.ndarray): Input image array of shape (H, W, C).
        mask (np.ndarray): Corresponding mask array of shape (H, W).
        center (bool): Whether to center the original image in the padded output.
        padding_ratio_range (list): Range [min, max] to randomly select padding ratio.

    Returns:
        newimg (np.ndarray): Padded image of shape (resize_side, resize_side, 3).
        newmask (np.ndarray): Padded mask of shape (resize_side, resize_side).
    """
    h, w = image.shape[:2]
    max_side = max(h, w)

    # Select padding ratio either fixed or randomly within the given range
    if padding_ratio_range[0] == padding_ratio_range[1]:
        padding_ratio = padding_ratio_range[0]
    else:
        padding_ratio = random.uniform(padding_ratio_range[0], padding_ratio_range[1])
    resize_side = int(max_side * padding_ratio)
    # resize_side = int(max_side * 1.15)

    pad_h = resize_side - h
    pad_w = resize_side - w
    if center:
        start_h = pad_h // 2
    else:
        start_h = pad_h - resize_side // 20
        
    start_w = pad_w // 2

    # Create new white image and black mask with padded size
    newimg = np.ones((resize_side, resize_side, 3), dtype=np.uint8) * 255
    newmask = np.zeros((resize_side, resize_side), dtype=np.uint8)
    
    # Place original image and mask into the padded canvas
    newimg[start_h:start_h + h, start_w:start_w + w] = image
    newmask[start_h:start_h + h, start_w:start_w + w] = mask
    
    return newimg, newmask


def viz_pc(surface, normal, image_input, name):
    image_input = image_input.cpu().numpy()
    image_input = image_input.transpose(1, 2, 0) * 0.5 + 0.5
    image_input = (image_input * 255).astype(np.uint8)
    cv2.imwrite(name + '.png', cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR))
    surface = surface.cpu().numpy()
    normal = normal.cpu().numpy()
    surface_mesh = trimesh.Trimesh(surface, vertex_colors=(normal + 1) / 2)
    surface_mesh.export(name + '.obj')

    
class AlignedShapeTextLatentDataset(torch.utils.data.dataset.IterableDataset):
    def __init__(
        self,
        data_list: str = None,
        cond_stage_key: str = "text",
        multimodal: bool = True,
        text_max_length: int = 512,
        image_transform = None,
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sharpedge_label: bool = False,
        return_normal: bool = False,
        deterministic = False,
        worker_seed = None,
        padding = True,
        padding_ratio_range=[1.15, 1.15]
    ):
        super().__init__()
        if isinstance(data_list, str) and data_list.endswith('.json'):
            self.data_list = read_json(data_list_json)
        elif isinstance(data_list, str) and os.path.isdir(data_list):
            self.data_list = glob.glob(data_list + '/*')
        else:
            self.data_list = data_list
        assert isinstance(self.data_list, list)
        self.rng = random.Random(0)
        
        self.cond_stage_key = cond_stage_key
        self.mulimodal = multimodal
        self.text_max_length = text_max_length
        self.image_transform = image_transform
                
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        
        rank_zero_info(f'*' * 50)
        rank_zero_info(f'Dataset Infos:')
        rank_zero_info(f'# of 3D file: {len(self.data_list)}')
        rank_zero_info(f'# of Surface Points: {self.pc_size}')
        rank_zero_info(f'# of Sharpedge Surface Points: {self.pc_sharpedge_size}')
        rank_zero_info(f'Using sharp edge label: {self.sharpedge_label}')
        rank_zero_info(f'*' * 50)


    def load_surface_sdf_points(self, rng, random_surface, sharpedge_surface):
        surface_normal = []
        if self.pc_size > 0:
            ind = rng.choice(random_surface.shape[0], self.pc_size, replace=False)
            random_surface = random_surface[ind]
            if self.sharpedge_label:
                sharpedge_label = np.zeros((self.pc_size, 1))
                random_surface = np.concatenate((random_surface, sharpedge_label), axis=1)
            surface_normal.append(random_surface)
            
        if self.pc_sharpedge_size > 0:
            ind_sharpedge = rng.choice(sharpedge_surface.shape[0], self.pc_sharpedge_size, replace=False)
            sharpedge_surface = sharpedge_surface[ind_sharpedge]
            if self.sharpedge_label:
                sharpedge_label = np.ones((self.pc_sharpedge_size, 1))
                sharpedge_surface = np.concatenate((sharpedge_surface, sharpedge_label), axis=1)
            surface_normal.append(sharpedge_surface)
            
        surface_normal = np.concatenate(surface_normal, axis=0)
        surface_normal = torch.FloatTensor(surface_normal)
        surface = surface_normal[:, 0:3]
        normal = surface_normal[:, 3:6]
        assert surface.shape[0] == self.pc_size + self.pc_sharpedge_size
        
        geo_points = 0.0
        normal = torch.nn.functional.normalize(normal, p=2, dim=1)
        if self.return_normal:
            surface = torch.cat([surface, normal], dim=-1)
        if self.sharpedge_label:
            surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)
        return surface, geo_points

    def load_render(self, imgs_path):
        imgs_choice = self.rng.sample(imgs_path, 1) # have all renders - take 1 BUT still as list 
        images, masks = [], []
        for image_path in imgs_choice:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            assert image.shape[2] == 4
            alpha = image[:, :, 3:4].astype(np.float32) / 255
            forground = image[:, :, :3]
            background = np.ones_like(forground) * 255
            img_new = forground * alpha + background * (1 - alpha)
            image = img_new.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = (alpha[:, :, 0] * 255).astype(np.uint8)

            if self.padding:            #crop + pad
                h, w = image.shape[:2]
                binary = mask > 0.3
                non_zero_coords = np.argwhere(binary)
                x_min, y_min = non_zero_coords.min(axis=0)
                x_max, y_max = non_zero_coords.max(axis=0)
                image, mask = padding(
                    image[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
                    mask[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
                    center=True, padding_ratio_range=self.padding_ratio_range)
            
            if self.image_transform:
                image = self.image_transform(image)
                mask = np.stack((mask, mask, mask), axis=-1)
                mask = self.image_transform(mask)
                
            images.append(image)
            masks.append(mask)
            
        images = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)[:1, ...]
        return images, masks


    def load_textprompt(self, prompt_path):
        def final_clean_text(text):
            if not text:
                return ""

            text = unicodedata.normalize('NFKC', text)

            text = text.replace("-\u00a0", "- ").replace("\u00a0", " ")
            text = text.replace("&", " and ")

            text = re.sub(r'[^a-zA-Z0-9\s.,:;\-_/()"\']', ' ', text)

            text = " ".join(text.split()).strip()

            return text.lower().capitalize()        
    
        prompt_data = read_json(prompt_path)
        
        raw_prompt = prompt_data['prompt']  # the full rich text above
        clean_prompt = final_clean_text(raw_prompt)
        
        dims = prompt_data['item_dimensions']
        
        unit_map = {
            'millimeters': 1.0, 'centimeters': 10.0,
            'meters': 1000.0, 'inches': 25.4
        }

        # ── Collect ALL measurements from JSON ──────────────────────────
        # Your JSON needs to store all feature measurements, not just L/H/W
        measurements = {}
        
        for key, dim_data in dims.items():
            val  = dim_data.get('value')
            unit = dim_data.get('unit', 'millimeters').lower().strip()
            if val is not None and unit in unit_map:
                measurements[key] = round(val * unit_map[unit], 2)  # all in mm

        # ── Compute per-sample scale ─────────────────────────────────────
        # Use max overall dimension — same formula as normalize_to_unit_box
        overall_dims = [
            measurements.get('length', 0),
            measurements.get('height', 0),
            measurements.get('width',  0),
        ]
        scale = max(overall_dims) * 1.01   # e.g. 397.50 * 1.01 = 401.47mm

        # ── Build aligned text ───────────────────────────────────────────
        # For EVERY measurement: show mm AND normalized value
        # So text encoder sees both "languages" simultaneously
        
        def fmt(mm_val):
            norm_val = mm_val / scale
            return f"{mm_val:.2f} mm (norm: {norm_val:.4f})"

        aligned_text = (
            f"{clean_prompt} "
            f"Total length: {fmt(measurements['length'])}. "
            f"Total height: {fmt(measurements.get('height', measurements['length']))}. "
            f"Total width: {fmt(measurements['width'])}. "
        )

        # Add feature-specific measurements if present
        feature_keys = [
            ('top_internal_width',    'Top internal width'),
            ('bottom_internal_width', 'Bottom internal width'),
            ('channel_depth',         'Channel depth'),
            ('wall_thickness',        'Wall thickness'),
            ('notch_width',           'Notch width'),
            ('notch_depth',           'Notch depth'),
        ]
        for key, label in feature_keys:
            if key in measurements:
                aligned_text += f"{label}: {fmt(measurements[key])}. "

        # ── Return BOTH text and extents tensor ──────────────────────────
        # extents tensor = normalized overall dims
        # used in ProstheticDiffuser for global_measurement_proj
        extents_norm = torch.tensor([
            measurements['length'] / scale,
            measurements.get('height', measurements['length']) / scale,
            measurements['width']  / scale,
        ], dtype=torch.float32)   # [3] all in [0,1], consistent with geometry

        return {
            'text':        aligned_text,
            'extents_norm': extents_norm,   # [3] — for conditioning token
            'scale':       scale,           # scalar — for inference rescaling
            'measurements_mm': measurements # raw mm — for inference rescaling
        }
        
    # def load_textprompt(self, prompt_path):      
        
    #     if not os.path.exists(prompt_path):
    #         raise FileNotFoundError(f"CRITICAL: Missing prompt file: {prompt_path}")
        
    #     prompt_data = read_json(prompt_path)
 
    #     raw_prompt = prompt_data['prompt']
    #     clean_prompt = final_clean_text(raw_prompt)

    #     dims_root = prompt_data['item_dimensions']
    #     converted = {}
    
    #     unit_map = {
    #         'millimeters': 1.0, 'centimeters': 10.0, 
    #         'meters': 1000.0, 'inches': 25.4
    #     }

    #     for dim in ['length', 'height', 'width']:
    #         dim_data = dims_root[dim]

    #         val = dim_data.get('value', dim_data.get('normalized_value', {}).get('value'))
    #         unit = dim_data.get('unit', dim_data.get('normalized_value', {}).get('unit'))

    #         if val is None or unit is None:
    #             raise KeyError(f"MISSING_DIMENSION: '{dim}' in {prompt_path}")

    #         unit_key = unit.lower().strip()
    #         if unit_key not in unit_map:
    #             raise ValueError(f"UNSUPPORTED_UNIT: '{unit_key}' in {prompt_path}")
            
    #         converted[dim] = round(val * unit_map[unit_key], 2)
                    
    #     # return (f"{clean_prompt}. "
    #     #         f"Total length: {converted['length']} mm. "
    #     #         f"Total height: {converted['height']} mm. "
    #     #         f"Total width: {converted['width']} mm.")
    #     return {
    #     'text': (f"{clean_prompt}. "
    #             f"Total length: {converted['length']} mm. "
    #             f"Total height: {converted['height']} mm. "
    #             f"Total width: {converted['width']} mm.")
    #     }
    
    def decode(self, item):
        uid = item.split('/')[-1]
        render_count = 20 #12 # was 24
        render_dir = os.path.join(item, 'render_cond')
        
        # TODO render_cond count renders ???
        # render_count = 0
        # if os.path.exists(os.path.join(item, f'render_cond')):
        #       for f in os.listdir(render_dir) 
        #         if f.lower().endswith('.png')
        #             render_count += 1
        
        render_img_paths = sorted(glob.glob(os.path.join(render_dir, "*.png")))      
        if not render_img_paths:
            raise FileNotFoundError(f"No renders found in {render_dir} for UUID {uid}")
                      
        #render_img_paths = [os.path.join(item, f'render_cond/{i:05d}.png') for i in range(render_count)]
        # transforms_json_path = os.path.join(item, 'render_cond/transforms.json')
        surface_npz_path = os.path.join(item, f'geo_data/{uid}_surface.npz')
        # sdf_npz_path = os.path.join(item, f'geo_data/{uid}_sdf.npz')
        # watertight_obj_path = os.path.join(item, f'geo_data/{uid}_watertight.obj')
        prompt_path = os.path.join(item, f'prompt/{uid}_prompt')
        
        sample = {}
        sample["image"] = render_img_paths
        surface_data = read_npz(surface_npz_path)
        sample["random_surface"] = surface_data['random_surface']
        sample["sharpedge_surface"] = surface_data['sharp_surface']
        sample["prompt"] = prompt_path
        return sample
    
    def transform(self, sample):
        rng = np.random.default_rng()

        prompt_data   = self.load_textprompt(sample["prompt"])
        text          = prompt_data['text']           # aligned text with norm values
        extents_norm  = prompt_data['extents_norm']   # [3] tensor
        scale         = prompt_data['scale']          # scalar for inference

        image_input, mask_input = self.load_render(sample['image'])
        surface, geo_points = self.load_surface_sdf_points(
            rng,
            sample["random_surface"],
            sample["sharpedge_surface"],
        )

        return {
            "surface":      surface,
            "image":        image_input,
            "mask":         mask_input,
            "prompt":       text,          # rich aligned text
            "extents_norm": extents_norm,  # [3] for ProstheticDiffuser
            "scale":        torch.tensor([scale], dtype=torch.float32),
        }
        
    # def transform(self, sample):
    #     rng = np.random.default_rng()
    #     random_surface = sample.get("random_surface", 0)
    #     sharpedge_surface = sample.get("sharpedge_surface", 0)
    #     # text = self.load_textprompt(sample["prompt"])
    #     prompt_data   = self.load_textprompt(sample["prompt"])
    #     text          = prompt_data['text']
    #     extents_mm    = prompt_data['extents']          # [length, height, width] in mm
        
    #     image_input, mask_input = self.load_render(sample['image'])
    #     surface, geo_points = self.load_surface_sdf_points(rng, random_surface, sharpedge_surface)
    #     #prompt = self.load_prompts(text)
    #     sample = {
    #         "surface": surface,
    #         "geo_points": geo_points,
    #         "image": image_input,
    #         "mask": mask_input,
    #         "prompt": text, # just string  #TODO LOOK
    #     }
    #     return sample

    def __iter__(self):
        total_num = 0
        failed_num = 0
        for data in ResampledShardsOnce(self.data_list): # was ResampledShards
            total_num += 1
            if total_num % 1000 == 0:
                print(f"Current failure rate of data loading:")
                print(f"{failed_num}/{total_num}={failed_num/total_num}")
            try:
                sample = self.decode(data)
                sample = self.transform(sample)
            except Exception as err:
                print(err)
                failed_num += 1
                continue
            yield sample


class AlignedShapeTextLatentModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        multimodal: bool = True,
        text_max_length: int = 512,
        val_num_workers: int = 2,
        train_data_list: str = None,
        val_data_list: str = None,
        train_paths: str = None,
        cond_stage_key: str = "text",
        image_size: int = 224,
        mean: Union[List[float], Tuple[float]] = (0.485, 0.456, 0.406),
        std: Union[List[float], Tuple[float]] = (0.229, 0.224, 0.225),
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sharpedge_label: bool = False,
        return_normal: bool = False, 
        padding = True,
        padding_ratio_range=[1.15, 1.15]       
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers
        
        self.train_paths = train_paths
        # self.train_data_list = 
        # #self.val_data_list = val_data_list
        
        self.cond_stage_key = cond_stage_key
        self.multimodal = multimodal
        self.text_max_length = text_max_length
        
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.train_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean, std=self.std)])
        self.val_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean, std=self.std)])

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        
    def train_dataloader(self):
        astl_params = {
            "data_list": self.train_data_list,
            "cond_stage_key": self.cond_stage_key,
            "multimodal": self.multimodal,
            "text_max_length": self.text_max_length,
            "image_transform": self.train_image_transform,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal,
            "padding": self.padding,
            "padding_ratio_range": self.padding_ratio_range
        }
        dataset = AlignedShapeTextLatentDataset(**astl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True, #
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        asl_params = {
            "data_list": self.val_data_list,
            "cond_stage_key": self.cond_stage_key,
            "multimodal": self.multimodal,
            "text_max_length": self.text_max_length,
            "image_transform": self.val_image_transform,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal, 
            "padding": self.padding,
            "padding_ratio_range": self.padding_ratio_range
        }
        dataset = AlignedShapeTextLatentDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
    
    def setup(self, stage=None):
        all_data = sorted([
            os.path.join(self.train_paths, d) 
            for d in os.listdir(self.train_paths) 
            if os.path.isdir(os.path.join(self.train_paths, d))
        ])

        random.seed(42)
        random.shuffle(all_data)
        
        total_count = len(all_data)
        if total_count == 0:
            raise FileNotFoundError(f"No folders found in {self.train_paths}")

        split_idx = int(total_count * 0.8)

        self.train_data_list = all_data[:split_idx]    # ~6362 with ADO
        self.val_data_list = all_data[split_idx:]      # ~1591 

        print(f"Dataset Split (80/20): Total={total_count}, Train={len(self.train_data_list)}, Val={len(self.val_data_list)}")

    
    # def setuplists(self, stage=None):
    # 1. Отримуємо повний список усіх папок з об'єктами

        # self.train_paths = all_data[:55000]
        # self.val_paths = all_data[55000:60000]
