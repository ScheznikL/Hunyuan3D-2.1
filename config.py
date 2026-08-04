"""Inference parameters for the vendored Hunyuan3D-2.1 project, loaded from config.yaml.

Also relocates two module-level flags the vendored source hardcodes to
False (hy3dshape.pipelines.TEST/USE_PRIOR_TR, hy3dshape's denoiser TEST) —
__init__.py sets those modules' attributes from TEST_MODE/USE_PRIOR_TR
before loading a pipeline, since they're plain globals the vendored code
branches on directly, not constructor params.
"""

from pathlib import Path

import yaml

_ROOT = Path(__file__).parent
_CONFIG_PATH = _ROOT / "config.yaml"
# backend/external/Hunyuan3D -> backend/external -> backend
_GENERATION_ASSETS_DIR = _ROOT.parent.parent / "generation" / "assets"

with _CONFIG_PATH.open("r", encoding="utf-8") as _f:
    _config = yaml.safe_load(_f)

MODEL_PATH: str = _config["model_path"]
SUBFOLDER: str = _config["subfolder"]
DEVICE: str = _config["device"]
USE_SAFETENSORS: bool = _config["use_safetensors"]
VARIANT: str | None = _config["variant"]

CHECKPOINT_PATH: str | None = _config["checkpoint_path"]
CHECKPOINT_CONFIG_PATH: str | None = _config["checkpoint_config_path"]
CHECKPOINT_BASE_MODEL_PATH: str = _config["checkpoint_base_model_path"]
CHECKPOINT_BASE_SUBFOLDER: str = _config["checkpoint_base_subfolder"]
CHECKPOINT_USE_PEFT: bool = _config["checkpoint_use_peft"]

TEST_MODE: bool = _config["test_mode"]
USE_PRIOR_TR: bool = _config["use_prior_tr"]

NUM_INFERENCE_STEPS: int = _config["num_inference_steps"]
GUIDANCE_SCALE: float = _config["guidance_scale"]
OCTREE_RESOLUTION: int = _config["octree_resolution"]
NUM_CHUNKS: int = _config["num_chunks"]
MC_ALGO: str | None = _config["mc_algo"]
SEED: int = _config["seed"]

PRIOR_PATH: str | None = (
    str(_GENERATION_ASSETS_DIR / _config["prior_path"]) if _config.get("prior_path") else None
)
PRIOR_ALPHA: float = _config["prior_alpha"]

LOW_VRAM_MODE: bool = _config["low_vram_mode"]
ENABLE_FLASHVDM: bool = _config["enable_flashvdm"]
COMPILE: bool = _config["compile"]

ENABLE_TEXTURE: bool = _config["enable_texture"]
PAINT_MAX_NUM_VIEW: int = _config["paint_max_num_view"]
PAINT_RESOLUTION: int = _config["paint_resolution"]
REALESRGAN_CKPT_PATH: str = str(_ROOT / _config["realesrgan_ckpt_path"])
MULTIVIEW_CFG_PATH: str = str(_ROOT / _config["multiview_cfg_path"])
PAINT_CUSTOM_PIPELINE: str = _config["paint_custom_pipeline"]

SCRATCH_DIR: str | None = _config["scratch_dir"]
