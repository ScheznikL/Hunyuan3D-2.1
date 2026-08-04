"""Programmatic entry point into the vendored Hunyuan3D-2.1 project.

torch/trimesh/hy3dshape/hy3dpaint (and peft, for the checkpoint branch) are
only imported inside the functions that need them (not at module scope) so
the rest of the backend can still boot when this submodule's requirements
aren't installed — only calling load_shape_pipeline()/generate() actually
requires them.

Mirrors model_worker.py::ModelWorker's real flow (shape generation, then an
attempted texture pass that falls back to the untextured mesh on failure),
not experiments/hungred/generate_single_entry.py (a personal, broken research
script — not imported from here). experiments/pipe_utils.py IS imported from
(it's real and working) for the checkpoint-loading path.
"""

from __future__ import annotations

import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    import trimesh
    from PIL.Image import Image as PILImage

_ROOT = Path(__file__).parent
_shape_pipeline = None
_paint_pipeline = None
_module_flags_applied = False


def _ensure_submodule_paths() -> None:
    """hy3dshape/hy3dpaint/experiments use bare `import textureGenPipeline`-style
    imports that only resolve if their own directories are on sys.path,
    mirroring what model_worker.py's `sys.path.insert(0, './hy3dshape')` does
    when run with the submodule root as cwd."""
    for sub in ("hy3dshape", "hy3dpaint", "experiments"):
        p = str(_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def _resolve_checkpoint() -> tuple[str, str] | None:
    """Local fine-tuned checkpoint takes priority over the default HF model
    when both its weights and training config are present on disk."""
    if not config.CHECKPOINT_PATH or not config.CHECKPOINT_CONFIG_PATH:
        return None
    if not os.path.exists(config.CHECKPOINT_PATH) or not os.path.exists(config.CHECKPOINT_CONFIG_PATH):
        return None
    return config.CHECKPOINT_PATH, config.CHECKPOINT_CONFIG_PATH


def _resolve_prior_path() -> str | None:
    """Prior mesh used to initialize/blend shape latents, if configured and present."""
    if config.PRIOR_PATH and os.path.exists(config.PRIOR_PATH):
        return config.PRIOR_PATH
    return None


def _apply_module_flags() -> None:
    """TEST/USE_PRIOR_TR are plain module globals the vendored code branches
    on directly (not constructor params) — set them from config instead of
    leaving them hardcoded False."""
    global _module_flags_applied
    if _module_flags_applied:
        return

    _ensure_submodule_paths()
    import hy3dshape.pipelines as _pipelines_module
    from hy3dshape.models.denoisers import hunyuan3ddit as _denoiser_module

    _pipelines_module.TEST = config.TEST_MODE
    _pipelines_module.USE_PRIOR_TR = config.USE_PRIOR_TR
    _denoiser_module.TEST = config.TEST_MODE  # currently unread there, set for parity/future-proofing

    _module_flags_applied = True


def load_shape_pipeline():
    """Load (and cache) the shape (image-to-mesh) pipeline, preferring a local checkpoint."""
    global _shape_pipeline
    if _shape_pipeline is not None:
        return _shape_pipeline

    _apply_module_flags()

    checkpoint = _resolve_checkpoint()
    if checkpoint is not None:
        ckpt_path, checkpoint_config_path = checkpoint
        from pipe_utils import setup_pipeline_with_custom_weights

        pipeline = setup_pipeline_with_custom_weights(
            ckpt_path=ckpt_path,
            config_path=checkpoint_config_path,
            model_path=config.CHECKPOINT_BASE_MODEL_PATH,
            default_subfolder=config.CHECKPOINT_BASE_SUBFOLDER,
            use_peft=config.CHECKPOINT_USE_PEFT,
        )
    else:
        from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipelineWithPriorInitialization

        pipeline_cls = (
            Hunyuan3DDiTFlowMatchingPipelineWithPriorInitialization
            if _resolve_prior_path()
            else Hunyuan3DDiTFlowMatchingPipeline
        )
        pipeline = pipeline_cls.from_pretrained(
            config.MODEL_PATH,
            subfolder=config.SUBFOLDER,
            device=config.DEVICE,
            use_safetensors=config.USE_SAFETENSORS,
            variant=config.VARIANT,
        )

    if config.ENABLE_FLASHVDM:
        pipeline.enable_flashvdm(mc_algo=config.MC_ALGO or "mc")
    if config.COMPILE:
        pipeline.compile()

    _shape_pipeline = pipeline
    return pipeline


def load_paint_pipeline():
    """Load (and cache) the texture-painting pipeline."""
    global _paint_pipeline
    if _paint_pipeline is not None:
        return _paint_pipeline

    _ensure_submodule_paths()
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    paint_config = Hunyuan3DPaintConfig(config.PAINT_MAX_NUM_VIEW, config.PAINT_RESOLUTION)
    paint_config.device = config.DEVICE
    paint_config.realesrgan_ckpt_path = config.REALESRGAN_CKPT_PATH
    paint_config.multiview_cfg_path = config.MULTIVIEW_CFG_PATH
    paint_config.custom_pipeline = config.PAINT_CUSTOM_PIPELINE

    _paint_pipeline = Hunyuan3DPaintPipeline(paint_config)
    return _paint_pipeline


def generate(image_bytes: bytes, prompt: str = "") -> "trimesh.Trimesh":
    """Run Hunyuan3D image-to-3D inference and return the resulting mesh.

    prompt is only consumed on the checkpoint/AlignedSpace branch (the only
    pipeline class here that actually implements text conditioning) — it's
    silently unused otherwise.
    """
    import torch
    from PIL import Image
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipelineInAlighnedSpace

    pipeline = load_shape_pipeline()
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")

    call_kwargs = {
        "image": image,
        "generator": torch.Generator().manual_seed(config.SEED),
        "output_type": "trimesh",
        "num_inference_steps": config.NUM_INFERENCE_STEPS,
        "guidance_scale": config.GUIDANCE_SCALE,
        "octree_resolution": config.OCTREE_RESOLUTION,
        "num_chunks": config.NUM_CHUNKS,
        "mc_algo": config.MC_ALGO,
    }

    prior_path = _resolve_prior_path()
    if prior_path:
        call_kwargs["prior"] = prior_path
        call_kwargs["alpha"] = config.PRIOR_ALPHA

    if isinstance(pipeline, Hunyuan3DDiTFlowMatchingPipelineInAlighnedSpace):
        call_kwargs["prompt"] = prompt

    with torch.inference_mode():
        outputs = pipeline(**call_kwargs)

    mesh: "trimesh.Trimesh" = outputs[0]
    if isinstance(mesh, list):
        mesh = mesh[0]

    if not config.ENABLE_TEXTURE:
        return mesh

    try:
        with tempfile.TemporaryDirectory(dir=config.SCRATCH_DIR) as scratch:
            initial_path = os.path.join(scratch, "shape.glb")
            mesh.export(initial_path)

            paint_pipeline = load_paint_pipeline()
            obj_path = os.path.join(scratch, "textured.obj")
            textured_obj_path = paint_pipeline(
                mesh_path=initial_path,
                image_path=image,
                output_mesh_path=obj_path,
                save_glb=False,
            )
            from hy3dpaint.convert_utils import create_glb_with_pbr_materials

            textures = {
                "albedo": textured_obj_path.replace(".obj", ".jpg"),
                "metallic": textured_obj_path.replace(".obj", "_metallic.jpg"),
                "roughness": textured_obj_path.replace(".obj", "_roughness.jpg"),
            }
            glb_path = os.path.join(scratch, "textured.glb")
            create_glb_with_pbr_materials(textured_obj_path, textures, glb_path)

            import trimesh

            return trimesh.load(glb_path)
    except Exception:
        return mesh  # fall back to the untextured in-memory mesh, same as ModelWorker.generate
