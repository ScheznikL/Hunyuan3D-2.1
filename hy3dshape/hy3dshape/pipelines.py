TEST = False
USE_PRIOR_TR = False
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

import copy
import importlib
import inspect
import os
import random
from typing import List, Optional, Union


import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_accelerate_version, is_accelerate_available
from tqdm import tqdm
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
import open3d as o3d


from .models.autoencoders import ShapeVAE
from .models.autoencoders import SurfaceExtractors
from .utils import logger, synchronize_timer, smart_load_model


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@synchronize_timer('Export to trimesh')
def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    cls = get_obj_from_str(config["target"])
    params = config.get("params", dict())
    kwargs.update(params)
    instance = cls(**kwargs)
    return instance


class Hunyuan3DDiTPipeline:
    model_cpu_offload_seq = "conditioner->model->vae"
    _exclude_from_cpu_offload = []

    @classmethod
    @synchronize_timer('Hunyuan3DDiTPipeline Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        **kwargs,
    ):
        # load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")
        logger.info(f"Loading model from {ckpt_path}")

        if use_safetensors:
            # parse safetensors
            import safetensors.torch
            safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
            ckpt = {}
            for key, value in safetensors_ckpt.items():
                model_name = key.split('.')[0]
                new_key = key[len(model_name) + 1:]
                if model_name not in ckpt:
                    ckpt[model_name] = {}
                ckpt[model_name][new_key] = value
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        # load model
        model = instantiate_from_config(config['model'])
        model.load_state_dict(ckpt['model'], strict=False) # TODO check strict=False was added
        vae = instantiate_from_config(config['vae'])
        vae.load_state_dict(ckpt['vae'], strict=False)
        conditioner = instantiate_from_config(config['conditioner'])
        if 'conditioner' in ckpt:
            if TEST:
                ckpt_cond = ckpt["conditioner"]
                image_state = {
                    k.replace("main_image_encoder.", ""): v
                    for k, v in ckpt_cond.items()
                    if k.startswith("main_image_encoder.")
                }
                #missing, unexpected =
                conditioner.image_encoder.load_state_dict(
                    image_state,
                    strict=False,
                    #strict=True
                    )    
            #   conditioner.load_state_dict(ckpt['conditioner']) #TODO change arch
            else:
                conditioner.load_state_dict(ckpt['conditioner']) #TODO change arch

        image_processor = instantiate_from_config(config['image_processor'])
        scheduler = instantiate_from_config(config['scheduler'])

        model_kwargs = dict(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor,
            device=device,
            dtype=dtype,
        )
        model_kwargs.update(kwargs)

        return cls(
            **model_kwargs
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-1',
        **kwargs,
    ):
        kwargs['from_pretrained_kwargs'] = dict(
            model_path=model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant,
            dtype=dtype,
            device=device,
        )
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )
        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    def __init__(
        self,
        vae,
        model,
        scheduler,
        conditioner,
        image_processor,
        device='cuda',
        dtype=torch.float16,
        **kwargs
    ):
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.image_processor = image_processor
        self.kwargs = kwargs
        self.to(device, dtype)

    def compile(self):
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        self.conditioner = torch.compile(self.conditioner)

    def enable_flashvdm(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='mc',
        replace_vae=True,
    ):
        if enabled:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            turbo_vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini-turbo'),
            }
            model_name = model_path.split('/')[-1]
            if replace_vae and model_name in turbo_vae_mapping:
                model_path, subfolder = turbo_vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(
                    model_path, subfolder=subfolder,
                    use_safetensors=self.kwargs['from_pretrained_kwargs']['use_safetensors'],
                    device=self.device,
                )
            self.vae.enable_flashvdm_decoder(
                enabled=enabled,
                adaptive_kv_selection=adaptive_kv_selection,
                topk_mode=topk_mode,
                mc_algo=mc_algo
            )
        else:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini'),
            }
            model_name = model_path.split('/')[-1]
            if model_name in vae_mapping:
                model_path, subfolder = vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(model_path, subfolder=subfolder)
            self.vae.enable_flashvdm_decoder(enabled=False)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            self.conditioner.to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            self.vae.to(device)
            self.model.to(device)
            self.conditioner.to(device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        if self.model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of "
                f"the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`)
        # or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu")
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()  
                # otherwise we don't see the memory savings (but they probably exist)

        all_model_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)
            if not isinstance(model, torch.nn.Module):
                continue

            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

        # CPU offload models that are not in the seq chain unless they are explicitly excluded
        # these models will stay on CPU until maybe_free_model_hooks is called
        # some models cannot be in the seq chain because they are iteratively called, 
        # such as controlnet
        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        r"""
        Function that offloads all components, removes all model hooks that were added when using
        `enable_model_cpu_offload` and then applies them again. In case the model has not been offloaded this function
        is a no-op. Make sure to add this function to the end of the `__call__` function of your pipeline so that it
        functions correctly when applying enable_model_cpu_offload.
        """
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            # `enable_model_cpu_offload` has not be called, so silently do nothing
            return

        for hook in self._all_hooks:
            # offload model and remove hook from model
            hook.offload()
            hook.remove()

        # make sure the model is in the same state as before calling it
        self.enable_model_cpu_offload()

    @synchronize_timer('Encode cond')
    def encode_cond(self, image, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
        bsz = image.shape[0]
        cond = self.conditioner(image=image, **additional_cond_inputs)

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            if dual_guidance:
                un_cond_drop_main = copy.deepcopy(un_cond)
                un_cond_drop_main['additional'] = cond['additional']

                def cat_recursive(a, b, c):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b, c], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k], c[k])
                    return out

                cond = cat_recursive(cond, un_cond_drop_main, un_cond)
            else:
                def cat_recursive(a, b):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k])
                    return out

                cond = cat_recursive(cond, un_cond)
        return cond

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, dtype, device, generator, latents=None):
        shape = (batch_size, *self.vae.latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents

    def prepare_image(self, image, mask=None) -> dict:
        if isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor):
            outputs = {
                'image': image,
                'mask': mask
            }
            return outputs
            
        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Couldn't find image at path {image}")

        if not isinstance(image, list):
            image = [image]

        outputs = []
        for img in image:
            output = self.image_processor(img)
            outputs.append(output)

        cond_input = {k: [] for k in outputs[0].keys()}
        for output in outputs:
            for key, value in output.items():
                cond_input[key].append(value)
        for key, value in cond_input.items():
            if isinstance(value[0], torch.Tensor):
                cond_input[key] = torch.cat(value, dim=0)

        return cond_input

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def set_surface_extractor(self, mc_algo):
        if mc_algo is None:
            return
        logger.info('The parameters `mc_algo` is deprecated, and will be removed in future versions.\n'
                    'Please use: \n'
                    'from hy3dshape.models.autoencoders import SurfaceExtractors\n'
                    'pipeline.vae.surface_extractor = SurfaceExtractors[mc_algo]() instead\n')
        if mc_algo not in SurfaceExtractors.keys():
            raise ValueError(f"Unknown mc_algo {mc_algo}")
        self.vae.surface_extractor = SurfaceExtractors[mc_algo]()

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        dual_guidance_scale: float = 10.5,
        dual_guidance: bool = True,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=-1 / 512,
        num_chunks=8000,
        mc_algo=None,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and \
                                      getattr(self.model, 'guidance_cond_proj_dim', None) is None
        dual_guidance = dual_guidance_scale >= 0 and dual_guidance

        if isinstance(image, torch.Tensor):
            pass
        else:
            cond_inputs = self.prepare_image(image)
            image = cond_inputs.pop('image')
        
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        batch_size = image.shape[0]

        t_dtype = torch.long
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas)

        logger.info(f"timespeps are: {timesteps}")

        latents = self.prepare_latents(batch_size, dtype, device, generator)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        guidance_cond = None
        if getattr(self.model, 'guidance_cond_proj_dim', None) is not None:
            logger.info('Using lcm guidance scale')
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
            guidance_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.guidance_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:", leave=False)):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * (3 if dual_guidance else 2))
                else:
                    latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                timestep_tensor = torch.tensor([t], dtype=t_dtype, device=device)
                timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
                noise_pred = self.model(latent_model_input, timestep_tensor, cond, guidance_cond=guidance_cond)

                # no drop, drop clip, all drop
                if do_classifier_free_guidance:
                    if dual_guidance:
                        noise_pred_clip, noise_pred_dino, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_clip - noise_pred_dino)
                            + dual_guidance_scale * (noise_pred_dino - noise_pred_uncond)
                        )
                    else:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1 
                outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
        )

    def _export(
        self,
        latents,
        output_type='trimesh',
        box_v=1.01,
        mc_level=0.0,
        num_chunks=20000,
        octree_resolution=256,
        mc_algo='mc',
        enable_pbar=True
    ):
        if not output_type == "latent":
            latents = 1. / self.vae.scale_factor * latents
            latents = self.vae(latents)
            outputs = self.vae.latents2mesh(
                latents,
                bounds=box_v,
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_algo=mc_algo,
                enable_pbar=enable_pbar,
            )
        else:
            outputs = latents

        if output_type == 'trimesh':
            outputs = export_to_trimesh(outputs)

        return outputs


class Hunyuan3DDiTFlowMatchingPipeline(Hunyuan3DDiTPipeline):

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        mask = None,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        # print('image', type(image), 'mask', type(mask))
        cond_inputs = self.prepare_image(image, mask)
        image = cond_inputs.pop('image')
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )

        batch_size = image.shape[0]

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        # logger.info(f"timespeps are: {timesteps}")
        
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
            # logger.info(f'Using guidance embed with scale {guidance_scale}')

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)


class Hunyuan3DDiTFlowMatchingPipelineWithPriorInitialization(Hunyuan3DDiTPipeline):

    @torch.inference_mode()
    def __call__(
        self,
        prior, # TODO def
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None, # TODO text
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        mask = None,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo) # 

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        # print('image', type(image), 'mask', type(mask))
        cond_inputs = self.prepare_image(image, mask)

        image = cond_inputs.pop('image')

        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )

        batch_size = image.shape[0]

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        
        z_prior = self.prepare_prior(prior=prior, debug_img_dir=kwargs.pop("debug_img_dir", None))

        alpha = kwargs.pop("alpha", None) ## TODO Add to params
        latents = self.prepare_latents_with_prior( #TODO check
            z_prior=z_prior,
            batch_size=batch_size, #image.shape[0]
            dtype=dtype,
            device=device,
            generator=generator,
            alpha=alpha, #0.8 init
        )

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
            # logger.info(f'Using guidance embed with scale {guidance_scale}')

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

                  # shape prior attraction TODO - Check difference
                #latents = latents + prior_weight * (z_prior - latents)

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )

    def prepare_prior(self, prior, debug_img_dir = None) -> torch.FloatTensor :
        from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader

        logger.info(f'we pass self.vae.encoder.pc_size eq {self.vae.encoder.pc_size}')
        
        loader = SharpEdgeSurfaceLoader(
        num_sharp_points=0, #5120,
        num_uniform_points=81920, #based on ShapeVAE81920
        )
        # mesh_demo = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        # surface = loader(mesh_demo).to('cuda', dtype=torch.float16)
        # surface[:, :, :3] = surface[:, :, :3] * 0.8 # normalize the cube to [-0.8, 0.8]
        #TODO DO
        surface = loader(prior).to('cuda', dtype=torch.float16)
        print("[INFO] loaded surface shape:", surface.shape)
        try:
            if debug_img_dir is not None:
                pcdd = self.surface_tensor_to_pcd(surface)
                print(pcdd)
                # self.save_pc_image(debug_img_dir, point_cloud_data = pcdd) 
                self.save_pc_image_matplotlib(debug_img_dir, point_cloud_data = pcdd) 
        except Exception as e: 
            logger.error(f"Debug image creation: {e}")
            pass 
        
        # vae = ShapeVAE.from_pretrained(
        #     'tencent/Hunyuan3D-2.1',
        #     use_safetensors=False,
        #     variant='fp16',
        # )
        # shape
        default_subfolder='hunyuan3d-vae-v2-mini-withencoder'
        model_path = 'tencent/Hunyuan3D-2mini'
        
        vae = ShapeVAE.from_pretrained(
            model_path,
            subfolder = default_subfolder,
            use_safetensors=False,
            variant='fp16',
        )
        # if isinstance(self.vae, ShapeVAE):
        #     latents = self.vae.encode(surface)

        with torch.no_grad():         
            latents = vae.encode(surface)
            logger.info("surface shape encoded.")
           # shape = (batch_size, *self.vae.latent_shape) #batch_size  = 1 (img count) and unpacked 512, 64
        
        return latents
            
    def surface_tensor_to_pcd(self, surface: torch.Tensor) -> o3d.geometry.PointCloud:
        surface_np = surface.detach().cpu().float().numpy()

       # xyz = surface_np[:, :3]
        xyz = surface_np[0, :, :3]  

        # Sanity checks
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Invalid XYZ shape: {xyz.shape}")

        if not np.isfinite(xyz).all():
            raise ValueError("XYZ contains NaNs or Infs")

        xyz = xyz.astype(np.float64)

        z = xyz[:, 2]
        z = xyz[:, 2]
        z_scale = np.ptp(z)

        z_norm = (z - z.min()) / (z_scale + 1e-8)

        # scale = np.ptp(xyz, axis=0)

        # z_norm = (z - z.min()) / (scale + 1e-8)

        colors = np.stack(
            [z_norm, np.zeros_like(z_norm), 1.0 - z_norm],
            axis=1
        ).astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def save_pc_image(
        self,
        out_path: str,
        point_cloud_data = None   
        ):

        """
        Render point cloud offscreen and save image to file.
        Works on SLURM / headless nodes.
        """
        width: int = 1024
        height: int = 1024
        point_size: float = 2.0
        bg_color=(1.0, 1.0, 1.0)

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            visible=False,
            # width=width,
            # height=height,
        )
        success = vis.create_window(visible=False, width=width, height=height)
    
        if not success:
            logger.error("The windows was not created Open3D. Check for EGL or Xvfb.")
            return

        vis.add_geometry(point_cloud_data)

        render_opt = vis.get_render_option()
        if render_opt is None:
            logger.error("RenderOption is None")
            vis.destroy_window()
            return
    
        render_opt.background_color = np.array(bg_color)
        render_opt.point_size = point_size

        vis.poll_events()
        vis.update_renderer()

        # Save image
        out_path = os.path.join(out_path,"debug_image.png")
        vis.capture_screen_image(out_path)
        vis.destroy_window()

    def save_pc_image_matplotlib(self, out_path, point_cloud_data):
        import matplotlib.pyplot as plt
        #rom mpl_toolkits.mplot3d import Axes3D
        """
        Зберігає зображення хмари точок без використання GPU/Open3D.
        Ідеально для SLURM.
        """
        # 1. Отримуємо точки (convert Open3D pcd to numpy)
        points = np.asarray(point_cloud_data.points)
        
        # Якщо у точок є кольори, дістаємо і їх
        colors = None
        if point_cloud_data.has_colors():
            colors = np.asarray(point_cloud_data.colors)

        # 2. Створюємо фігуру
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 3. Малюємо Scatter Plot
        # s - розмір точки, c - колір
        if colors is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=colors)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c='blue')

        # 4. Налаштовуємо вигляд (щоб не було осей)
        ax.set_axis_off()
        
        # Можна налаштувати кут огляду
        ax.view_init(elev=20, azim=45)

        # 5. Зберігаємо
        os.makedirs(out_path, exist_ok=True)
        final_path = os.path.join(out_path, "debug_image_plt.png")
        
        # savefig працює в headless режимі без проблем
        plt.savefig(final_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig) # Обов'язково закриваємо, щоб не текла пам'ять
        print(f"Дебаг-зображення збережено через Matplotlib: {final_path}")

    def prepare_latents_with_prior(
        self,
        z_prior: torch.FloatTensor,
        batch_size: int,
        dtype,
        device,
        generator,
        alpha: float = 0.7,
    ):
        """
        z0 = alpha * z_prior + (1 - alpha) * noise
        """
        shape = (batch_size, *self.vae.latent_shape) #batch_size  = 1 (img count) and unpacked 512, 64
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        assert z_prior.shape == latents.shape

        # scale noise as scheduler expects
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)

        z_prior = z_prior.to(device=device, dtype=dtype)
        
        latents  = alpha * z_prior + np.sqrt(1 - alpha**2) * latents 

        #latents = alpha * z_prior + (1.0 - alpha) * noise
        return latents



class Hunyuan3DDiTFlowMatchingPipelineInAlighnedSpace(Hunyuan3DDiTPipeline):

    def __init__(self, vae, model, scheduler, conditioner, image_processor, device='cuda', dtype=torch.float16, **kwargs):
        super().__init__(vae, model, scheduler, conditioner, image_processor, device, dtype, **kwargs)
        
        default_subfolder='hunyuan3d-vae-v2-mini-withencoder'
        model_path = 'tencent/Hunyuan3D-2mini'
        
        self.vae = ShapeVAE.from_pretrained(
            model_path,
            subfolder = default_subfolder,
            use_safetensors=False,
            variant='fp16',
        )
    
    @torch.inference_mode()
    def __call__(
        self,
        prompt,
        #prior: str = None, # TODO def        
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None, 
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        mask = None,
         # NEW — stress weights per part, optional
        part_stress: Optional[dict] = None,  
        part_token_indices: Optional[dict] = None,  # from your ablation test
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        DEFAULT_PROSTHETIC_STRESS = {
            'internal_cavity': 0.95,   # safety-critical — almost frozen
            'connector':       0.80,   # load-bearing
            'external_bottom': 0.25,   # some freedom
            'external_top':    0.10,   # most stylistic freedom
        }
        part_stress = part_stress or DEFAULT_PROSTHETIC_STRESS
        
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo) # 

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        # print('image', type(image), 'mask', type(mask))
        if image is not None:
            cond_inputs = self.prepare_image(image, mask) 
            image = cond_inputs.pop('image')

# TODO Add text along side outputs['main']
        
        cond = self.encode_cond(
            image=image,
            text=prompt,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )

        batch_size = image.shape[0]

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        
        prior = kwargs.pop("prior", None)
        
        z_prior = self.prepare_prior(prior=prior, debug_img_dir=kwargs.pop("debug_img_dir", None)) if prior else None
        alpha = kwargs.pop("alpha", None) ## TODO Add to params
        
        # ── 2. Prepare latents (modified) ─────────────────────────────────
        if z_prior is not None and USE_PRIOR_TR:
            latents = self.prepare_latents_with_prior_mechstyle(
                z_prior=z_prior,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
                generator=generator,
                alpha=alpha,                      # global fallback
                part_stress=part_stress,          # per-part override
                part_token_indices=part_token_indices,  # None until ablation test done
            )
        elif USE_PRIOR_TR:
            latents = self.prepare_latents(batch_size, dtype, device, generator)
            logger.info("No prior being processed")
            
        if z_prior is not None: 
            logger.info("Using prepare_latents_with_prior")
            latents = self.prepare_latents_with_prior( #TODO check
                z_prior=z_prior,
                batch_size=batch_size, #image.shape[0]
                dtype=dtype,
                device=device,
                generator=generator,
                alpha=alpha, #0.8 init
            )

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
            # logger.info(f'Using guidance embed with scale {guidance_scale}')

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample
                
                # NEW: MechStyle temporal attraction
                # if z_prior is not None and part_token_indices is not None:
                #     t_progress = i / num_inference_steps
                #     latents = self._apply_stress_attraction(
                #         latents, z_prior, part_stress,
                #         part_token_indices, t_progress
                #     )
                  
                # shape prior attraction TODO - Check difference 
                #latents = latents + prior_weight * (z_prior - latents)

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )
        
    def _apply_stress_attraction(
        self,
        latents: torch.Tensor,
        z_prior: torch.Tensor,
        part_stress: dict,
        part_token_indices: dict,
        t_progress: float,          # 0.0 (start) → 1.0 (end)
    ) -> torch.Tensor:
        """
        MechStyle temporal scheduling:
        - Early steps: strong pull toward prior for all parts
        - Late steps: only high-stress parts stay attracted
        
        High stress (cavity=0.95): attraction persists whole denoising
        Low stress  (shell=0.10):  attraction fades quickly → style emerges
        """
        latents = latents.clone()
        
        for part_name, stress in part_stress.items():
            if part_name not in part_token_indices:
                continue
                
            idx = part_token_indices[part_name]
            
            # Stress controls how long attraction persists
            # stress=0.95: still 0.95 attraction at t_progress=0.9
            # stress=0.10: already ~0.0 attraction at t_progress=0.3
            attraction = stress * max(0.0, 1.0 - t_progress / stress)
            
            if attraction > 0:
                latents[:, idx, :] = (
                    latents[:, idx, :] +
                    attraction * (z_prior[:, idx, :] - latents[:, idx, :])
                )
        
        return latents
## Why `part_token_indices=None` is fine for now

# The design is staged — everything works at each stage:
# ```
# Stage 1 NOW:   part_token_indices=None
#                → stress attraction skipped (None guard)  
#                → alpha still applied globally via prepare_latents_with_prior
#                → run your ablation test

# Stage 2 AFTER ABLATION TEST:
#                → fill in part_token_indices from results
#                → per-part noise init + temporal attraction both active

# Stage 3 LATER: 
#                → swap DEFAULT_PROSTHETIC_STRESS for real FEA values
#                → no other code changes needed
               
    @synchronize_timer('Encode cond')
    def encode_cond(self, image, text, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
        bsz = image.shape[0]
        cond = self.conditioner(image=image, text=text, **additional_cond_inputs)
        #cond['text'] = self.text_conditioner(text)

        # ← TODO clear
        # logger.info(" --- --- --- Start of Debbuging prints  --- --- --- ")
        # print("cond type:", type(cond))
        # if isinstance(cond, dict):
        #     print("cond keys:", list(cond.keys()))
        #     for k, v in cond.items():
        #         if isinstance(v, torch.Tensor):
        #             print(f"  {k:12}: {tuple(v.shape)}")
        #         elif isinstance(v, (list, tuple)):
        #             print(f"  {k:12}: list/tuple of len {len(v)}")
        #         else:
        #             print(f"  {k:12}: {type(v)}")
        # elif isinstance(cond, torch.Tensor):
        #     print("cond is tensor:", tuple(cond.shape))
        # else:
        #     print("cond is something else:", type(cond))
        # logger.info(" --- --- --- End of Debbuging prints  --- --- --- ")
        # ... rest of the function unchanged ...

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            if dual_guidance:
                un_cond_drop_main = copy.deepcopy(un_cond)
                un_cond_drop_main['additional'] = cond['additional']

                def cat_recursive(a, b, c):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b, c], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k], c[k])
                    return out

                cond = cat_recursive(cond, un_cond_drop_main, un_cond)
            else:
                def cat_recursive(a, b):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k])
                    return out

                cond = cat_recursive(cond, un_cond)
                
        return cond

    def prepare_prior(self, prior, debug_img_dir = None):
        from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
        if prior:
            logger.info(f'we pass self.vae.encoder.pc_size eq {self.vae.encoder.pc_size}')
            print(f"[INFO PRINT] Dubb the logger - {self.vae.encoder.pc_size}")

            loader = SharpEdgeSurfaceLoader(
            num_sharp_points=0,
            num_uniform_points=81920, #based on ShapeVAE
            )
            # mesh_demo = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
            # surface = loader(mesh_demo).to('cuda', dtype=torch.float16)
            # surface[:, :, :3] = surface[:, :, :3] * 0.8 # normalize the cube to [-0.8, 0.8]
            #TODO DO
            surface = loader(prior).to('cuda', dtype=torch.float16)
            print("[INFO] loaded surface shape:", surface.shape)
            try:
                if debug_img_dir is not None:
                    pcdd = self.surface_tensor_to_pcd(surface)
                    print(pcdd)
                    # self.save_pc_image(debug_img_dir, point_cloud_data = pcdd) 
                    self.save_pc_image_matplotlib(debug_img_dir, point_cloud_data = pcdd) 
            except Exception as e: 
                logger.error(f"Debug image creation: {e}")
                pass 
            
            # vae = ShapeVAE.from_pretrained(
            #     'tencent/Hunyuan3D-2.1',
            #     use_safetensors=False,
            #     variant='fp16',
            # )
            # shape
           

            with torch.no_grad(): 
                if isinstance(self.vae, ShapeVAE):
                    latents = self.vae.encode(surface)        
                    logger.info("latents encoded")
                else:
                    logger.error("NO SHAPEVAE")
                    
                    # latents = vae.encode(surface)
                    # logger.info("surface shape encoded.")
            # shape = (batch_size, *self.vae.latent_shape) #batch_size  = 1 (img count) and unpacked 512, 64
            
            return latents
        else:
            logger.error("No PRIOR Passed")
            return None
            
    def surface_tensor_to_pcd(self, surface: torch.Tensor) -> o3d.geometry.PointCloud:
        surface_np = surface.detach().cpu().float().numpy()

       # xyz = surface_np[:, :3]
        xyz = surface_np[0, :, :3]  

        # Sanity checks
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Invalid XYZ shape: {xyz.shape}")

        if not np.isfinite(xyz).all():
            raise ValueError("XYZ contains NaNs or Infs")

        xyz = xyz.astype(np.float64)

        z = xyz[:, 2]
        z = xyz[:, 2]
        z_scale = np.ptp(z)

        z_norm = (z - z.min()) / (z_scale + 1e-8)

        # scale = np.ptp(xyz, axis=0)

        # z_norm = (z - z.min()) / (scale + 1e-8)

        colors = np.stack(
            [z_norm, np.zeros_like(z_norm), 1.0 - z_norm],
            axis=1
        ).astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def save_pc_image(
        self,
        out_path: str,
        point_cloud_data = None   
        ):

        """
        Render point cloud offscreen and save image to file.
        Works on SLURM / headless nodes.
        """
        width: int = 1024
        height: int = 1024
        point_size: float = 2.0
        bg_color=(1.0, 1.0, 1.0)

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            visible=False,
            # width=width,
            # height=height,
        )
        success = vis.create_window(visible=False, width=width, height=height)
    
        if not success:
            logger.error("The windows was not created Open3D. Check for EGL or Xvfb.")
            return

        vis.add_geometry(point_cloud_data)

        render_opt = vis.get_render_option()
        if render_opt is None:
            logger.error("RenderOption is None")
            vis.destroy_window()
            return
    
        render_opt.background_color = np.array(bg_color)
        render_opt.point_size = point_size

        vis.poll_events()
        vis.update_renderer()

        # Save image
        out_path = os.path.join(out_path,"debug_image.png")
        vis.capture_screen_image(out_path)
        vis.destroy_window()

    def save_pc_image_matplotlib(self, out_path, point_cloud_data):
        import matplotlib.pyplot as plt
        #rom mpl_toolkits.mplot3d import Axes3D
        """
        Зберігає зображення хмари точок без використання GPU/Open3D.
        Ідеально для SLURM.
        """
        # 1. Отримуємо точки (convert Open3D pcd to numpy)
        points = np.asarray(point_cloud_data.points)
        
        # Якщо у точок є кольори, дістаємо і їх
        colors = None
        if point_cloud_data.has_colors():
            colors = np.asarray(point_cloud_data.colors)

        # 2. Створюємо фігуру
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 3. Малюємо Scatter Plot
        # s - розмір точки, c - колір
        if colors is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=colors)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c='blue')

        # 4. Налаштовуємо вигляд (щоб не було осей)
        ax.set_axis_off()
        
        # Можна налаштувати кут огляду
        ax.view_init(elev=20, azim=45)

        # 5. Зберігаємо
        os.makedirs(out_path, exist_ok=True)
        final_path = os.path.join(out_path, "debug_image_plt.png")
        
        # savefig працює в headless режимі без проблем
        plt.savefig(final_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig) # Обов'язково закриваємо, щоб не текла пам'ять
        print(f"Дебаг-зображення збережено через Matplotlib: {final_path}")

    def prepare_latents_with_prior_mechstyle(
    self,
    z_prior,
    part_stress: dict,        # from FEA or heuristic
    part_token_indices: dict, # which latent tokens = which part
    batch_size, dtype, device, generator,
    ):
        """
        MechStyle's key insight applied to latent space:
        high-stress parts get low noise (preserved),
        low-stress parts get high noise (free to restyle).
        
        This is their "selectively frozen" strategy, which they found
        most effective for structural integrity + style quality.
        """
        shape = (batch_size, *self.vae.latent_shape)
        base_noise = randn_tensor(shape, generator=generator,
                                device=device, dtype=dtype)
        
        latents = z_prior.clone().to(device=device, dtype=dtype)
        
        for part_name, stress in part_stress.items():
            idx = part_token_indices[part_name]
            
            # stress=1.0 → alpha=1.0 → pure prior (frozen)
            # stress=0.0 → alpha=0.0 → pure noise (free)
            alpha = stress  # directly from FEA stress value
            
            latents[:, idx, :] = (
                alpha * z_prior[:, idx, :] +
                np.sqrt(1 - alpha**2) * base_noise[:, idx, :]
                # same spherical interpolation as your existing code
            )
        
        return latents

    def prepare_latents_with_prior(
        self,
        z_prior: torch.FloatTensor,
        batch_size: int,
        dtype,
        device,
        generator,
        alpha: float = 0.7,
    ):
        """
        z0 = alpha * z_prior + (1 - alpha) * noise
        """
        shape = (batch_size, *self.vae.latent_shape) #batch_size  = 1 (img count) and unpacked 512, 64
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        assert z_prior.shape == latents.shape

        # scale noise as scheduler expects
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)

        z_prior = z_prior.to(device=device, dtype=dtype)
        
        latents  = alpha * z_prior + np.sqrt(1 - alpha**2) * latents 

        #latents = alpha * z_prior + (1.0 - alpha) * noise
        return latents



class Hunyuan3DDiTFlowMatchingTextPipeline(Hunyuan3DDiTPipeline):

    def __init__(self, vae, model, scheduler, conditioner, image_processor, device='cuda', dtype=torch.float16, **kwargs):
        super().__init__(vae, model, scheduler, conditioner, image_processor, device, dtype, **kwargs)
        
        default_subfolder='hunyuan3d-vae-v2-mini-withencoder'
        model_path = 'tencent/Hunyuan3D-2mini'
        
        self.vae = ShapeVAE.from_pretrained(
            model_path,
            subfolder = default_subfolder,
            use_safetensors=False,
            variant='fp16',
        )
    
    @torch.inference_mode()
    def __call__(
        self,
        prompt,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        mask = None,
         # NEW — stress weights per part, optional
        part_stress: Optional[dict] = None,  
        part_token_indices: Optional[dict] = None,  # from your ablation test
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
       
        
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo) # 

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        cond = self.encode_textcond(
            text=prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )

        batch_size = 1

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        
        prior = kwargs.pop("prior", None)
        
        z_prior = self.prepare_prior(prior=prior, debug_img_dir=kwargs.pop("debug_img_dir", None)) if prior else None
        alpha = kwargs.pop("alpha", None) ## TODO Add to params
    
        latents = self.prepare_latents_with_prior( #TODO check
            z_prior=z_prior,
            batch_size=batch_size, #image.shape[0]
            dtype=dtype,
            device=device,
            generator=generator,
            alpha=alpha, #0.8 init
        )

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
            # logger.info(f'Using guidance embed with scale {guidance_scale}')

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample
                
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )
        
    def encode_textcond(self, text, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
        
        cond = self.conditioner(text=text, **additional_cond_inputs)
 
        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(**additional_cond_inputs)

            if dual_guidance:
                un_cond_drop_main = copy.deepcopy(un_cond)
                un_cond_drop_main['additional'] = cond['additional']

                def cat_recursive(a, b, c):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b, c], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k], c[k])
                    return out

                cond = cat_recursive(cond, un_cond_drop_main, un_cond)
            else:
                def cat_recursive(a, b):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k])
                    return out

                cond = cat_recursive(cond, un_cond)
                
        return cond
         
    def _apply_stress_attraction(
        self,
        latents: torch.Tensor,
        z_prior: torch.Tensor,
        part_stress: dict,
        part_token_indices: dict,
        t_progress: float,          # 0.0 (start) → 1.0 (end)
    ) -> torch.Tensor:
        """
        MechStyle temporal scheduling:
        - Early steps: strong pull toward prior for all parts
        - Late steps: only high-stress parts stay attracted
        
        High stress (cavity=0.95): attraction persists whole denoising
        Low stress  (shell=0.10):  attraction fades quickly → style emerges
        """
        latents = latents.clone()
        
        for part_name, stress in part_stress.items():
            if part_name not in part_token_indices:
                continue
                
            idx = part_token_indices[part_name]
            
            # Stress controls how long attraction persists
            # stress=0.95: still 0.95 attraction at t_progress=0.9
            # stress=0.10: already ~0.0 attraction at t_progress=0.3
            attraction = stress * max(0.0, 1.0 - t_progress / stress)
            
            if attraction > 0:
                latents[:, idx, :] = (
                    latents[:, idx, :] +
                    attraction * (z_prior[:, idx, :] - latents[:, idx, :])
                )
        
        return latents
## Why `part_token_indices=None` is fine for now

# The design is staged — everything works at each stage:
# ```
# Stage 1 NOW:   part_token_indices=None
#                → stress attraction skipped (None guard)  
#                → alpha still applied globally via prepare_latents_with_prior
#                → run your ablation test

# Stage 2 AFTER ABLATION TEST:
#                → fill in part_token_indices from results
#                → per-part noise init + temporal attraction both active

# Stage 3 LATER: 
#                → swap DEFAULT_PROSTHETIC_STRESS for real FEA values
#                → no other code changes needed
               
    @synchronize_timer('Encode cond')
    def encode_cond(self, image, text, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
        bsz = image.shape[0]
        cond = self.conditioner(image=image, text=text, **additional_cond_inputs)
        #cond['text'] = self.text_conditioner(text)

        # ← TODO clear
        # logger.info(" --- --- --- Start of Debbuging prints  --- --- --- ")
        # print("cond type:", type(cond))
        # if isinstance(cond, dict):
        #     print("cond keys:", list(cond.keys()))
        #     for k, v in cond.items():
        #         if isinstance(v, torch.Tensor):
        #             print(f"  {k:12}: {tuple(v.shape)}")
        #         elif isinstance(v, (list, tuple)):
        #             print(f"  {k:12}: list/tuple of len {len(v)}")
        #         else:
        #             print(f"  {k:12}: {type(v)}")
        # elif isinstance(cond, torch.Tensor):
        #     print("cond is tensor:", tuple(cond.shape))
        # else:
        #     print("cond is something else:", type(cond))
        # logger.info(" --- --- --- End of Debbuging prints  --- --- --- ")
        # ... rest of the function unchanged ...

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            if dual_guidance:
                un_cond_drop_main = copy.deepcopy(un_cond)
                un_cond_drop_main['additional'] = cond['additional']

                def cat_recursive(a, b, c):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b, c], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k], c[k])
                    return out

                cond = cat_recursive(cond, un_cond_drop_main, un_cond)
            else:
                def cat_recursive(a, b):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k])
                    return out

                cond = cat_recursive(cond, un_cond)
                
        return cond

    def prepare_prior(self, prior, debug_img_dir = None):
        from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
        if prior:
            logger.info(f'we pass self.vae.encoder.pc_size eq {self.vae.encoder.pc_size}')
            print(f"[INFO PRINT] Dubb the logger - {self.vae.encoder.pc_size}")

            loader = SharpEdgeSurfaceLoader(
            num_sharp_points=0,
            num_uniform_points=81920, #based on ShapeVAE
            )
            # mesh_demo = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
            # surface = loader(mesh_demo).to('cuda', dtype=torch.float16)
            # surface[:, :, :3] = surface[:, :, :3] * 0.8 # normalize the cube to [-0.8, 0.8]
            #TODO DO
            surface = loader(prior).to('cuda', dtype=torch.float16)
            print("[INFO] loaded surface shape:", surface.shape)
            try:
                if debug_img_dir is not None:
                    pcdd = self.surface_tensor_to_pcd(surface)
                    print(pcdd)
                    # self.save_pc_image(debug_img_dir, point_cloud_data = pcdd) 
                    self.save_pc_image_matplotlib(debug_img_dir, point_cloud_data = pcdd) 
            except Exception as e: 
                logger.error(f"Debug image creation: {e}")
                pass 
            
            # vae = ShapeVAE.from_pretrained(
            #     'tencent/Hunyuan3D-2.1',
            #     use_safetensors=False,
            #     variant='fp16',
            # )
            # shape
           

            with torch.no_grad(): 
                if isinstance(self.vae, ShapeVAE):
                    latents = self.vae.encode(surface)        
                    logger.info("latents encoded")
                else:
                    logger.error("NO SHAPEVAE")
                    
                    # latents = vae.encode(surface)
                    # logger.info("surface shape encoded.")
            # shape = (batch_size, *self.vae.latent_shape) #batch_size  = 1 (img count) and unpacked 512, 64
            
            return latents
        else:
            logger.error("No PRIOR Passed")
            return None
            
    def surface_tensor_to_pcd(self, surface: torch.Tensor) -> o3d.geometry.PointCloud:
        surface_np = surface.detach().cpu().float().numpy()

       # xyz = surface_np[:, :3]
        xyz = surface_np[0, :, :3]  

        # Sanity checks
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Invalid XYZ shape: {xyz.shape}")

        if not np.isfinite(xyz).all():
            raise ValueError("XYZ contains NaNs or Infs")

        xyz = xyz.astype(np.float64)

        z = xyz[:, 2]
        z = xyz[:, 2]
        z_scale = np.ptp(z)

        z_norm = (z - z.min()) / (z_scale + 1e-8)

        # scale = np.ptp(xyz, axis=0)

        # z_norm = (z - z.min()) / (scale + 1e-8)

        colors = np.stack(
            [z_norm, np.zeros_like(z_norm), 1.0 - z_norm],
            axis=1
        ).astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def save_pc_image(
        self,
        out_path: str,
        point_cloud_data = None   
        ):

        """
        Render point cloud offscreen and save image to file.
        Works on SLURM / headless nodes.
        """
        width: int = 1024
        height: int = 1024
        point_size: float = 2.0
        bg_color=(1.0, 1.0, 1.0)

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            visible=False,
            # width=width,
            # height=height,
        )
        success = vis.create_window(visible=False, width=width, height=height)
    
        if not success:
            logger.error("The windows was not created Open3D. Check for EGL or Xvfb.")
            return

        vis.add_geometry(point_cloud_data)

        render_opt = vis.get_render_option()
        if render_opt is None:
            logger.error("RenderOption is None")
            vis.destroy_window()
            return
    
        render_opt.background_color = np.array(bg_color)
        render_opt.point_size = point_size

        vis.poll_events()
        vis.update_renderer()

        # Save image
        out_path = os.path.join(out_path,"debug_image.png")
        vis.capture_screen_image(out_path)
        vis.destroy_window()

    def save_pc_image_matplotlib(self, out_path, point_cloud_data):
        import matplotlib.pyplot as plt
        #rom mpl_toolkits.mplot3d import Axes3D
        """
        Зберігає зображення хмари точок без використання GPU/Open3D.
        Ідеально для SLURM.
        """
        # 1. Отримуємо точки (convert Open3D pcd to numpy)
        points = np.asarray(point_cloud_data.points)
        
        # Якщо у точок є кольори, дістаємо і їх
        colors = None
        if point_cloud_data.has_colors():
            colors = np.asarray(point_cloud_data.colors)

        # 2. Створюємо фігуру
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 3. Малюємо Scatter Plot
        # s - розмір точки, c - колір
        if colors is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=colors)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c='blue')

        # 4. Налаштовуємо вигляд (щоб не було осей)
        ax.set_axis_off()
        
        # Можна налаштувати кут огляду
        ax.view_init(elev=20, azim=45)

        # 5. Зберігаємо
        os.makedirs(out_path, exist_ok=True)
        final_path = os.path.join(out_path, "debug_image_plt.png")
        
        # savefig працює в headless режимі без проблем
        plt.savefig(final_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig) # Обов'язково закриваємо, щоб не текла пам'ять
        print(f"Дебаг-зображення збережено через Matplotlib: {final_path}")

    def prepare_latents_with_prior_mechstyle(
    self,
    z_prior,
    part_stress: dict,        # from FEA or heuristic
    part_token_indices: dict, # which latent tokens = which part
    batch_size, dtype, device, generator,
    ):
        """
        MechStyle's key insight applied to latent space:
        high-stress parts get low noise (preserved),
        low-stress parts get high noise (free to restyle).
        
        This is their "selectively frozen" strategy, which they found
        most effective for structural integrity + style quality.
        """
        shape = (batch_size, *self.vae.latent_shape)
        base_noise = randn_tensor(shape, generator=generator,
                                device=device, dtype=dtype)
        
        latents = z_prior.clone().to(device=device, dtype=dtype)
        
        for part_name, stress in part_stress.items():
            idx = part_token_indices[part_name]
            
            # stress=1.0 → alpha=1.0 → pure prior (frozen)
            # stress=0.0 → alpha=0.0 → pure noise (free)
            alpha = stress  # directly from FEA stress value
            
            latents[:, idx, :] = (
                alpha * z_prior[:, idx, :] +
                np.sqrt(1 - alpha**2) * base_noise[:, idx, :]
                # same spherical interpolation as your existing code
            )
        
        return latents

    def prepare_latents_with_prior(
        self,
        z_prior: torch.FloatTensor,
        batch_size: int,
        dtype,
        device,
        generator,
        alpha: float = 0.7,
    ):
        """
        z0 = alpha * z_prior + (1 - alpha) * noise
        """
        shape = (batch_size, *self.vae.latent_shape) #batch_size  = 1 (img count) and unpacked 512, 64
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        assert z_prior.shape == latents.shape

        # scale noise as scheduler expects
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)

        z_prior = z_prior.to(device=device, dtype=dtype)
        
        latents  = alpha * z_prior + np.sqrt(1 - alpha**2) * latents 

        #latents = alpha * z_prior + (1.0 - alpha) * noise
        return latents



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    return str(seed)

class HunyuanDiTPipeline:
    def __init__(
        self,
        model_path="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        device='cuda'
    ):
        self.device = device
        self.pipe = AutoPipelineForText2Image.from_pretrained(
           model_path,
           torch_dtype=torch.float16,
           enable_pag=True,
           pag_applied_layers=["blocks.(16|17|18|19)"]
        ).to(device)
        self.seed_toprint = "0"
        #self.pipe = DiffusionPipeline.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda", enable_pag=True)

        self.pos_txt = ",白色背景,3D风格,最佳质量"
        self.neg_txt = "文本,特写,裁剪,出框,最差质量,低质量,JPEG伪影,PGLY,重复,病态," \
                       "残缺,多余的手指,变异的手,画得不好的手,画得不好的脸,变异,畸形,模糊,脱水,糟糕的解剖学," \
                       "糟糕的比例,多余的肢体,克隆的脸,毁容,恶心的比例,畸形的肢体,缺失的手臂,缺失的腿," \
                       "额外的手臂,额外的腿,融合的手指,手指太多,长脖子"

    def compile(self):
        # accelarate hunyuan-dit transformer,first inference will cost long time
        torch.set_float32_matmul_precision('high')
        self.pipe.transformer = torch.compile(self.pipe.transformer, fullgraph=True)
        # self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, fullgraph=True)
        generator = torch.Generator(device=self.pipe.device)  # infer once for hot-start
        out_img = self.pipe(
            prompt='美少女战士',
            negative_prompt='模糊',
            num_inference_steps=25,
            pag_scale=1.3,
            width=1024,
            height=1024,
            generator=generator,
            return_dict=False
        )[0][0]

    @torch.no_grad()
    def __call__(self, prompt, seed=0):
        self.seed_toprint = seed_everything(seed)
        generator = torch.Generator(device=self.pipe.device)
        generator = generator.manual_seed(int(seed))
        out_img = self.pipe(
            prompt=prompt[:60] + self.pos_txt,
            negative_prompt=self.neg_txt,
            num_inference_steps=25,
            pag_scale=1.3,
            width=1024,
            height=1024,
            generator=generator,
            return_dict=False
        )[0][0]
        return out_img
