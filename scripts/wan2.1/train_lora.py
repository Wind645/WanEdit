"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import json
import contextlib
import gc
import logging
import math
import os
import pickle
import shutil
import sys

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
try:
    from accelerate.utils import DeepSpeedPlugin
except Exception:
    DeepSpeedPlugin = None
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
from videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                           ASPECT_RATIO_RANDOM_CROP_512,
                                           ASPECT_RATIO_RANDOM_CROP_PROB,
                                           CUSTOM_ASPECT_RATIOS,
                                           AspectRatioBatchImageVideoSampler,
                                           RandomSampler, get_closest_ratio)

# 为自定义分辨率创建均匀概率分布
CUSTOM_ASPECT_RATIO_PROB = np.array([1.0] * len(CUSTOM_ASPECT_RATIOS)) / len(CUSTOM_ASPECT_RATIOS)
from videox_fun.data.dataset_image_video import (ImageVideoDataset,
                                                ImageVideoSampler,
                                                VideoEditDataset,
                                                VideoEditReasoningDataset,
                                                get_random_mask)
from videox_fun.data.singleturn_dataset import (CachedSingleTurnLatentDataset,
                                                SingleTurnEditDataset)
from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                              WanTransformer3DModel)
from videox_fun.pipeline import WanPipeline
try:
    from videox_fun.pipeline import WanI2VPipeline
except ImportError:
    class WanI2VPipeline:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "WanI2VPipeline is not included in the minimal VideoCoF training release. "
                "Use train_mode='normal' unless you copy the I2V pipeline implementation."
            )
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.lora_utils import create_network, merge_lora, unmerge_lora
from videox_fun.utils.singleturn_utils import (CORNE_SINGLETURN_PROMPT,
                                               build_singleturn_object_removal_latents,
                                               build_singleturn_loss_mask_like,
                                               compute_singleturn_object_removal_total_frames,
                                               compute_wan_seq_len_from_latents,
                                               generate_singleturn_sample,
                                               normalize_singleturn_sample_size,
                                               prepare_singleturn_noisy_latents,
                                               preprocess_singleturn_image,
                                               preprocess_singleturn_mask_frame,
                                               resize_singleturn_mask_to_latent_grid,
                                               SINGLETURN_REFINEMENT_FIXED_PREFIX_FRAMES,
                                               SINGLETURN_SOURCE_CONDITION_FRAME_INDEX,
                                               SINGLETURN_TAIL_START,
                                               save_singleturn_outputs)
from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid

if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def resolve_model_path(model_root, subpath, default_subpath):
    subpath = str(subpath or default_subpath)
    if os.path.isabs(subpath):
        return subpath

    model_root_candidate = os.path.join(model_root, subpath)
    if os.path.exists(model_root_candidate):
        return model_root_candidate

    sibling_candidate = os.path.join(os.path.dirname(os.path.abspath(model_root)), subpath)
    if os.path.exists(sibling_candidate):
        return sibling_candidate

    return subpath


class CachedVideoLatentDataset(Dataset):
    def __init__(self, manifest_path, data_root=None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("Cached data manifest must be a non-empty list.")
        self.data_root = data_root
        self.manifest = manifest

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        entry = self.manifest[index]
        cache_path = entry["cache_path"]
        if self.data_root is not None and not os.path.isabs(cache_path):
            cache_path = os.path.join(self.data_root, cache_path)
        payload = load_cache_payload(cache_path)
        missing = [
            key
            for key in ("full_latents", "prompt_embeds", "prompt_seq_len")
            if key not in payload
        ]
        if missing:
            raise ValueError(f"Cached sample {cache_path} is missing keys: {missing}")
        return {
            "full_latents": payload["full_latents"],
            "prompt_embeds": payload["prompt_embeds"],
            "prompt_seq_len": int(payload["prompt_seq_len"]),
            "text": payload.get("text", ""),
            "data_type": "video",
            "idx": index,
        }


def load_cache_payload(cache_path):
    try:
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(cache_path, map_location="cpu")


def load_cached_batch(batch, weight_dtype, device):
    latents = batch["full_latents"].to(device=device, dtype=weight_dtype, non_blocking=True)
    prompt_embeds_padded = batch["prompt_embeds"].to(device=device, dtype=weight_dtype, non_blocking=True)
    prompt_seq_lens = batch["prompt_seq_len"].tolist()
    prompt_embeds = [embed[:seq_len] for embed, seq_len in zip(prompt_embeds_padded, prompt_seq_lens)]
    return latents, prompt_embeds


def load_cached_singleturn_batch(batch, weight_dtype, device, refinement_mode=False):
    latent_key = "input_latents" if refinement_mode else "full_latents"
    full_latents = batch[latent_key].to(device=device, dtype=weight_dtype, non_blocking=True)
    prompt_embeds_padded = batch["prompt_embeds"].to(device=device, dtype=weight_dtype, non_blocking=True)
    prompt_seq_lens = batch["prompt_seq_len"].tolist()
    prompt_embeds = [embed[:seq_len] for embed, seq_len in zip(prompt_embeds_padded, prompt_seq_lens)]
    target_latents = None
    refinement_loss_weight_map = None
    if refinement_mode:
        target_latents = batch["target_latents"].to(device=device, dtype=weight_dtype, non_blocking=True)
        refinement_loss_weight_map = batch["refinement_loss_weight_map"].to(
            device=device,
            dtype=weight_dtype,
            non_blocking=True,
        )
    text_split_point = None
    if "text_split_point" in batch:
        tsp_val = int(batch["text_split_point"][0].item())
        if tsp_val >= 0:
            text_split_point = tsp_val
    return full_latents, prompt_embeds, target_latents, refinement_loss_weight_map, text_split_point


def load_singleturn_shared_prompt_cache(prompt_cache_path, weight_dtype, device):
    try:
        payload = torch.load(prompt_cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(prompt_cache_path, map_location="cpu")
    missing = [key for key in ("prompt_embeds", "prompt_seq_len") if key not in payload]
    if missing:
        raise ValueError(f"SingleTurn shared prompt cache {prompt_cache_path} is missing keys: {missing}")
    prompt_embeds = payload["prompt_embeds"].to(device=device, dtype=weight_dtype, non_blocking=True)
    if prompt_embeds.ndim != 2:
        raise ValueError(
            f"Shared prompt_embeds must have shape (seq_len, hidden_dim), got {tuple(prompt_embeds.shape)}"
        )
    prompt_seq_len = int(payload["prompt_seq_len"])
    return {
        "prompt_embeds": prompt_embeds[:prompt_seq_len],
        "prompt_seq_len": prompt_seq_len,
        "text": payload.get("text", ""),
        "formatted_text": payload.get("formatted_text", payload.get("text", "")),
    }

def get_random_downsample_ratio(sample_size, image_ratio=[],
                                all_choices=False, rng=None):
    def _create_special_list(length):
        if length == 1:
            return [1.0]
        if length >= 2:
            first_element = 0.75
            remaining_sum = 1.0 - first_element
            other_elements_value = remaining_sum / (length - 1)
            special_list = [first_element] + [other_elements_value] * (length - 1)
            return special_list
            
    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio 
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]

    if all_choices:
        return number_list

    number_list_prob = np.array(_create_special_list(len(number_list)))
    if rng is None:
        return np.random.choice(number_list, p = number_list_prob)
    else:
        return rng.choice(number_list, p = number_list_prob)

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def wandb_reporting_enabled(args):
    return is_wandb_available() and args.report_to in {"wandb", "all"}


def resolve_singleturn_sample_size(args):
    sample_size = args.singleturn_sample_size if args.singleturn_sample_size is not None else args.video_sample_size
    return normalize_singleturn_sample_size(sample_size)


def log_singleturn_validation_to_wandb(args, global_step, source_image_path, formatted_prompt, output_paths):
    if not wandb_reporting_enabled(args):
        return

    wandb_payload = {
        "singleturn_validation/source": wandb.Image(source_image_path, caption="source image"),
        "singleturn_validation/full_last_frame": wandb.Image(
            output_paths["full_last_frame"],
            caption=formatted_prompt,
        ),
        "singleturn_validation/tail_last_frame": wandb.Image(
            output_paths["tail_last_frame"],
            caption=formatted_prompt,
        ),
        "singleturn_validation/full_sequence": wandb.Video(
            output_paths["full_video"],
            fps=args.singleturn_validation_fps,
            format="mp4",
        ),
        "singleturn_validation/tail_sequence": wandb.Video(
            output_paths["tail_video"],
            fps=args.singleturn_validation_fps,
            format="mp4",
        ),
    }
    wandb.log(wandb_payload, step=global_step)

def log_validation(vae, text_encoder, tokenizer, clip_image_encoder, transformer3d, network, config, args, accelerator, weight_dtype, global_step):
    try:
        if getattr(args, "singleturn_mode", False):
            if getattr(args, "singleturn_refine_mode", False):
                logger.info("Skipping SingleTurn refinement validation because the validation hook only supports coarse generation.")
                return
            if not args.singleturn_validation_image_path or not args.singleturn_validation_mask_path:
                logger.info("Skipping SingleTurn validation because no validation image/mask pair was provided.")
                return

            logger.info("Running SingleTurn validation...")
            transformer3d_val = WanTransformer3DModel.from_pretrained(
                resolve_model_path(
                    args.pretrained_model_name_or_path,
                    config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer'),
                    'transformer',
                ),
                transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            ).to(weight_dtype)
            transformer3d_val.load_state_dict(accelerator.unwrap_model(transformer3d).state_dict())
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
            )

            pipeline = WanPipeline(
                vae=accelerator.unwrap_model(vae).to(weight_dtype),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=transformer3d_val,
                scheduler=scheduler,
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline = merge_lora(
                pipeline,
                None,
                1,
                accelerator.device,
                state_dict=accelerator.unwrap_model(network).state_dict(),
                transformer_only=True,
            )

            generator = None
            if args.seed is not None:
                validation_seed = args.singleturn_validation_seed if args.singleturn_validation_seed is not None else args.seed
                generator = torch.Generator(device=accelerator.device).manual_seed(validation_seed)

            sample_size = resolve_singleturn_sample_size(args)
            source_tensor = preprocess_singleturn_image(
                args.singleturn_validation_image_path,
                sample_size,
            ).to(device=accelerator.device, dtype=weight_dtype)
            mask_frame_tensor = preprocess_singleturn_mask_frame(
                args.singleturn_validation_mask_path,
                sample_size,
            ).to(device=accelerator.device, dtype=weight_dtype)

            with torch.no_grad():
                generation = generate_singleturn_sample(
                    pipeline=pipeline,
                    mask_frame_tensor=mask_frame_tensor,
                    source_tensor=source_tensor,
                    negative_prompt=args.singleturn_validation_negative_prompt,
                    guidance_scale=args.singleturn_validation_guidance_scale,
                    num_inference_steps=args.singleturn_validation_num_inference_steps,
                    generator=generator,
                    weight_dtype=weight_dtype,
                    total_frames=compute_singleturn_object_removal_total_frames(
                        args.singleturn_cache_corruption_frames,
                        args.singleturn_cache_restoration_frames,
                    ),
                )

            output_paths = save_singleturn_outputs(
                full_frames=generation["full_frames"],
                tail_frames=generation["tail_frames"],
                output_dir=os.path.join(args.output_dir, "sample"),
                stem=f"singleturn-step-{global_step}",
                fps=args.singleturn_validation_fps,
            )
            log_singleturn_validation_to_wandb(
                args=args,
                global_step=global_step,
                source_image_path=args.singleturn_validation_image_path,
                formatted_prompt=generation["formatted_prompt"],
                output_paths=output_paths,
            )

            del pipeline
            del transformer3d_val
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            return

        logger.info("Running validation... ")

        transformer3d_val = WanTransformer3DModel.from_pretrained(
            resolve_model_path(
                args.pretrained_model_name_or_path,
                config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer'),
                'transformer',
            ),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)
        transformer3d_val.load_state_dict(accelerator.unwrap_model(transformer3d).state_dict())
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        
        if args.train_mode != "normal":
            pipeline = WanI2VPipeline(
                vae=accelerator.unwrap_model(vae).to(weight_dtype), 
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=transformer3d_val,
                scheduler=scheduler,
                clip_image_encoder=clip_image_encoder,
            )
        else:
            pipeline = WanPipeline(
                vae=accelerator.unwrap_model(vae).to(weight_dtype), 
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=transformer3d_val,
                scheduler=scheduler,
            )
        pipeline = pipeline.to(accelerator.device)

        pipeline = merge_lora(
            pipeline, None, 1, accelerator.device, state_dict=accelerator.unwrap_model(network).state_dict(), transformer_only=True
        )

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        for i in range(len(args.validation_prompts)):
            with torch.no_grad():
                if args.train_mode != "normal":
                    with torch.autocast("cuda", dtype=weight_dtype):
                        video_length = int((args.video_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_sample_n_frames != 1 else 1
                        input_video, input_video_mask, _ = get_image_to_video_latent(None, None, video_length=video_length, sample_size=[args.video_sample_size, args.video_sample_size])
                        sample = pipeline(
                            args.validation_prompts[i],
                            num_frames = video_length,
                            negative_prompt = "bad detailed",
                            height      = args.video_sample_size,
                            width       = args.video_sample_size,
                            guidance_scale = 6.0,
                            generator   = generator,

                            video        = input_video,
                            mask_video   = input_video_mask,
                        ).videos
                        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                        save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-{i}.gif"))

                        video_length = 1
                        input_video, input_video_mask, _ = get_image_to_video_latent(None, None, video_length=video_length, sample_size=[args.video_sample_size, args.video_sample_size])
                        sample = pipeline(
                            args.validation_prompts[i],
                            num_frames = video_length,
                            negative_prompt = "bad detailed",
                            height      = args.video_sample_size,
                            width       = args.video_sample_size,
                            guidance_scale = 6.0,
                            generator   = generator, 

                            video        = input_video,
                            mask_video   = input_video_mask,
                        ).videos
                        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                        save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-image-{i}.gif"))
                else:
                    with torch.autocast("cuda", dtype=weight_dtype):
                        sample = pipeline(
                            args.validation_prompts[i], 
                            num_frames = args.video_sample_n_frames,
                            negative_prompt = "bad detailed",
                            height      = args.video_sample_size,
                            width       = args.video_sample_size,
                            generator   = generator
                        ).videos
                        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                        save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-{i}.gif"))

                        sample = pipeline(
                            args.validation_prompts[i], 
                            num_frames = 1,
                            negative_prompt = "bad detailed",
                            height      = args.video_sample_size,
                            width       = args.video_sample_size,
                            generator   = generator
                        ).videos
                        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                        save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-image-{i}.gif"))

        del pipeline
        del transformer3d_val
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error with info {e}")
        return None

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_manifest",
        type=str,
        default=None,
        help="Optional SingleTurn JSON/JSONL manifest with source_image, edited_image, and prompt records.",
    )
    parser.add_argument(
        "--cached_data_meta",
        type=str,
        default=None,
        help="JSON manifest for cached training samples.",
    )
    parser.add_argument(
        "--cached_data_dir",
        type=str,
        default=None,
        help="Optional root directory used to resolve relative cache_path entries.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--tracker_entity",
        type=str,
        default=None,
        help="Optional W&B entity/team name used when initializing experiment tracking.",
    )
    
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_init_path",
        type=str,
        default=None,
        help="Optional LoRA weights file used to initialize the trainable LoRA network without resuming optimizer/scheduler state.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--auto_tile_batch_size", action="store_true", help="Whether to auto tile batch size.",
    )
    parser.add_argument(
        "--noise_share_in_frames", action="store_true", help="Whether enable noise share in frames."
    )
    parser.add_argument(
        "--noise_share_in_frames_ratio", type=float, default=0.5, help="Noise share ratio.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true", 
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--fix_sample_size", 
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--video_edit_loss_on_edited_frames_only", action="store_true", help="Whether enable video edit loss on edited frames only.",
    )
    parser.add_argument(
        "--source_frames",
        type=int,
        default=9,
        help="Number of frames from the original video in VideoEditDataset.",
    )
    parser.add_argument(
        "--edit_frames",
        type=int,
        default=8,
        help="Number of frames from the edited video in VideoEditDataset.",
    )
    parser.add_argument(
        "--reasoning_frames",
        type=int,
        default=4,
        help="Number of grounded frames in VideoEditReasoningDataset.",
    )
    parser.add_argument(
        "--use_reasoning_dataset",
        action="store_true",
        default=False,
        help="Use VideoEditReasoningDataset (triplet: original/grounded/edited).",
    )
    parser.add_argument(
        "--singleturn_mode",
        action="store_true",
        help="Enable SingleTurn image-edit training with a dedicated image-pair dataset and 7-frame latent recipe.",
    )
    parser.add_argument(
        "--singleturn_refine_mode",
        action="store_true",
        help="Enable cached SingleTurn refinement training on coarse latent trajectories with a one-step tail correction target.",
    )
    parser.add_argument(
        "--singleturn_reconstruction_mode",
        action="store_true",
        help="Deprecated and unsupported for CORNE object-removal mode.",
    )
    parser.add_argument(
        "--singleturn_reconstruction_prompt_cache",
        type=str,
        default=None,
        help="Deprecated and unsupported for CORNE object-removal mode.",
    )
    parser.add_argument(
        "--singleturn_null_drop_prob",
        type=float,
        default=0.0,
        help="Deprecated and unsupported for CORNE object-removal mode.",
    )
    parser.add_argument(
        "--singleturn_sdedit_strength_min",
        type=float,
        default=None,
        help="Deprecated and unsupported for CORNE object-removal mode.",
    )
    parser.add_argument(
        "--singleturn_sdedit_strength_max",
        type=float,
        default=None,
        help="Deprecated and unsupported for CORNE object-removal mode.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=CORNE_SINGLETURN_PROMPT,
        help="Deprecated and ignored. Prompt is fixed for CORNE object-removal mode.",
    )
    parser.add_argument(
        "--singleturn_sample_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional non-square sample size for SingleTurn mode only.",
    )
    parser.add_argument(
        "--singleturn_cache_corruption_frames",
        type=int,
        default=2,
        help="Number of interpolated frames between source and noisy anchor when cached full_latents are absent.",
    )
    parser.add_argument(
        "--singleturn_cache_restoration_frames",
        type=int,
        default=5,
        help="Number of interpolated frames between noisy anchor and target when cached full_latents are absent.",
    )
    parser.add_argument(
        "--singleturn_cache_interpolation_gamma",
        type=float,
        default=2.0,
        help="Gamma for non-linear interpolation when cached full_latents are absent.",
    )
    parser.add_argument(
        "--singleturn_train_mask_sam_only",
        action="store_true",
        help="Use mask_sam_latent for the training mask condition frame instead of 50/50 mask_check/mask_sam.",
    )
    parser.add_argument(
        "--singleturn_validation_image_path",
        type=str,
        default=None,
        help="Optional validation source image used for SingleTurn media logging.",
    )
    parser.add_argument(
        "--singleturn_validation_prompt",
        type=str,
        default=None,
        help="Deprecated and ignored. Prompt is fixed for CORNE object-removal mode.",
    )
    parser.add_argument(
        "--singleturn_validation_mask_path",
        type=str,
        default=None,
        help="Optional validation mask path used for SingleTurn media logging.",
    )
    parser.add_argument(
        "--singleturn_validation_negative_prompt",
        type=str,
        default="",
        help="Negative prompt used for SingleTurn validation generation.",
    )
    parser.add_argument(
        "--singleturn_validation_guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale used for SingleTurn validation generation.",
    )
    parser.add_argument(
        "--singleturn_validation_num_inference_steps",
        type=int,
        default=50,
        help="Inference steps used for SingleTurn validation generation.",
    )
    parser.add_argument(
        "--singleturn_validation_seed",
        type=int,
        default=None,
        help="Optional validation seed override for SingleTurn validation generation.",
    )
    parser.add_argument(
        "--singleturn_validation_fps",
        type=int,
        default=4,
        help="GIF FPS used when saving and logging SingleTurn validation sequences.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")

    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="Path to DeepSpeed json config. If unset, will try env ACCELERATE_DEEPSPEED_CONFIG_FILE or defaults.",
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="normal",
        help=(
            'The format of training data. Support `"normal"`'
            ' (default), `"i2v"`.'
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--lora_skip_name",
        type=str,
        default=None,
        help=("The module is not trained in loras. "),
    )

    parser.add_argument(
        "--debug_shapes",
        action="store_true",
        help="Log input/latent shapes to verify final HxW and VAE compression.",
    )
    parser.add_argument(
        "--debug_log_interval",
        type=int,
        default=100,
        help="Log shapes every N global steps when --debug_shapes is enabled.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.singleturn_refine_mode and not args.singleturn_mode:
        raise ValueError("SingleTurn refinement mode requires --singleturn_mode.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    if args.singleturn_mode:
        if args.singleturn_refine_mode and args.cached_data_meta is None:
            raise ValueError("SingleTurn refinement mode requires --cached_data_meta.")
        if args.singleturn_refine_mode and args.train_data_dir is not None:
            raise ValueError("SingleTurn refinement mode uses cached latents only and does not accept --train_data_dir.")
        if args.singleturn_refine_mode and args.train_data_manifest is not None:
            raise ValueError("SingleTurn refinement mode does not accept --train_data_manifest.")
        if args.singleturn_reconstruction_mode:
            raise ValueError("CORNE object-removal SingleTurn mode no longer supports --singleturn_reconstruction_mode.")
        if args.singleturn_reconstruction_prompt_cache is not None:
            raise ValueError(
                "CORNE object-removal SingleTurn mode no longer supports --singleturn_reconstruction_prompt_cache."
            )
        if args.singleturn_null_drop_prob not in (0, 0.0):
            raise ValueError("CORNE object-removal SingleTurn mode no longer supports --singleturn_null_drop_prob.")
        if args.singleturn_sdedit_strength_min is not None or args.singleturn_sdedit_strength_max is not None:
            raise ValueError(
                "CORNE object-removal SingleTurn mode no longer supports --singleturn_sdedit_strength_min/max."
            )
        if args.singleturn_sample_size is not None:
            singleturn_sample_size = normalize_singleturn_sample_size(args.singleturn_sample_size)
            if any(dim % 16 != 0 for dim in singleturn_sample_size):
                raise ValueError(
                    "SingleTurn sample size must be divisible by 16 for Wan VAE compatibility, "
                    f"got {singleturn_sample_size}."
                )
        if args.train_data_meta is not None:
            raise ValueError("SingleTurn mode uses --train_data_manifest instead of --train_data_meta.")
        if args.cached_data_meta is not None and args.train_data_manifest is not None:
            raise ValueError("SingleTurn cached training uses only --cached_data_meta/--cached_data_dir, not --train_data_manifest.")
        if args.cached_data_meta is None and args.train_data_dir is None:
            raise ValueError("SingleTurn mode requires either --train_data_dir or --cached_data_meta.")
        if args.train_mode != "normal":
            raise ValueError("SingleTurn mode currently supports only --train_mode normal.")
        if args.use_reasoning_dataset:
            raise ValueError("SingleTurn mode has its own dataset and does not support --use_reasoning_dataset.")
        if args.enable_bucket:
            raise ValueError("SingleTurn mode does not support --enable_bucket.")
        if args.enable_text_encoder_in_dataloader:
            raise ValueError("SingleTurn mode does not support --enable_text_encoder_in_dataloader.")
        unsupported_flags = []
        for flag_name in (
            "random_frame_crop",
            "keep_all_node_same_token_length",
            "training_with_video_token_length",
            "auto_tile_batch_size",
        ):
            if getattr(args, flag_name):
                unsupported_flags.append(f"--{flag_name.replace('_', '-')}")
        if unsupported_flags:
            raise ValueError(
                "SingleTurn mode does not support the following options: "
                + ", ".join(unsupported_flags)
            )
    else:
        if args.train_data_meta is None and args.cached_data_meta is None:
            raise ValueError("Provide either --train_data_meta or --cached_data_meta.")
        if args.train_data_manifest is not None:
            raise ValueError("--train_data_manifest is only valid with --singleturn_mode.")
    if args.cached_data_meta is not None:
        if args.train_mode != "normal":
            raise ValueError("Cached training currently supports only --train_mode normal.")
        if args.enable_text_encoder_in_dataloader:
            raise ValueError("Cached training does not support --enable_text_encoder_in_dataloader.")
        if args.validation_prompts is not None:
            raise ValueError("Cached training does not support runtime validation prompts.")
        if args.low_vram:
            raise ValueError("Cached training does not use runtime VAE/T5, so --low_vram is not applicable.")

    if (args.singleturn_validation_image_path is None) != (args.singleturn_validation_mask_path is None):
        raise ValueError(
            "Provide both --singleturn_validation_image_path and --singleturn_validation_mask_path together for SingleTurn validation."
        )
    if args.singleturn_validation_image_path is not None and not os.path.exists(args.singleturn_validation_image_path):
        raise ValueError(f"SingleTurn validation image does not exist: {args.singleturn_validation_image_path}")
    if args.singleturn_validation_mask_path is not None and not os.path.exists(args.singleturn_validation_mask_path):
        raise ValueError(f"SingleTurn validation mask does not exist: {args.singleturn_validation_mask_path}")

    return args


def main():
    args = parse_args()
    use_cached_data = args.cached_data_meta is not None
    need_runtime_encoders = (not use_cached_data) or (
        args.singleturn_mode and args.singleturn_validation_image_path is not None
    )

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # Initialize DeepSpeed plugin (works for torchrun and accelerate CLI)
    configured_deepspeed_plugin = None
    try:
        use_ds_env = os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() in ("1", "true", "yes")
        want_ds = use_ds_env or getattr(args, "use_deepspeed", False)
        if want_ds and DeepSpeedPlugin is not None:
            # Priority: explicit CLI config > env > default
            ds_config_file = args.deepspeed_config or os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE", "config/zero_stage2_config.json")
            zero_stage_val = int(os.environ.get("ACCELERATE_ZERO_STAGE", "2"))
            try:
                # Newer accelerate supports file path directly
                configured_deepspeed_plugin = DeepSpeedPlugin(zero_stage=zero_stage_val, deepspeed_config_file=ds_config_file)
            except TypeError:
                # Older accelerate: load json to hf_ds_config
                with open(ds_config_file, "r") as f:
                    ds_cfg = json.load(f)
                configured_deepspeed_plugin = DeepSpeedPlugin(zero_stage=zero_stage_val, hf_ds_config=ds_cfg)
    except Exception:
        configured_deepspeed_plugin = None

    accelerator_kwargs = dict(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if configured_deepspeed_plugin is not None:
        accelerator_kwargs["deepspeed_plugin"] = configured_deepspeed_plugin

    accelerator = Accelerator(**accelerator_kwargs)

    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"Using DeepSpeed Zero stage: {zero_stage}")

        args.use_deepspeed = True
        if zero_stage == 3:
            print(f"Auto set save_state to True because zero_stage == 3")
            args.save_state = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None: # The fsdp_plugin.sharding_strategy is None in FSDP 2.
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"Using FSDP stage: {fsdp_stage}")

        args.use_fsdp = True
        if fsdp_stage == 3:
            print(f"Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed is not enabled.")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    singleturn_shared_prompt_cache = None

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    tokenizer = None
    text_encoder = None
    vae = None
    clip_image_encoder = None

    if need_runtime_encoders:
        tokenizer = AutoTokenizer.from_pretrained(
            resolve_model_path(
                args.pretrained_model_name_or_path,
                config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'),
                'tokenizer',
            ),
        )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    if need_runtime_encoders:
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            # Get Text encoder
            text_encoder = WanT5EncoderModel.from_pretrained(
                resolve_model_path(
                    args.pretrained_model_name_or_path,
                    config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder'),
                    'text_encoder',
                ),
                additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=weight_dtype,
            )
            text_encoder = text_encoder.eval()
            # Get Vae
            vae = AutoencoderKLWan.from_pretrained(
                resolve_model_path(
                    args.pretrained_model_name_or_path,
                    config['vae_kwargs'].get('vae_subpath', 'vae'),
                    'vae',
                ),
                additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
            )
            vae.eval()
            # Get Clip Image Encoder
            if args.train_mode != "normal":
                clip_image_encoder = CLIPModel.from_pretrained(
                    resolve_model_path(
                        args.pretrained_model_name_or_path,
                        config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder'),
                        'image_encoder',
                    ),
                )
                clip_image_encoder = clip_image_encoder.eval()
            
    # Get Transformer
    transformer3d = WanTransformer3DModel.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer'),
            'transformer',
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True, 
        torch_dtype=weight_dtype,
    ).to(weight_dtype)

    # Freeze vae and text_encoder and set transformer3d to trainable
    if vae is not None:
        vae.requires_grad_(False)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)
    if clip_image_encoder is not None:
        clip_image_encoder.requires_grad_(False)

    # Lora will work with this...
    zero3_ctx = deepspeed_plugin.zero3_init_context_manager() if deepspeed_plugin is not None else contextlib.nullcontext()
    with zero3_ctx:
        network = create_network(
            1.0,
            args.rank,
            args.network_alpha,
            None,
            transformer3d,
            neuron_dropout=None,
            skip_name=args.lora_skip_name,
        )
    network.apply_to(text_encoder, transformer3d, args.train_text_encoder and not args.training_with_video_token_length, True)

    if args.lora_init_path is not None:
        print(f"Initialize LoRA weights from: {args.lora_init_path}")
        if args.lora_init_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.lora_init_path)
        else:
            state_dict = torch.load(args.lora_init_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = network.load_state_dict(state_dict, strict=False)
        print(f"LoRA init loaded. missing keys: {len(m)}, unexpected keys: {len(u)}")

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if args.vae_path is not None and vae is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_sampler_state(output_dir):
            if batch_sampler is None or not hasattr(batch_sampler, "sampler") or not hasattr(batch_sampler.sampler, "_pos_start"):
                return
            with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

        def load_sampler_state(input_dir):
            if batch_sampler is None or not hasattr(batch_sampler, "sampler") or not hasattr(batch_sampler.sampler, "_pos_start"):
                return
            pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    loaded_number, _ = pickle.load(file)
                    batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file

                    safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                    network_state_dict = {}
                    for key in accelerate_state_dict:
                        if "network" in key:
                            network_state_dict[key.replace("network.", "")] = accelerate_state_dict[key].to(weight_dtype)

                    save_file(network_state_dict, safetensor_save_path, metadata={"format": "pt"})
                    save_sampler_state(output_dir)

            def load_model_hook(models, input_dir):
                load_sampler_state(input_dir)
        elif zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    save_sampler_state(output_dir)

            def load_model_hook(models, input_dir):
                load_sampler_state(input_dir)
        else:
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                    save_model(safetensor_save_path, accelerator.unwrap_model(models[-1]))
                    if not args.use_deepspeed:
                        for _ in range(len(weights)):
                            weights.pop()

                    save_sampler_state(output_dir)

            def load_model_hook(models, input_dir):
                load_sampler_state(input_dir)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    logging.info("Add network parameters")
    trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    trainable_params_optim = network.prepare_optimizer_params(args.learning_rate / 2, args.learning_rate, args.learning_rate)

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    batch_sampler = None
    sample_n_frames_bucket_interval = 4 if vae is None else vae.config.temporal_compression_ratio

    if use_cached_data:
        if args.singleturn_mode:
            expected_mode = "singleturn_object_removal_refine_v1" if args.singleturn_refine_mode else "singleturn_object_removal_cached"
            train_dataset = CachedSingleTurnLatentDataset(
                args.cached_data_meta,
                args.cached_data_dir,
                expected_mode=expected_mode,
                corruption_frames=args.singleturn_cache_corruption_frames,
                restoration_frames=args.singleturn_cache_restoration_frames,
                interpolation_gamma=args.singleturn_cache_interpolation_gamma,
                random_mask_frame_latent=args.singleturn_mode and not args.singleturn_train_mask_sam_only,
                mask_condition_source="mask_sam" if args.singleturn_train_mask_sam_only else "mask_check",
            )
        else:
            train_dataset = CachedVideoLatentDataset(args.cached_data_meta, args.cached_data_dir)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        if args.singleturn_mode:
            train_dataset = SingleTurnEditDataset(
                data_root=args.train_data_dir,
                manifest_path=args.train_data_manifest,
                video_sample_size=resolve_singleturn_sample_size(args),
                text_drop_ratio=0.1,
            )
            batch_sampler_generator = torch.Generator().manual_seed(args.seed)
            batch_sampler = ImageVideoSampler(
                RandomSampler(train_dataset, generator=batch_sampler_generator),
                train_dataset,
                args.train_batch_size,
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                persistent_workers=True if args.dataloader_num_workers != 0 else False,
                num_workers=args.dataloader_num_workers,
            )

        if (not args.singleturn_mode) and args.fix_sample_size is not None and args.enable_bucket:
            args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
            args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
            args.training_with_video_token_length = False
            args.random_hw_adapt = False

        use_reasoning_dataset = args.use_reasoning_dataset

        if args.singleturn_mode:
            pass
        elif use_reasoning_dataset:
            print("Detected grounded triplet metadata. Using VideoEditReasoningDataset.")
            train_dataset = VideoEditReasoningDataset(
                ann_path=args.train_data_meta,
                data_root=args.train_data_dir,
                video_sample_stride=args.video_sample_stride,
                video_sample_n_frames=args.video_sample_n_frames,
                source_frames=args.source_frames,
                reasoning_frames=args.reasoning_frames,
                edit_frames=args.edit_frames,
                text_drop_ratio=0.1,
                enable_bucket=args.enable_bucket,
                enable_inpaint=True if args.train_mode != "normal" else False,
            )
        else:
            train_dataset = VideoEditDataset(
                ann_path=args.train_data_meta,
                data_root=args.train_data_dir,
                video_sample_stride=args.video_sample_stride,
                video_sample_n_frames=args.video_sample_n_frames,
                source_frames=args.source_frames,
                edit_frames=args.edit_frames,
                text_drop_ratio=0.1,
                enable_bucket=args.enable_bucket,
                enable_inpaint=True if args.train_mode != "normal" else False,
            )

        if (not args.singleturn_mode) and args.enable_bucket:
            #aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
            aspect_ratio_sample_size = CUSTOM_ASPECT_RATIOS
            batch_sampler_generator = torch.Generator().manual_seed(args.seed)
            batch_sampler = AspectRatioBatchImageVideoSampler(
                sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset,
                batch_size=args.train_batch_size, train_folder=args.train_data_dir, drop_last=True,
                aspect_ratios=aspect_ratio_sample_size,
            )

            def get_length_to_frame_num(token_length):
                if args.image_sample_size > args.video_sample_size:
                    sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))

                    if sample_sizes[-1] != args.image_sample_size:
                        sample_sizes.append(args.image_sample_size)
                else:
                    sample_sizes = [args.image_sample_size]

                length_to_frame_num = {
                    sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
                }

                return length_to_frame_num

            def collate_fn(examples):
                # Get token length
                target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
                length_to_frame_num = get_length_to_frame_num(target_token_length)

                # Create new output
                new_examples = {}
                new_examples["target_token_length"] = target_token_length
                new_examples["pixel_values"] = []
                new_examples["text"] = []
                if args.train_mode != "normal":
                    new_examples["mask_pixel_values"] = []
                    new_examples["mask"] = []
                    new_examples["clip_pixel_values"] = []

                pixel_value = examples[0]["pixel_values"]
                data_type = examples[0]["data_type"]
                f, h, w, c = np.shape(pixel_value)
                if data_type == 'image':
                    random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size], rng=rng)

                    aspect_ratio_sample_size = CUSTOM_ASPECT_RATIOS
                    aspect_ratio_random_crop_sample_size = CUSTOM_ASPECT_RATIOS
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    if args.random_hw_adapt:
                        if args.training_with_video_token_length:
                            local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for example in examples]))
                            choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                            if len(choice_list) == 0:
                                choice_list = list(length_to_frame_num.keys())
                            if rng is None:
                                local_video_sample_size = np.random.choice(choice_list)
                            else:
                                local_video_sample_size = rng.choice(choice_list)
                            batch_video_length = length_to_frame_num[local_video_sample_size]
                            random_downsample_ratio = args.video_sample_size / local_video_sample_size
                        else:
                            random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size, rng=rng)
                            batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                    else:
                        random_downsample_ratio = 1
                        batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

                    aspect_ratio_sample_size = CUSTOM_ASPECT_RATIOS
                    aspect_ratio_random_crop_sample_size = CUSTOM_ASPECT_RATIOS

                if args.fix_sample_size is not None:
                    fix_sample_size = [int(x / 16) * 16 for x in args.fix_sample_size]
                elif args.random_ratio_crop:
                    if rng is None:
                        random_sample_size = aspect_ratio_random_crop_sample_size[
                            np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p=CUSTOM_ASPECT_RATIO_PROB)
                        ]
                    else:
                        random_sample_size = aspect_ratio_random_crop_sample_size[
                            rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p=CUSTOM_ASPECT_RATIO_PROB)
                        ]
                    random_sample_size = [int(x / 16) * 16 for x in random_sample_size]
                else:
                    closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
                    closest_size = [int(x / 16) * 16 for x in closest_size]

                for example in examples:
                    if args.fix_sample_size is not None:
                        pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                        pixel_values = pixel_values / 255.
                        fix_sample_size = list(map(lambda x: int(x), fix_sample_size))
                        transform = transforms.Compose([
                            transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(fix_sample_size),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                        ])
                    elif args.random_ratio_crop:
                        pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                        pixel_values = pixel_values / 255.
                        b, c, h, w = pixel_values.size()
                        th, tw = random_sample_size
                        if th / tw > h / w:
                            nh = int(th)
                            nw = int(w / h * nh)
                        else:
                            nw = int(tw)
                            nh = int(h / w * nw)

                        transform = transforms.Compose([
                            transforms.Resize([nh, nw]),
                            transforms.CenterCrop([int(x) for x in random_sample_size]),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                        ])
                    else:
                        pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                        pixel_values = pixel_values / 255.
                        closest_size = list(map(lambda x: int(x), closest_size))
                        if closest_size[0] / h > closest_size[1] / w:
                            resize_size = closest_size[0], int(w * closest_size[0] / h)
                        else:
                            resize_size = int(h * closest_size[1] / w), closest_size[1]

                        transform = transforms.Compose([
                            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(closest_size),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                        ])
                    new_examples["pixel_values"].append(transform(pixel_values))
                    new_examples["text"].append(example["text"])

                    batch_video_length = int(min(batch_video_length, len(pixel_values)))
                    batch_video_length = (batch_video_length - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1

                    if batch_video_length <= 0:
                        batch_video_length = 1

                    if args.train_mode != "normal":
                        mask = get_random_mask(new_examples["pixel_values"][-1].size(), image_start_only=True)
                        mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask)
                        new_examples["mask_pixel_values"].append(mask_pixel_values)
                        new_examples["mask"].append(mask)

                        clip_pixel_values = new_examples["pixel_values"][-1][0].permute(1, 2, 0).contiguous()
                        clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                        new_examples["clip_pixel_values"].append(clip_pixel_values)

                new_examples["pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["pixel_values"]])
                if args.train_mode != "normal":
                    new_examples["mask_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["mask_pixel_values"]])
                    new_examples["mask"] = torch.stack([example[:batch_video_length] for example in new_examples["mask"]])
                    new_examples["clip_pixel_values"] = torch.stack([example for example in new_examples["clip_pixel_values"]])

                if args.enable_text_encoder_in_dataloader:
                    prompt_ids = tokenizer(
                        new_examples['text'],
                        max_length=args.tokenizer_max_length,
                        padding="max_length",
                        add_special_tokens=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    encoder_hidden_states = text_encoder(
                        prompt_ids.input_ids
                    )[0]
                    new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                    new_examples['encoder_hidden_states'] = encoder_hidden_states

                return new_examples

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                persistent_workers=True if args.dataloader_num_workers != 0 else False,
                num_workers=args.dataloader_num_workers,
            )
        elif not args.singleturn_mode:
            batch_sampler_generator = torch.Generator().manual_seed(args.seed)
            batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                persistent_workers=True if args.dataloader_num_workers != 0 else False,
                num_workers=args.dataloader_num_workers,
            )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    if fsdp_stage != 0:
        transformer3d.network = network
        transformer3d = transformer3d.to(weight_dtype)
        transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer3d, optimizer, train_dataloader, lr_scheduler
        )
    else:
        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            network, optimizer, train_dataloader, lr_scheduler
        )

    if zero_stage == 3:
        from functools import partial
        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        transformer3d = shard_fn(transformer3d)

    if fsdp_stage != 0:
        from functools import partial
        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        if text_encoder is not None:
            text_encoder = shard_fn(text_encoder)

    # Move text_encode and vae to gpu and cast to weight_dtype
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    if text_encoder is not None and not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device)
    if clip_image_encoder is not None:
        clip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop("fix_sample_size")
        init_kwargs = {}
        if args.report_to in {"wandb", "all"} and args.tracker_entity:
            init_kwargs["wandb"] = {"entity": args.tracker_entity}
        accelerator.init_trackers(args.tracker_project_name, tracker_config, init_kwargs=init_kwargs)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            checkpoint_folder_path = os.path.join(args.output_dir, path)
            pkl_path = os.path.join(checkpoint_folder_path, "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")

            if zero_stage != 3 and not args.use_fsdp:
                from safetensors.torch import load_file
                state_dict = load_file(os.path.join(checkpoint_folder_path, "lora_diffusion_pytorch_model.safetensors"), device=str(accelerator.device))
                m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

                optimizer_file_pt = os.path.join(checkpoint_folder_path, "optimizer.pt")
                optimizer_file_bin = os.path.join(checkpoint_folder_path, "optimizer.bin")
                optimizer_file_to_load = None

                if os.path.exists(optimizer_file_pt):
                    optimizer_file_to_load = optimizer_file_pt
                elif os.path.exists(optimizer_file_bin):
                    optimizer_file_to_load = optimizer_file_bin

                if optimizer_file_to_load:
                    try:
                        accelerator.print(f"Loading optimizer state from {optimizer_file_to_load}")
                        optimizer_state = torch.load(optimizer_file_to_load, map_location=accelerator.device)
                        optimizer.load_state_dict(optimizer_state)
                        accelerator.print("Optimizer state loaded successfully.")
                    except Exception as e:
                        accelerator.print(f"Failed to load optimizer state from {optimizer_file_to_load}: {e}")

                scheduler_file_pt = os.path.join(checkpoint_folder_path, "scheduler.pt")
                scheduler_file_bin = os.path.join(checkpoint_folder_path, "scheduler.bin")
                scheduler_file_to_load = None

                if os.path.exists(scheduler_file_pt):
                    scheduler_file_to_load = scheduler_file_pt
                elif os.path.exists(scheduler_file_bin):
                    scheduler_file_to_load = scheduler_file_bin

                if scheduler_file_to_load:
                    try:
                        accelerator.print(f"Loading scheduler state from {scheduler_file_to_load}")
                        scheduler_state = torch.load(scheduler_file_to_load, map_location=accelerator.device)
                        lr_scheduler.load_state_dict(scheduler_state)
                        accelerator.print("Scheduler state loaded successfully.")
                    except Exception as e:
                        accelerator.print(f"Failed to load scheduler state from {scheduler_file_to_load}: {e}")

                if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                    scaler_file = os.path.join(checkpoint_folder_path, "scaler.pt")
                    if os.path.exists(scaler_file):
                        try:
                            accelerator.print(f"Loading GradScaler state from {scaler_file}")
                            scaler_state = torch.load(scaler_file, map_location=accelerator.device)
                            accelerator.scaler.load_state_dict(scaler_state)
                            accelerator.print("GradScaler state loaded successfully.")
                        except Exception as e:
                            accelerator.print(f"Failed to load GradScaler state: {e}")

            else:
                accelerator.load_state(checkpoint_folder_path)
                accelerator.print("accelerator.load_state() completed for zero_stage 3.")

    else:
        initial_global_step = 0

    # function for saving/removing
    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream and args.train_mode != "normal":
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        if batch_sampler is not None and hasattr(batch_sampler, "sampler") and hasattr(batch_sampler.sampler, "generator"):
            batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            if (not use_cached_data) and epoch == first_epoch and step == 0:
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                if args.singleturn_mode:
                    src_pixel_values = rearrange(batch['pixel_values_src_image'].cpu(), "b f c h w -> b c f h w")
                    tgt_pixel_values = rearrange(batch['pixel_values_tgt_image'].cpu(), "b f c h w -> b c f h w")
                    for idx, (src_value, tgt_value, text) in enumerate(zip(src_pixel_values, tgt_pixel_values, batch['text'])):
                        gif_name = '-'.join(text.replace('/', '').split()[:10]) if text != '' else f'{global_step}-{idx}'
                        save_videos_grid(src_value[None, ...], f"{args.output_dir}/sanity_check/{gif_name[:10]}_src.gif", rescale=True)
                        save_videos_grid(tgt_value[None, ...], f"{args.output_dir}/sanity_check/{gif_name[:10]}_tgt.gif", rescale=True)
                else:
                    pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                        save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)
                    if args.train_mode != "normal":
                        clip_pixel_values, mask_pixel_values, texts = batch['clip_pixel_values'].cpu(), batch['mask_pixel_values'].cpu(), batch['text']
                        mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
                        for idx, (clip_pixel_value, pixel_value, text) in enumerate(zip(clip_pixel_values, mask_pixel_values, texts)):
                            pixel_value = pixel_value[None, ...]
                            Image.fromarray(np.uint8(clip_pixel_value)).save(f"{args.output_dir}/sanity_check/clip_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.png")
                            save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/mask_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.gif", rescale=True)

            with accelerator.accumulate(transformer3d):
                batch_texts = None
                singleturn_loss_mask = None
                singleturn_supervised_start_frames = None
                singleturn_refine_target_latents = None
                singleturn_refine_loss_weight_map = None
                text_split_point = None
                latent_split_point = None
                if use_cached_data:
                    with torch.no_grad():
                        if args.singleturn_mode:
                            (
                                latents,
                                prompt_embeds,
                                singleturn_refine_target_latents,
                                singleturn_refine_loss_weight_map,
                                text_split_point,
                            ) = load_cached_singleturn_batch(
                                batch=batch,
                                weight_dtype=weight_dtype,
                                device=accelerator.device,
                                refinement_mode=args.singleturn_refine_mode,
                            )
                        else:
                            latents, prompt_embeds = load_cached_batch(
                                batch=batch,
                                weight_dtype=weight_dtype,
                                device=accelerator.device,
                            )
                    inpaint_latents = None
                    clip_context = None
                else:
                    if args.singleturn_mode:
                        src_pixel_values = batch["pixel_values_src_image"].to(weight_dtype)
                        mask_frame_pixel_values = batch["pixel_values_mask_frame"].to(weight_dtype)
                        tgt_pixel_values = batch["pixel_values_tgt_image"].to(weight_dtype)
                        mask_check_pixel_values = batch["pixel_values_mask_check"].to(weight_dtype)
                        batch_texts = [CORNE_SINGLETURN_PROMPT] * src_pixel_values.shape[0]
                        inpaint_latents = None
                        clip_context = None
                    else:
                        # Convert images to latent space
                        pixel_values = batch["pixel_values"].to(weight_dtype)
                        batch_texts = batch["text"]

                        # Increase the batch size when the length of the latent sequence of the current sample is small
                        if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                            if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                                pixel_values = torch.tile(pixel_values, (4, 1, 1, 1, 1))
                                if args.enable_text_encoder_in_dataloader:
                                    batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (4, 1, 1))
                                    batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (4, 1))
                                else:
                                    batch['text'] = batch['text'] * 4
                                    batch_texts = batch['text']
                            elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                                pixel_values = torch.tile(pixel_values, (2, 1, 1, 1, 1))
                                if args.enable_text_encoder_in_dataloader:
                                    batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (2, 1, 1))
                                    batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (2, 1))
                                else:
                                    batch['text'] = batch['text'] * 2
                                    batch_texts = batch['text']
                
                if (not use_cached_data) and (not args.singleturn_mode) and args.train_mode != "normal":
                    clip_pixel_values = batch["clip_pixel_values"].to(weight_dtype)
                    mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
                    mask = batch["mask"].to(weight_dtype)
                    # Increase the batch size when the length of the latent sequence of the current sample is small
                    if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                        if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            clip_pixel_values = torch.tile(clip_pixel_values, (4, 1, 1, 1))
                            mask_pixel_values = torch.tile(mask_pixel_values, (4, 1, 1, 1, 1))
                            mask = torch.tile(mask, (4, 1, 1, 1, 1))
                        elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            clip_pixel_values = torch.tile(clip_pixel_values, (2, 1, 1, 1))
                            mask_pixel_values = torch.tile(mask_pixel_values, (2, 1, 1, 1, 1))
                            mask = torch.tile(mask, (2, 1, 1, 1, 1))

                if (not use_cached_data) and (not args.singleturn_mode) and args.random_frame_crop:
                    def _create_special_list(length):
                        if length == 1:
                            return [1.0]
                        if length >= 2:
                            last_element = 0.90
                            remaining_sum = 1.0 - last_element
                            other_elements_value = remaining_sum / (length - 1)
                            special_list = [other_elements_value] * (length - 1) + [last_element]
                            return special_list
                    select_frames = [_tmp for _tmp in list(range(sample_n_frames_bucket_interval + 1, args.video_sample_n_frames + sample_n_frames_bucket_interval, sample_n_frames_bucket_interval))]
                    select_frames_prob = np.array(_create_special_list(len(select_frames)))
                    
                    if len(select_frames) != 0:
                        if rng is None:
                            temp_n_frames = np.random.choice(select_frames, p = select_frames_prob)
                        else:
                            temp_n_frames = rng.choice(select_frames, p = select_frames_prob)
                    else:
                        temp_n_frames = 1

                    # Magvae needs the number of frames to be 4n + 1.
                    temp_n_frames = (temp_n_frames - 1) // sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :temp_n_frames, :, :]

                    if args.train_mode != "normal":
                        mask_pixel_values = mask_pixel_values[:, :temp_n_frames, :, :]
                        mask = mask[:, :temp_n_frames, :, :]
                    
                # Keep all node same token length to accelerate the traning when resolution grows.
                if (not use_cached_data) and (not args.singleturn_mode) and args.keep_all_node_same_token_length:
                    if args.token_sample_size > 256:
                        numbers_list = list(range(256, args.token_sample_size + 1, 128))

                        if numbers_list[-1] != args.token_sample_size:
                            numbers_list.append(args.token_sample_size)
                    else:
                        numbers_list = [256]
                    numbers_list = [_number * _number * args.video_sample_n_frames for _number in  numbers_list]
            
                    actual_token_length = index_rng.choice(numbers_list)
                    actual_video_length = (min(
                            actual_token_length / pixel_values.size()[-1] / pixel_values.size()[-2], args.video_sample_n_frames
                    ) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1
                    actual_video_length = int(max(actual_video_length, 1))

                    # Magvae needs the number of frames to be 4n + 1.
                    actual_video_length = (actual_video_length - 1) // sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :actual_video_length, :, :]
                    if args.train_mode != "normal":
                        mask_pixel_values = mask_pixel_values[:, :actual_video_length, :, :]
                        mask = mask[:, :actual_video_length, :, :]

                # Make the inpaint latents to be zeros.
                if (not use_cached_data) and (not args.singleturn_mode) and args.train_mode != "normal":
                    t2v_flag = [(_mask == 1).all() for _mask in mask]
                    new_t2v_flag = []
                    for _mask in t2v_flag:
                        if _mask and np.random.rand() < 0.90:
                            new_t2v_flag.append(0)
                        else:
                            new_t2v_flag.append(1)
                    t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(accelerator.device, dtype=weight_dtype)

                if (not use_cached_data) and args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    if args.train_mode != "normal":
                        clip_image_encoder.to(accelerator.device)
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to("cpu")

                if not use_cached_data:
                    with torch.no_grad():
                        # This way is quicker when batch grows up
                        def _batch_encode_vae(pixel_values, use_mode=False):
                            pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                            bs = args.vae_mini_batch
                            new_pixel_values = []
                            for i in range(0, pixel_values.shape[0], bs):
                                pixel_values_bs = pixel_values[i : i + bs]
                                pixel_values_bs = vae.encode(pixel_values_bs)[0]
                                pixel_values_bs = pixel_values_bs.mode() if use_mode else pixel_values_bs.sample()
                                new_pixel_values.append(pixel_values_bs)
                            return torch.cat(new_pixel_values, dim = 0)

                        # Debug: log shapes before VAE encode
                        if args.debug_shapes and accelerator.is_local_main_process:
                            should_log = (global_step == 0 and step < 2) or (global_step > 0 and (global_step % args.debug_log_interval == 0))
                            if should_log:
                                try:
                                    if args.singleturn_mode:
                                        src_dbg = tuple(src_pixel_values.shape)
                                        tgt_dbg = tuple(tgt_pixel_values.shape)
                                        print(f"[DEBUG] singleturn src/tgt shapes: {src_dbg} / {tgt_dbg} dtype={src_pixel_values.dtype}")
                                    else:
                                        bsz_dbg, f_dbg, c_dbg, h_dbg, w_dbg = pixel_values.shape
                                        print(f"[DEBUG] pixel_values shape (B,F,C,H,W): {(bsz_dbg, f_dbg, c_dbg, h_dbg, w_dbg)} dtype={pixel_values.dtype}")
                                except Exception as _:
                                    pass
                        if args.singleturn_mode:
                            mask_frame_latents = _batch_encode_vae(mask_frame_pixel_values, use_mode=True)
                            source_latents = _batch_encode_vae(src_pixel_values, use_mode=True)
                            target_latents = _batch_encode_vae(tgt_pixel_values, use_mode=True)
                            latent_mask = resize_singleturn_mask_to_latent_grid(mask_check_pixel_values, source_latents)
                            noise_latents = torch.randn(
                                source_latents.size(),
                                device=source_latents.device,
                                generator=torch_rng,
                                dtype=weight_dtype,
                            )
                            latents = build_singleturn_object_removal_latents(
                                mask_frame_latent=mask_frame_latents,
                                source_frame_latent=source_latents,
                                bg_latent=target_latents,
                                mask_check_latent=latent_mask,
                                noise_latent=noise_latents,
                                total_frames=compute_singleturn_object_removal_total_frames(
                                    args.singleturn_cache_corruption_frames,
                                    args.singleturn_cache_restoration_frames,
                                ),
                                corruption_frames=args.singleturn_cache_corruption_frames,
                                restoration_frames=args.singleturn_cache_restoration_frames,
                                interpolation_gamma=args.singleturn_cache_interpolation_gamma,
                            )
                        elif vae_stream_1 is not None:
                            vae_stream_1.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(vae_stream_1):
                                latents = _batch_encode_vae(pixel_values)
                        else:
                            latents = _batch_encode_vae(pixel_values)

                        if (not args.singleturn_mode) and args.train_mode != "normal":
                            mask = rearrange(mask, "b f c h w -> b c f h w")
                            mask = torch.concat(
                                [
                                    torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
                                    mask[:, :, 1:]
                                ], dim=2
                            )
                            mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
                            mask = mask.transpose(1, 2)
                            mask = resize_mask(1 - mask, latents)

                            # Encode inpaint latents.
                            mask_latents = _batch_encode_vae(mask_pixel_values)
                            if vae_stream_2 is not None:
                                torch.cuda.current_stream().wait_stream(vae_stream_2) 

                            inpaint_latents = torch.concat([mask, mask_latents], dim=1)
                            inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents

                            clip_context = []
                            for clip_pixel_value in clip_pixel_values:
                                clip_image = Image.fromarray(np.uint8(clip_pixel_value.float().cpu().numpy()))
                                clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(clip_image_encoder.device, weight_dtype)
                                _clip_context = clip_image_encoder([clip_image[:, None, :, :]])
                                clip_context.append(_clip_context)
                            clip_context = torch.cat(clip_context)
                        
                    # wait for latents = vae.encode(pixel_values) to complete
                    if vae_stream_1 is not None:
                        torch.cuda.current_stream().wait_stream(vae_stream_1)

                # Debug: log shapes after VAE encode and effective compression
                if (not use_cached_data) and args.debug_shapes and accelerator.is_local_main_process:
                    should_log = (global_step == 0 and step < 2) or (global_step > 0 and (global_step % args.debug_log_interval == 0))
                    if should_log:
                        try:
                            b_lat, c_lat, f_lat, h_lat, w_lat = latents.shape
                            if args.singleturn_mode:
                                b_in, f_in, c_in, h_in, w_in = src_pixel_values.shape
                            else:
                                b_in, f_in, c_in, h_in, w_in = pixel_values.shape
                            print(f"[DEBUG] latents shape (B,C,F,H,W): {(b_lat, c_lat, f_lat, h_lat, w_lat)} dtype={latents.dtype}")
                        except Exception as _:
                            pass

                if (not use_cached_data) and args.low_vram:
                    vae.to('cpu')
                    if args.train_mode != "normal":
                        clip_image_encoder.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                if not use_cached_data:
                    if args.enable_text_encoder_in_dataloader:
                        prompt_embeds = batch['encoder_hidden_states'].to(device=latents.device)
                    else:
                        with torch.no_grad():
                            prompt_ids = tokenizer(
                                batch_texts,
                                padding="max_length",
                                max_length=args.tokenizer_max_length,
                                truncation=True,
                                add_special_tokens=True,
                                return_tensors="pt"
                            )
                            text_input_ids = prompt_ids.input_ids
                            prompt_attention_mask = prompt_ids.attention_mask

                            seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                            prompt_embeds = text_encoder(
                                text_input_ids.to(latents.device),
                                attention_mask=prompt_attention_mask.to(latents.device),
                            )[0]
                            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                    if args.low_vram and not args.enable_text_encoder_in_dataloader:
                        text_encoder.to('cpu')
                        torch.cuda.empty_cache()

                bsz, channel, num_frames, height, width = latents.size()
                if args.singleturn_refine_mode:
                    noise = None
                    timesteps = torch.zeros((bsz,), device=latents.device, dtype=noise_scheduler.timesteps.dtype)
                else:
                    noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                    if not args.uniform_sampling:
                        u = compute_density_for_timestep_sampling(
                            weighting_scheme=args.weighting_scheme,
                            batch_size=bsz,
                            logit_mean=args.logit_mean,
                            logit_std=args.logit_std,
                            mode_scale=args.mode_scale,
                        )
                        indices = (u * noise_scheduler.config.num_train_timesteps).long()
                    else:
                        indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                        indices = indices.long().cpu()
                    timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                if args.singleturn_refine_mode:
                    noisy_latents = latents
                    target = singleturn_refine_target_latents - latents
                else:
                    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                    if args.singleturn_mode:
                        expected_singleturn_total_frames = compute_singleturn_object_removal_total_frames(
                            args.singleturn_cache_corruption_frames,
                            args.singleturn_cache_restoration_frames,
                        )
                        use_dynamic_singleturn_tail_start = False
                        singleturn_source_condition_frame = None
                        if latents.shape[2] == expected_singleturn_total_frames:
                            if use_cached_data:
                                batch_total_frames = batch.get("total_frames")
                                if isinstance(batch_total_frames, torch.Tensor):
                                    total_frames_match = bool(
                                        (batch_total_frames == expected_singleturn_total_frames).all().item()
                                    )
                                else:
                                    total_frames_values = (
                                        [int(value) for value in batch_total_frames]
                                        if isinstance(batch_total_frames, (list, tuple))
                                        else [int(batch_total_frames)]
                                    )
                                    total_frames_match = all(
                                        value == expected_singleturn_total_frames for value in total_frames_values
                                    )
                                use_dynamic_singleturn_tail_start = total_frames_match
                            else:
                                use_dynamic_singleturn_tail_start = True

                        if use_dynamic_singleturn_tail_start:
                            singleturn_supervised_start_frames = torch.full(
                                (latents.shape[0],),
                                2,
                                device=latents.device,
                                dtype=torch.long,
                            )
                            singleturn_source_condition_frame = SINGLETURN_SOURCE_CONDITION_FRAME_INDEX
                        noisy_latents, target, singleturn_loss_mask = prepare_singleturn_noisy_latents(
                            latents,
                            noise,
                            sigmas,
                            supervised_start_frames=singleturn_supervised_start_frames,
                        )
                        if singleturn_source_condition_frame is not None and singleturn_source_condition_frame < latents.shape[2]:
                            source_slice = slice(singleturn_source_condition_frame, singleturn_source_condition_frame + 1)
                            noisy_latents[:, :, source_slice] = latents[:, :, source_slice]
                            singleturn_loss_mask[:, :, source_slice] = 0
                        if (
                            singleturn_supervised_start_frames is not None
                            and args.debug_shapes
                            and accelerator.is_local_main_process
                        ):
                            should_log = (
                                (global_step == 0 and step < 2)
                                or (global_step > 0 and (global_step % args.debug_log_interval == 0))
                            )
                            if should_log:
                                sampled_start_frames = singleturn_supervised_start_frames.detach().cpu()
                                sampled_hist_frames, sampled_hist_counts = torch.unique(
                                    sampled_start_frames,
                                    return_counts=True,
                                )
                                hist_payload = {
                                    int(frame): int(count)
                                    for frame, count in zip(sampled_hist_frames.tolist(), sampled_hist_counts.tolist())
                                }
                                if singleturn_source_condition_frame is not None:
                                    supervised_frame_counts = (
                                        expected_singleturn_total_frames - sampled_start_frames
                                    ).tolist()
                                else:
                                    supervised_frame_counts = (
                                        expected_singleturn_total_frames - sampled_start_frames + 1
                                    ).tolist()
                                print(
                                    "[DEBUG] singleturn supervised_start_frames="
                                    f"{sampled_start_frames.tolist()} supervised_frame_counts={supervised_frame_counts} "
                                    f"hist={hist_payload}"
                                )
                    else:
                        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                        #####temporal in context init ##############
                        edited_start_frame = (args.source_frames) // 4 + 1
                        noisy_latents[:, :, :edited_start_frame] = latents[:, :, :edited_start_frame]
                        #####temporal in context init ##############

                        # Add noise
                        target = noise - latents

                seq_len = compute_wan_seq_len_from_latents(
                    latents,
                    accelerator.unwrap_model(transformer3d).config.patch_size,
                )
                # Compute latent_split_point for decoupled cross-attention
                if text_split_point is not None and args.singleturn_mode:
                    _patch_size = accelerator.unwrap_model(transformer3d).config.patch_size
                    _, _, _num_frames, _h_latent, _w_latent = latents.shape
                    _tokens_per_frame = (_h_latent // _patch_size[1]) * (_w_latent // _patch_size[2])
                    _noisy_anchor_frame_index = SINGLETURN_TAIL_START + args.singleturn_cache_corruption_frames
                    latent_split_point = (_noisy_anchor_frame_index + 1) * _tokens_per_frame
                # Predict the noise residual
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=timesteps,
                        seq_len=seq_len,
                        y=inpaint_latents if args.train_mode != "normal" else None,
                        clip_fea=clip_context if args.train_mode != "normal" else None,
                        latent_split_point=latent_split_point,
                        text_split_point=text_split_point,
                    )
                
                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50, loss_mask=None):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
                    mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * mask
                    if loss_mask is not None:
                        masked_loss = masked_loss * loss_mask.float()
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                    final_loss = masked_loss.mean()
                    return final_loss
                
                if args.singleturn_refine_mode:
                    loss_weights = singleturn_refine_loss_weight_map.float()
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = (loss * loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
                else:
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                    # Check if video edit loss mode is enabled
                    if args.singleturn_mode:
                        loss = custom_mse_loss(
                            noise_pred.float(),
                            target.float(),
                            weighting.float(),
                            loss_mask=singleturn_loss_mask.float(),
                        )
                    elif args.video_edit_loss_on_edited_frames_only:
                        # For video editing: only compute loss on edited frames (second half)
                        # Calculate latent frame indices for edited frames
                        source_frames_latent = (args.source_frames - 1) // 4 + 1  # Convert to latent space
                        edited_start_frame = source_frames_latent
                        # Extract edited frames latents
                        noise_pred_edited = noise_pred[:, :, edited_start_frame:, :, :]
                        target_edited = target[:, :, edited_start_frame:, :, :]
                        loss = custom_mse_loss(noise_pred_edited.float(), target_edited.float(), weighting.float())
                    else:
                        # Standard loss calculation for all frames
                        loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())

                if args.motion_sub_loss and noise_pred.size()[1] > 2:
                    gt_sub_noise = noise_pred[:, :, 1:].float() - noise_pred[:, :, :-1].float()
                    pre_sub_noise = target[:, :, 1:].float() - target[:, :, :-1].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        if not args.save_state:
                            safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                            save_model(safetensor_save_path, accelerator.unwrap_model(network))
                            logger.info(f"Saved safetensor to {safetensor_save_path}")
                        else:
                            accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(accelerator_save_path)
                            logger.info(f"Saved state to {accelerator_save_path}")

                if accelerator.is_main_process:
                    if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            clip_image_encoder,
                            transformer3d,
                            network,
                            config,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    clip_image_encoder,
                    transformer3d,
                    network,
                    config,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        if not args.save_state:
            safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
            save_model(safetensor_save_path, accelerator.unwrap_model(network))
        else:
            accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(accelerator_save_path)
            logger.info(f"Saved state to {accelerator_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
