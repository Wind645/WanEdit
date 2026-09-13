#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.data.singleturn_dataset import CachedSingleTurnLatentDataset, load_singleturn_cache_payload
from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel, WanTransformer3DModel
from videox_fun.pipeline import WanPipeline
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.singleturn_utils import (
    CORNE_SINGLETURN_PROMPT,
    SINGLETURN_ENDPOINT_TOTAL_FRAMES,
    SINGLETURN_MASK_PREDICTION_FRAME_INDEX,
    SINGLETURN_TAIL_START,
    SINGLETURN_TOTAL_FRAMES,
    compute_singleturn_object_removal_total_frames,
    generate_singleturn_sample,
    generate_singleturn_sample_from_latents,
    is_supported_singleturn_object_removal_mode,
    normalize_singleturn_sample_size,
    preprocess_singleturn_image,
    preprocess_singleturn_mask_frame,
    preprocess_singleturn_mask,
    refine_singleturn_sample_from_latents,
    save_singleturn_uncertainty_visuals,
    save_singleturn_outputs,
)
from videox_fun.utils.utils import filter_kwargs


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


def _first_item(value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value[0]
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _resolve_cached_data_dir(args) -> Optional[str]:
    if args.cached_data_dir:
        return args.cached_data_dir
    if args.cached_data_meta:
        return str(Path(args.cached_data_meta).resolve().parent)
    if args.cached_sample_path:
        return str(Path(args.cached_sample_path).resolve().parent)
    return None


def _resolve_cache_path(cache_path: str, cached_data_dir: Optional[str]) -> str:
    if os.path.isabs(cache_path) or cached_data_dir is None:
        return cache_path
    return os.path.join(cached_data_dir, cache_path)


def _load_shared_prompt_cache(shared_prompt_cache: str):
    payload = load_singleturn_cache_payload(shared_prompt_cache)
    # New format: part1/part2 keys for decoupled cross-attention
    if "part1" in payload and "part2" in payload:
        part1 = payload["part1"]
        part2 = payload["part2"]
        part1_embed = part1["embedding"]
        part2_embed = part2["embedding"]
        prompt_embeds = torch.cat([part1_embed, part2_embed], dim=0)
        prompt_seq_len = prompt_embeds.shape[0]
        text_split_point = int(part1["seq_len"]) if "seq_len" in part1 else part1_embed.shape[0]
        result = {
            "prompt_embeds": prompt_embeds,
            "prompt_seq_len": prompt_seq_len,
            "text_split_point": text_split_point,
            "text": payload.get("text", CORNE_SINGLETURN_PROMPT),
            "formatted_text": payload.get("formatted_text", payload.get("text", CORNE_SINGLETURN_PROMPT)),
        }
        if "original" in payload:
            result["original_prompt_embeds"] = payload["original"]["embedding"]
            result["original_prompt_seq_len"] = int(payload["original"]["seq_len"])
        return result
    missing = [key for key in ("prompt_embeds", "prompt_seq_len") if key not in payload]
    if missing:
        raise ValueError(f"Shared prompt cache {shared_prompt_cache} is missing keys: {missing}")
    return {
        "prompt_embeds": payload["prompt_embeds"],
        "prompt_seq_len": int(payload["prompt_seq_len"]),
        "text_split_point": None,
        "text": payload.get("text", CORNE_SINGLETURN_PROMPT),
        "formatted_text": payload.get("formatted_text", payload.get("text", CORNE_SINGLETURN_PROMPT)),
    }


def _encode_fixed_prompt(tokenizer, text_encoder, device: torch.device, weight_dtype: torch.dtype, tokenizer_max_length: int = 512):
    with torch.no_grad():
        prompt_ids = tokenizer(
            [CORNE_SINGLETURN_PROMPT],
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_attention_mask = prompt_ids.attention_mask
        prompt_embeds = text_encoder(
            prompt_ids.input_ids.to(device),
            attention_mask=prompt_attention_mask.to(device),
        )[0][0].detach().cpu().to(weight_dtype)
        prompt_seq_len = int(prompt_attention_mask.gt(0).sum(dim=1)[0].item())
    return {
        "prompt_embeds": prompt_embeds,
        "prompt_seq_len": prompt_seq_len,
        "text": CORNE_SINGLETURN_PROMPT,
        "formatted_text": CORNE_SINGLETURN_PROMPT,
    }


def _get_distributed_context() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, max(1, world_size)


def _resolve_runtime_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank, _ = _get_distributed_context()
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cuda")


def _save_singleturn_variant(
    *,
    output_dir: str,
    stem: str,
    generation: dict,
    fps: int,
):
    os.makedirs(output_dir, exist_ok=True)
    output_paths = save_singleturn_outputs(
        full_frames=generation["full_frames"],
        tail_frames=generation["tail_frames"],
        output_dir=output_dir,
        stem=stem,
        fps=fps,
    )
    uncertainty_map = generation.get("uncertainty_map")
    if uncertainty_map is not None:
        output_paths.update(
            save_singleturn_uncertainty_visuals(
                uncertainty_map=uncertainty_map,
                frames=generation["full_frames"][0],
                output_dir=output_dir,
                stem=stem,
            )
        )
    return output_paths


def _write_singleturn_metadata(
    *,
    output_dir: str,
    stem: str,
    metadata: dict,
):
    metadata_path = os.path.join(output_dir, f"{stem}_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def _save_singleturn_result(
    *,
    output_dir: str,
    stem: str,
    generation: dict,
    metadata: dict,
    fps: int,
):
    output_paths = _save_singleturn_variant(
        output_dir=output_dir,
        stem=stem,
        generation=generation,
        fps=fps,
    )
    metadata_path = _write_singleturn_metadata(
        output_dir=output_dir,
        stem=stem,
        metadata={**metadata, "outputs": output_paths},
    )
    return output_paths, metadata_path


def _enable_refinement_lora(pipeline, args, device: torch.device, weight_dtype: torch.dtype):
    if args.lora_path:
        pipeline = unmerge_lora(
            pipeline,
            args.lora_path,
            args.lora_alpha,
            device=device,
            dtype=weight_dtype,
            sub_transformer_name="transformer",
        )
    pipeline = merge_lora(
        pipeline,
        args.refinement_lora_path,
        args.refinement_lora_alpha,
        device=device,
        dtype=weight_dtype,
        transformer_only=True,
    )
    return pipeline


def _restore_coarse_lora(pipeline, args, device: torch.device, weight_dtype: torch.dtype):
    pipeline = unmerge_lora(
        pipeline,
        args.refinement_lora_path,
        args.refinement_lora_alpha,
        device=device,
        dtype=weight_dtype,
        sub_transformer_name="transformer",
    )
    if args.lora_path:
        pipeline = merge_lora(
            pipeline,
            args.lora_path,
            args.lora_alpha,
            device=device,
            dtype=weight_dtype,
            transformer_only=True,
        )
    return pipeline


def _save_two_stage_singleturn_result(
    *,
    output_dir: str,
    stem: str,
    coarse_generation: dict,
    refined_generation: Optional[dict],
    metadata: dict,
    fps: int,
):
    if refined_generation is None:
        output_paths = _save_singleturn_variant(
            output_dir=output_dir,
            stem=stem,
            generation=coarse_generation,
            fps=fps,
        )
        metadata_path = _write_singleturn_metadata(
            output_dir=output_dir,
            stem=stem,
            metadata={**metadata, "outputs": output_paths},
        )
        return output_paths, None, metadata_path

    coarse_output_paths = _save_singleturn_variant(
        output_dir=output_dir,
        stem=f"{stem}_coarse",
        generation=coarse_generation,
        fps=fps,
    )
    refined_output_paths = _save_singleturn_variant(
        output_dir=output_dir,
        stem=f"{stem}_refined",
        generation=refined_generation,
        fps=fps,
    )
    metadata_path = _write_singleturn_metadata(
        output_dir=output_dir,
        stem=stem,
        metadata={
            **metadata,
            "outputs": coarse_output_paths,
            "coarse_outputs": coarse_output_paths,
            "refined_outputs": refined_output_paths,
        },
    )
    return coarse_output_paths, refined_output_paths, metadata_path


def _tensor_image_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor with shape (C, H, W), got {tuple(tensor.shape)}")
    tensor = tensor.detach().cpu().float()
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    tensor = ((tensor.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def _tensor_mask_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim != 3:
        raise ValueError(f"Expected mask tensor with shape (1, H, W), got {tuple(tensor.shape)}")
    tensor = tensor.detach().cpu().float().clamp(0, 1)
    array = (tensor[0].numpy() * 255).astype("uint8")
    return Image.fromarray(array, mode="L")


def _letterbox_geometry(width: int, height: int, sample_size: tuple[int, int]) -> tuple[int, int, int, int]:
    target_height, target_width = sample_size
    scale = min(target_width / float(width), target_height / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    return resized_width, resized_height, left, top


def _restore_generation_to_original_size(
    generation: dict,
    *,
    original_size: tuple[int, int],
    sample_size: tuple[int, int],
) -> dict:
    orig_width, orig_height = original_size
    resized_width, resized_height, left, top = _letterbox_geometry(orig_width, orig_height, sample_size)
    full_frames = generation["full_frames"]
    tail_frames = generation["tail_frames"]

    def restore(frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"Expected frames with shape (B, C, T, H, W), got {tuple(frames.shape)}")
        frames = frames[..., top : top + resized_height, left : left + resized_width]
        return frames

    return {
        **generation,
        "full_frames": restore(full_frames),
        "tail_frames": restore(tail_frames),
    }


def _load_source_frame_for_blending(source_path: str, *, height: int, width: int, sample_size: tuple[int, int]) -> torch.Tensor:
    if (height, width) == tuple(sample_size):
        source = preprocess_singleturn_image(source_path, sample_size, add_batch_dim=False, add_frame_dim=False)
        source = ((source.float() + 1.0) / 2.0).clamp(0, 1)
    else:
        with Image.open(source_path) as image:
            image = image.convert("RGB").resize((width, height), resample=Image.BILINEAR)
            source = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return source.unsqueeze(0).unsqueeze(2)


def _validate_odd_kernel_size(name: str, value: int) -> int:
    value = int(value)
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}")
    return value


def _objectclear_style_blend_alpha(
    alpha: torch.Tensor,
    *,
    threshold: float,
    dilation_kernel_size: int,
    blur_kernel_size: int,
    blur_sigma: float,
) -> torch.Tensor:
    if alpha.ndim != 5 or alpha.shape[1] != 1 or alpha.shape[2] != 1:
        raise ValueError(f"alpha must have shape (B, 1, 1, H, W), got {tuple(alpha.shape)}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    dilation_kernel_size = _validate_odd_kernel_size("dilation_kernel_size", dilation_kernel_size)
    blur_kernel_size = _validate_odd_kernel_size("blur_kernel_size", blur_kernel_size)
    if blur_sigma <= 0:
        raise ValueError(f"blur_sigma must be positive, got {blur_sigma}")

    alpha_2d = alpha[:, 0, 0].detach().cpu().numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    softened = []
    for sample in alpha_2d:
        binary = (sample >= threshold).astype(np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1).astype(np.float32)
        blurred = cv2.GaussianBlur(dilated, (blur_kernel_size, blur_kernel_size), sigmaX=float(blur_sigma))
        merged = np.maximum(binary.astype(np.float32), blurred)
        softened.append(torch.from_numpy(merged))

    softened_alpha = torch.stack(softened, dim=0).unsqueeze(1).unsqueeze(2)
    return softened_alpha.to(device=alpha.device, dtype=alpha.dtype)


def _save_singleturn_blended_last_frames(
    generation: dict,
    output_paths: dict,
    *,
    source_path: str,
    sample_size: tuple[int, int],
    mask_blend_threshold: float,
    mask_blend_dilate_kernel_size: int,
    mask_blend_blur_kernel_size: int,
    mask_blend_blur_sigma: float,
) -> None:
    if not source_path or not os.path.exists(source_path):
        return

    full_frames = generation["full_frames"].detach().cpu().float()
    if full_frames.ndim != 5 or full_frames.shape[2] <= max(SINGLETURN_MASK_PREDICTION_FRAME_INDEX, SINGLETURN_TAIL_START):
        return

    height, width = full_frames.shape[-2:]
    source_frame = _load_source_frame_for_blending(
        source_path,
        height=height,
        width=width,
        sample_size=sample_size,
    ).to(dtype=full_frames.dtype)
    alpha = full_frames[:, :, SINGLETURN_MASK_PREDICTION_FRAME_INDEX : SINGLETURN_MASK_PREDICTION_FRAME_INDEX + 1]
    alpha = alpha.mean(dim=1, keepdim=True).clamp(0, 1)
    alpha = _objectclear_style_blend_alpha(
        alpha,
        threshold=mask_blend_threshold,
        dilation_kernel_size=mask_blend_dilate_kernel_size,
        blur_kernel_size=mask_blend_blur_kernel_size,
        blur_sigma=mask_blend_blur_sigma,
    )

    last_frame = full_frames[:, :, -1:]
    blended_last = alpha * last_frame + (1.0 - alpha) * source_frame
    frame = blended_last[0, :, 0].permute(1, 2, 0).clamp(0, 1).numpy()
    image = Image.fromarray((frame * 255).astype(np.uint8))

    for key in ("full_last_frame", "tail_last_frame"):
        path = output_paths.get(key)
        if path:
            image.save(path)


def _save_singleturn_input_visuals(
    *,
    output_dir: str,
    stem: str,
    source_path: str,
    mask_path: str,
    sample_size: tuple[int, int],
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    source_tensor = preprocess_singleturn_image(source_path, sample_size)[0, 0]
    mask_tensor = preprocess_singleturn_mask(mask_path, sample_size)[0, 0]
    mask_frame_tensor = preprocess_singleturn_mask_frame(mask_path, sample_size)[0, 0]

    source_output_path = os.path.join(output_dir, f"{stem}_source.png")
    mask_output_path = os.path.join(output_dir, f"{stem}_mask.png")
    mask_frame_output_path = os.path.join(output_dir, f"{stem}_mask_frame_input.png")
    source_frame_output_path = os.path.join(output_dir, f"{stem}_source_frame_input.png")

    _tensor_image_to_pil(source_tensor).save(source_output_path)
    _tensor_mask_to_pil(mask_tensor).save(mask_output_path)
    _tensor_image_to_pil(mask_frame_tensor).save(mask_frame_output_path)
    _tensor_image_to_pil(source_tensor).save(source_frame_output_path)

    return {
        "source_preview": source_output_path,
        "mask_preview": mask_output_path,
        "mask_frame_input_preview": mask_frame_output_path,
        "source_frame_input_preview": source_frame_output_path,
    }


def _resolve_raw_image_dir(raw_data_dir: str) -> str:
    for dirname in ("img", "images", "input"):
        path = os.path.join(raw_data_dir, dirname)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Missing raw image folder: expected img/ or images/ or input/ under {raw_data_dir}")


def _resolve_raw_mask_dir(raw_data_dir: str) -> str:
    for dirname in ("mask", "masks", "condition_mask"):
        path = os.path.join(raw_data_dir, dirname)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Missing raw mask folder: expected mask/ or masks/ or condition_mask/ under {raw_data_dir}")


def _find_raw_file(root: str, rel_stem: str, suffixes: tuple[str, ...]) -> Optional[str]:
    for suffix in suffixes:
        path = os.path.join(root, f"{rel_stem}{suffix}")
        if os.path.exists(path):
            return path
    return None


def _load_raw_folder_triplets(raw_data_dir: str, raw_selected_triplets: Optional[str]) -> list[str]:
    manifest_path = raw_selected_triplets or os.path.join(raw_data_dir, "selected_triplets.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            triplets = payload.get("selected_triplets")
        else:
            triplets = payload
        if not isinstance(triplets, list) or not triplets:
            raise ValueError(f"Invalid raw triplet manifest: {manifest_path}")
        return [str(item) for item in triplets]

    image_dir = _resolve_raw_image_dir(raw_data_dir)
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    triplets = [
        path.relative_to(image_dir).with_suffix("").as_posix()
        for path in sorted(Path(image_dir).rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    if not triplets:
        raise ValueError(f"No raw images found under {image_dir}")
    return triplets


def _resolve_raw_folder_paths(raw_data_dir: str, rel_triplet: str) -> tuple[str, str, str]:
    suffixes = (".jpg", ".jpeg", ".png", ".webp")
    image_dir = _resolve_raw_image_dir(raw_data_dir)
    mask_dir = _resolve_raw_mask_dir(raw_data_dir)
    image_path = _find_raw_file(image_dir, rel_triplet, suffixes)
    mask_path = _find_raw_file(mask_dir, rel_triplet, suffixes)
    if mask_path is None:
        mask_path = _find_raw_file(mask_dir, f"{rel_triplet}_M", suffixes)
    gt_path = ""
    gt_dir = os.path.join(raw_data_dir, "gt")
    if os.path.isdir(gt_dir):
        gt_path = _find_raw_file(gt_dir, rel_triplet, suffixes) or ""
    if image_path is None or not os.path.exists(image_path):
        raise FileNotFoundError(f"Missing raw image for triplet {rel_triplet}: expected image under {image_dir}")
    if mask_path is None or not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing raw mask for triplet {rel_triplet}: expected same stem or _M suffix under {mask_dir}")
    return image_path, mask_path, gt_path


def _run_singleturn_image_mode(pipeline, args, weight_dtype, generator):
    source_tensor = preprocess_singleturn_image(args.image_path, args.sample_size).to(
        device=pipeline._execution_device,
        dtype=weight_dtype,
    )
    mask_frame_tensor = preprocess_singleturn_mask_frame(args.mask_path, args.sample_size).to(
        device=pipeline._execution_device,
        dtype=weight_dtype,
    )

    prompt_cache = _encode_fixed_prompt(pipeline.tokenizer, pipeline.text_encoder, pipeline._execution_device, weight_dtype)
    with torch.no_grad():
        coarse_generation = generate_singleturn_sample(
            pipeline=pipeline,
            mask_frame_tensor=mask_frame_tensor,
            source_tensor=source_tensor,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            weight_dtype=weight_dtype,
            total_frames=args.singleturn_total_frames,
            enable_uncertainty_viz=args.enable_uncertainty_viz,
            uncertainty_last_steps=args.uncertainty_last_steps,
            enable_trajectory_refinement=args.enable_trajectory_refinement,
            trajectory_refinement_remaining_steps=args.trajectory_refinement_remaining_steps,
            trajectory_refinement_corruption_frames=args.singleturn_cache_corruption_frames,
            trajectory_refinement_restoration_frames=args.singleturn_cache_restoration_frames,
            trajectory_refinement_gamma=args.trajectory_refinement_gamma,
            trajectory_refinement_strength=args.trajectory_refinement_strength,
        )
        refined_generation = None
        if args.enable_refinement:
            if args.singleturn_total_frames != SINGLETURN_TOTAL_FRAMES:
                raise ValueError(
                    "The current refinement stage only supports 8-frame coarse latents. "
                    f"Got singleturn_total_frames={args.singleturn_total_frames}."
                )
            pipeline = _enable_refinement_lora(pipeline, args, pipeline._execution_device, weight_dtype)
            try:
                refined_generation = refine_singleturn_sample_from_latents(
                    pipeline=pipeline,
                    coarse_latents=coarse_generation["latents"],
                    prompt_embeds=prompt_cache["prompt_embeds"],
                    prompt_seq_len=prompt_cache["prompt_seq_len"],
                    negative_prompt=args.negative_prompt,
                    guidance_scale=args.refinement_guidance_scale,
                    weight_dtype=weight_dtype,
                )
            finally:
                pipeline = _restore_coarse_lora(pipeline, args, pipeline._execution_device, weight_dtype)

    output_name = args.output_name or Path(args.image_path).stem
    preview_paths = _save_singleturn_input_visuals(
        output_dir=args.output_dir,
        stem=output_name,
        source_path=args.image_path,
        mask_path=args.mask_path,
        sample_size=args.sample_size,
    )
    coarse_output_paths, refined_output_paths, metadata_path = _save_two_stage_singleturn_result(
        output_dir=args.output_dir,
        stem=output_name,
        coarse_generation=coarse_generation,
        refined_generation=refined_generation,
        metadata={
            "mode": "image",
            "image_path": args.image_path,
            "mask_path": args.mask_path,
            "prompt": CORNE_SINGLETURN_PROMPT,
            "formatted_prompt": CORNE_SINGLETURN_PROMPT,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "sample_size": list(args.sample_size),
            "singleturn_total_frames": args.singleturn_total_frames,
            "mask_blending_enabled": bool(args.enable_mask_blending),
            "mask_blend_threshold": args.mask_blend_threshold,
            "mask_blend_dilate_kernel_size": args.mask_blend_dilate_kernel_size,
            "mask_blend_blur_kernel_size": args.mask_blend_blur_kernel_size,
            "mask_blend_blur_sigma": args.mask_blend_blur_sigma,
            "uncertainty_viz_enabled": bool(args.enable_uncertainty_viz),
            "uncertainty_last_steps": args.uncertainty_last_steps,
            "trajectory_refinement_enabled": bool(args.enable_trajectory_refinement),
            "trajectory_refinement_remaining_steps": args.trajectory_refinement_remaining_steps,
            "trajectory_refinement_gamma": args.trajectory_refinement_gamma,
            "trajectory_refinement_strength": args.trajectory_refinement_strength,
            "input_previews": preview_paths,
            "coarse_lora_path": args.lora_path or "",
            "coarse_lora_alpha": args.lora_alpha,
            "refinement_enabled": bool(args.enable_refinement),
            "refinement_lora_path": args.refinement_lora_path or "",
            "refinement_lora_alpha": args.refinement_lora_alpha,
            "refinement_guidance_scale": args.refinement_guidance_scale,
        },
        fps=args.fps,
    )
    if args.enable_mask_blending:
        _save_singleturn_blended_last_frames(
            coarse_generation,
            coarse_output_paths,
            source_path=args.image_path,
            sample_size=args.sample_size,
            mask_blend_threshold=args.mask_blend_threshold,
            mask_blend_dilate_kernel_size=args.mask_blend_dilate_kernel_size,
            mask_blend_blur_kernel_size=args.mask_blend_blur_kernel_size,
            mask_blend_blur_sigma=args.mask_blend_blur_sigma,
        )
        if refined_generation is not None and refined_output_paths is not None:
            _save_singleturn_blended_last_frames(
                refined_generation,
                refined_output_paths,
                source_path=args.image_path,
                sample_size=args.sample_size,
                mask_blend_threshold=args.mask_blend_threshold,
                mask_blend_dilate_kernel_size=args.mask_blend_dilate_kernel_size,
                mask_blend_blur_kernel_size=args.mask_blend_blur_kernel_size,
                mask_blend_blur_sigma=args.mask_blend_blur_sigma,
            )

    return {
        "outputs": coarse_output_paths,
        "refined_outputs": refined_output_paths,
        "metadata": metadata_path,
    }


def _run_singleturn_raw_folder_mode(pipeline, args, weight_dtype, generator):
    local_rank, world_size = _get_distributed_context()
    triplets = _load_raw_folder_triplets(args.raw_data_dir, args.raw_selected_triplets)
    start_index = max(0, int(args.cached_start_index))
    if start_index >= len(triplets):
        raise ValueError(f"--cached_start_index {start_index} is out of range for raw dataset of length {len(triplets)}.")
    end_index = len(triplets)
    if args.cached_num_samples is not None:
        if args.cached_num_samples <= 0:
            raise ValueError("--cached_num_samples must be positive when provided.")
        end_index = min(len(triplets), start_index + int(args.cached_num_samples))
    indices = list(range(start_index, end_index))
    if world_size > 1:
        indices = indices[local_rank::world_size]
        if not indices:
            return {"outputs": [], "summary": None}

    prompt_cache = _encode_fixed_prompt(pipeline.tokenizer, pipeline.text_encoder, pipeline._execution_device, weight_dtype)
    results = []
    for index in tqdm(indices, desc="Running raw folder SingleTurn inference"):
        rel_triplet = triplets[index]
        sample_generator = generator
        if args.reset_seed_per_sample:
            sample_generator = torch.Generator(
                device=pipeline._execution_device
            ).manual_seed(args.seed)
        image_path, mask_path, gt_path = _resolve_raw_folder_paths(args.raw_data_dir, rel_triplet)
        with Image.open(image_path) as image:
            original_size = image.size

        source_tensor = preprocess_singleturn_image(image_path, args.sample_size).to(
            device=pipeline._execution_device,
            dtype=weight_dtype,
        )
        mask_frame_tensor = preprocess_singleturn_mask_frame(mask_path, args.sample_size).to(
            device=pipeline._execution_device,
            dtype=weight_dtype,
        )

        with torch.no_grad():
            coarse_generation = generate_singleturn_sample(
                pipeline=pipeline,
                mask_frame_tensor=mask_frame_tensor,
                source_tensor=source_tensor,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=sample_generator,
                weight_dtype=weight_dtype,
                total_frames=args.singleturn_total_frames,
                enable_uncertainty_viz=args.enable_uncertainty_viz,
                uncertainty_last_steps=args.uncertainty_last_steps,
                enable_trajectory_refinement=args.enable_trajectory_refinement,
                trajectory_refinement_remaining_steps=args.trajectory_refinement_remaining_steps,
                trajectory_refinement_corruption_frames=args.singleturn_cache_corruption_frames,
                trajectory_refinement_restoration_frames=args.singleturn_cache_restoration_frames,
                trajectory_refinement_gamma=args.trajectory_refinement_gamma,
                trajectory_refinement_strength=args.trajectory_refinement_strength,
            )
            refined_generation = None
            if args.enable_refinement:
                if args.singleturn_total_frames != SINGLETURN_TOTAL_FRAMES:
                    raise ValueError(
                        "The current refinement stage only supports 8-frame coarse latents. "
                        f"Got singleturn_total_frames={args.singleturn_total_frames}."
                    )
                pipeline = _enable_refinement_lora(pipeline, args, pipeline._execution_device, weight_dtype)
                try:
                    refined_generation = refine_singleturn_sample_from_latents(
                        pipeline=pipeline,
                        coarse_latents=coarse_generation["latents"],
                        prompt_embeds=prompt_cache["prompt_embeds"],
                        prompt_seq_len=prompt_cache["prompt_seq_len"],
                        negative_prompt=args.negative_prompt,
                        guidance_scale=args.refinement_guidance_scale,
                        weight_dtype=weight_dtype,
                    )
                finally:
                    pipeline = _restore_coarse_lora(pipeline, args, pipeline._execution_device, weight_dtype)

        coarse_generation = _restore_generation_to_original_size(
            coarse_generation,
            original_size=original_size,
            sample_size=args.sample_size,
        )
        if refined_generation is not None:
            refined_generation = _restore_generation_to_original_size(
                refined_generation,
                original_size=original_size,
                sample_size=args.sample_size,
            )
        stem = Path(rel_triplet).name
        sample_output_root = args.output_dir if world_size == 1 else os.path.join(args.output_dir, f"rank{local_rank}")
        sample_output_dir = os.path.join(sample_output_root, f"{index:06d}_{stem}")
        preview_paths = _save_singleturn_input_visuals(
            output_dir=sample_output_dir,
            stem=stem,
            source_path=image_path,
            mask_path=mask_path,
            sample_size=args.sample_size,
        )
        coarse_output_paths, refined_output_paths, metadata_path = _save_two_stage_singleturn_result(
            output_dir=sample_output_dir,
            stem=stem,
            coarse_generation=coarse_generation,
            refined_generation=refined_generation,
            metadata={
                "mode": "raw_folder",
                "raw_data_dir": args.raw_data_dir,
                "rel_triplet": rel_triplet,
                "image_path": image_path,
                "mask_path": mask_path,
                "gt_path": gt_path if os.path.exists(gt_path) else "",
                "prompt": CORNE_SINGLETURN_PROMPT,
                "formatted_prompt": CORNE_SINGLETURN_PROMPT,
                "seed": args.seed,
                "seed_policy": (
                    "fixed_reset_per_sample"
                    if args.reset_seed_per_sample
                    else "one_generator_per_process_advancing_across_samples"
                ),
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "sample_size": list(args.sample_size),
                "original_size": list(original_size),
                "singleturn_total_frames": args.singleturn_total_frames,
                "mask_blending_enabled": bool(args.enable_mask_blending),
                "mask_blend_threshold": args.mask_blend_threshold,
                "mask_blend_dilate_kernel_size": args.mask_blend_dilate_kernel_size,
                "mask_blend_blur_kernel_size": args.mask_blend_blur_kernel_size,
                "mask_blend_blur_sigma": args.mask_blend_blur_sigma,
                "uncertainty_viz_enabled": bool(args.enable_uncertainty_viz),
                "uncertainty_last_steps": args.uncertainty_last_steps,
                "trajectory_refinement_enabled": bool(args.enable_trajectory_refinement),
                "trajectory_refinement_remaining_steps": args.trajectory_refinement_remaining_steps,
                "trajectory_refinement_gamma": args.trajectory_refinement_gamma,
                "trajectory_refinement_strength": args.trajectory_refinement_strength,
                "input_previews": preview_paths,
                "coarse_lora_path": args.lora_path or "",
                "coarse_lora_alpha": args.lora_alpha,
                "refinement_enabled": bool(args.enable_refinement),
                "refinement_lora_path": args.refinement_lora_path or "",
                "refinement_lora_alpha": args.refinement_lora_alpha,
                "refinement_guidance_scale": args.refinement_guidance_scale,
            },
            fps=args.fps,
        )
        if args.enable_mask_blending:
            _save_singleturn_blended_last_frames(
                coarse_generation,
                coarse_output_paths,
                source_path=image_path,
                sample_size=args.sample_size,
                mask_blend_threshold=args.mask_blend_threshold,
                mask_blend_dilate_kernel_size=args.mask_blend_dilate_kernel_size,
                mask_blend_blur_kernel_size=args.mask_blend_blur_kernel_size,
                mask_blend_blur_sigma=args.mask_blend_blur_sigma,
            )
            if refined_generation is not None and refined_output_paths is not None:
                _save_singleturn_blended_last_frames(
                    refined_generation,
                    refined_output_paths,
                    source_path=image_path,
                    sample_size=args.sample_size,
                    mask_blend_threshold=args.mask_blend_threshold,
                    mask_blend_dilate_kernel_size=args.mask_blend_dilate_kernel_size,
                    mask_blend_blur_kernel_size=args.mask_blend_blur_kernel_size,
                    mask_blend_blur_sigma=args.mask_blend_blur_sigma,
                )
        results.append(
            {
                "output_paths": coarse_output_paths,
                "refined_output_paths": refined_output_paths,
                "metadata": metadata_path,
                "sample_output_dir": sample_output_dir,
                "rel_triplet": rel_triplet,
                "image_path": image_path,
                "mask_path": mask_path,
                "gt_path": gt_path if os.path.exists(gt_path) else "",
            }
        )

    summary_root = args.output_dir if world_size == 1 else os.path.join(args.output_dir, f"rank{local_rank}")
    summary_path = os.path.join(summary_root, "raw_infer_manifest.json")
    os.makedirs(summary_root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return {"outputs": results, "summary": summary_path}


def _run_singleturn_cached_sample(
    *,
    pipeline,
    args,
    weight_dtype,
    generator,
    sample,
    prompt_cache,
    output_dir: str,
    stem: str,
):
    mask_frame_path = sample.get("mask_sam_image", "") if sample.get("used_mask_sam", False) else ""
    if not mask_frame_path:
        mask_frame_path = sample.get("mask_check_image", "")

    with torch.no_grad():
        total_frames = int(sample.get("total_frames", SINGLETURN_TOTAL_FRAMES))
        if args.singleturn_endpoint_mode:
            total_frames = SINGLETURN_ENDPOINT_TOTAL_FRAMES

        # Compute split points for decoupled cross-attention
        text_split_point = prompt_cache.get("text_split_point")
        latent_split_point = None

        if args.rollback_cross_attn:
            text_split_point = None
            use_prompt_embeds = prompt_cache.get("original_prompt_embeds", prompt_cache["prompt_embeds"])
            use_prompt_seq_len = prompt_cache.get("original_prompt_seq_len", prompt_cache["prompt_seq_len"])
        else:
            use_prompt_embeds = prompt_cache["prompt_embeds"]
            use_prompt_seq_len = prompt_cache["prompt_seq_len"]
            if text_split_point is not None:
                patch_size = pipeline.transformer.config.patch_size
                _h_latent = sample["source_frame_latent"].shape[-2]
                _w_latent = sample["source_frame_latent"].shape[-1]
                tokens_per_frame = (_h_latent // patch_size[1]) * (_w_latent // patch_size[2])
                if args.singleturn_endpoint_mode:
                    latent_split_point = SINGLETURN_TAIL_START * tokens_per_frame
                else:
                    noisy_anchor_frame_index = SINGLETURN_TAIL_START + args.singleturn_cache_corruption_frames
                    latent_split_point = (noisy_anchor_frame_index + 1) * tokens_per_frame

        # RoPE temporal index: [0, 1, 1, 2, 3, ..., f-2] — unless rolled back
        if args.rollback_rope:
            temporal_index_map = None
        else:
            t_indices = list(range(total_frames - 1))
            t_indices.insert(1, 1)
            batch_size = 2 if args.guidance_scale > 1.0 else 1
            temporal_index_map = [t_indices] * batch_size

        coarse_generation = generate_singleturn_sample_from_latents(
            pipeline=pipeline,
            mask_frame_latent=sample["mask_frame_latent"],
            source_frame_latent=sample["source_frame_latent"],
            prompt_embeds=use_prompt_embeds,
            prompt_seq_len=use_prompt_seq_len,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            weight_dtype=weight_dtype,
            total_frames=total_frames,
            enable_uncertainty_viz=args.enable_uncertainty_viz,
            uncertainty_last_steps=args.uncertainty_last_steps,
            enable_trajectory_refinement=args.enable_trajectory_refinement,
            trajectory_refinement_remaining_steps=args.trajectory_refinement_remaining_steps,
            trajectory_refinement_corruption_frames=args.singleturn_cache_corruption_frames,
            trajectory_refinement_restoration_frames=args.singleturn_cache_restoration_frames,
            trajectory_refinement_gamma=args.trajectory_refinement_gamma,
            trajectory_refinement_strength=args.trajectory_refinement_strength,
            latent_split_point=latent_split_point,
            text_split_point=text_split_point,
            temporal_index_map=temporal_index_map,
        )
        refined_generation = None
        if args.enable_refinement:
            if total_frames != SINGLETURN_TOTAL_FRAMES:
                raise ValueError(
                    "The current refinement stage only supports 8-frame coarse cached samples. "
                    f"Got total_frames={total_frames} for cache_path={sample.get('cache_path', '')}."
                )
            pipeline = _enable_refinement_lora(pipeline, args, pipeline._execution_device, weight_dtype)
            try:
                refined_generation = refine_singleturn_sample_from_latents(
                    pipeline=pipeline,
                    coarse_latents=coarse_generation["latents"],
                    prompt_embeds=prompt_cache["prompt_embeds"],
                    prompt_seq_len=prompt_cache["prompt_seq_len"],
                    negative_prompt=args.negative_prompt,
                    guidance_scale=args.refinement_guidance_scale,
                    weight_dtype=weight_dtype,
                )
            finally:
                pipeline = _restore_coarse_lora(pipeline, args, pipeline._execution_device, weight_dtype)

    preview_paths = {}
    source_path = sample.get("source_image", "")
    if sample.get("source_image") and mask_frame_path and os.path.exists(source_path) and os.path.exists(mask_frame_path):
        preview_paths = _save_singleturn_input_visuals(
            output_dir=output_dir,
            stem=stem,
            source_path=sample["source_image"],
            mask_path=mask_frame_path,
            sample_size=args.sample_size,
        )

    coarse_output_paths, refined_output_paths, metadata_path = _save_two_stage_singleturn_result(
        output_dir=output_dir,
        stem=stem,
        coarse_generation=coarse_generation,
        refined_generation=refined_generation,
        metadata={
            "mode": "cache",
            "cache_path": sample.get("cache_path", ""),
            "source_image": sample.get("source_image", ""),
            "bg_image": sample.get("bg_image", ""),
            "mask_check_image": sample.get("mask_check_image", ""),
            "mask_sam_image": sample.get("mask_sam_image", ""),
            "used_mask_sam": bool(sample.get("used_mask_sam", False)),
            "prompt": prompt_cache["text"],
            "formatted_prompt": prompt_cache["formatted_text"],
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "singleturn_total_frames": total_frames,
            "mask_blending_enabled": bool(args.enable_mask_blending),
            "mask_blend_threshold": args.mask_blend_threshold,
            "mask_blend_dilate_kernel_size": args.mask_blend_dilate_kernel_size,
            "mask_blend_blur_kernel_size": args.mask_blend_blur_kernel_size,
            "mask_blend_blur_sigma": args.mask_blend_blur_sigma,
            "uncertainty_viz_enabled": bool(args.enable_uncertainty_viz),
            "uncertainty_last_steps": args.uncertainty_last_steps,
            "trajectory_refinement_enabled": bool(args.enable_trajectory_refinement),
            "trajectory_refinement_remaining_steps": args.trajectory_refinement_remaining_steps,
            "trajectory_refinement_gamma": args.trajectory_refinement_gamma,
            "trajectory_refinement_strength": args.trajectory_refinement_strength,
            "cache_mode": sample.get("cache_mode", ""),
            "mask_frame_image": mask_frame_path,
            "input_previews": preview_paths,
            "coarse_lora_path": args.lora_path or "",
            "coarse_lora_alpha": args.lora_alpha,
            "refinement_enabled": bool(args.enable_refinement),
            "refinement_lora_path": args.refinement_lora_path or "",
            "refinement_lora_alpha": args.refinement_lora_alpha,
            "refinement_guidance_scale": args.refinement_guidance_scale,
        },
        fps=args.fps,
    )
    if args.enable_mask_blending:
        _save_singleturn_blended_last_frames(
            coarse_generation,
            coarse_output_paths,
            source_path=source_path,
            sample_size=args.sample_size,
            mask_blend_threshold=args.mask_blend_threshold,
            mask_blend_dilate_kernel_size=args.mask_blend_dilate_kernel_size,
            mask_blend_blur_kernel_size=args.mask_blend_blur_kernel_size,
            mask_blend_blur_sigma=args.mask_blend_blur_sigma,
        )
        if refined_generation is not None and refined_output_paths is not None:
            _save_singleturn_blended_last_frames(
                refined_generation,
                refined_output_paths,
                source_path=source_path,
                sample_size=args.sample_size,
                mask_blend_threshold=args.mask_blend_threshold,
                mask_blend_dilate_kernel_size=args.mask_blend_dilate_kernel_size,
                mask_blend_blur_kernel_size=args.mask_blend_blur_kernel_size,
                mask_blend_blur_sigma=args.mask_blend_blur_sigma,
            )

    return {
        "output_paths": coarse_output_paths,
        "refined_output_paths": refined_output_paths,
        "metadata": metadata_path,
        "sample_output_dir": output_dir,
        "stem": stem,
        "cache_path": sample.get("cache_path", ""),
        "source_image": sample.get("source_image", ""),
        "mask_check_image": sample.get("mask_check_image", ""),
        "used_mask_sam": bool(sample.get("used_mask_sam", False)),
        "formatted_prompt": prompt_cache["formatted_text"],
    }


def _run_singleturn_cached_mode(pipeline, args, weight_dtype, generator, default_prompt_cache):
    cached_data_dir = _resolve_cached_data_dir(args)
    local_rank, world_size = _get_distributed_context()
    results = []
    shared_prompt_cache = default_prompt_cache
    if args.shared_prompt_cache is not None:
        shared_prompt_cache = _load_shared_prompt_cache(args.shared_prompt_cache)

    if args.cached_sample_path is not None:
        cache_path = _resolve_cache_path(args.cached_sample_path, cached_data_dir)
        payload = load_singleturn_cache_payload(cache_path)
        if "source_latent_mean" in payload or "target_latent_mean" in payload:
            raise ValueError(
                f"Cached SingleTurn sample {cache_path} uses the deprecated instructpix2pix posterior payload. "
                "Re-run CORNE object-removal preprocess."
            )
        payload_mode = payload.get("mode")
        if not (is_supported_singleturn_object_removal_mode(payload_mode) or payload_mode == "singleturn_object_removal_sam_strict_keyframe_cache_v1"):
            raise ValueError(
                f"Cached SingleTurn sample {cache_path} has unsupported mode={payload_mode!r}. "
                "Re-run CORNE object-removal preprocess / patch to generate a supported cache."
            )
        used_mask_sam = args.singleturn_mask_condition_source == "mask_sam"
        if used_mask_sam:
            mask_frame_latent = payload.get("mask_sam_latent")
        else:
            mask_frame_latent = payload.get("mask_check_latent")
            if mask_frame_latent is None:
                mask_frame_latent = payload.get("mask_frame_latent")
        missing = [key for key in ("source_frame_latent",) if key not in payload]
        if mask_frame_latent is None:
            missing.append("mask_sam_latent" if used_mask_sam else "mask_check_latent")
        if missing:
            raise ValueError(f"Cached SingleTurn sample {cache_path} is missing keys: {missing}.")
        prompt_cache = shared_prompt_cache
        if args.shared_prompt_cache is None and payload.get("shared_prompt_cache") is not None:
            prompt_cache = _load_shared_prompt_cache(_resolve_cache_path(payload["shared_prompt_cache"], cached_data_dir))

        sample = {
            "mask_frame_latent": mask_frame_latent,
            "source_frame_latent": payload["source_frame_latent"],
            "cache_path": cache_path,
            "source_image": payload.get("source_image", ""),
            "bg_image": payload.get("bg_image", ""),
            "mask_check_image": payload.get("mask_check_image", ""),
            "mask_frame_image": payload.get("mask_frame_image", ""),
            "mask_sam_image": payload.get("mask_sam_image", ""),
            "used_mask_sam": used_mask_sam,
            "total_frames": int(payload.get("total_frames", payload.get("full_latents", torch.empty(0, args.singleturn_total_frames)).shape[1] if "full_latents" in payload else args.singleturn_total_frames)),
            "cache_mode": payload.get("mode", ""),
            "idx": 0,
        }
        output_name = args.output_name or Path(cache_path).stem
        sample_output_dir = args.output_dir if args.output_name else os.path.join(args.output_dir, output_name)
        results.append(
            _run_singleturn_cached_sample(
                pipeline=pipeline,
                args=args,
                weight_dtype=weight_dtype,
                generator=generator,
                sample=sample,
                prompt_cache=prompt_cache,
                output_dir=sample_output_dir,
                stem="singleturn",
            )
        )
    else:
        dataset = CachedSingleTurnLatentDataset(
            args.cached_data_meta,
            cached_data_dir,
            corruption_frames=args.singleturn_cache_corruption_frames,
            restoration_frames=args.singleturn_cache_restoration_frames,
            interpolation_gamma=args.singleturn_cache_interpolation_gamma,
            mask_condition_source=args.singleturn_mask_condition_source,
        )
        start_index = max(0, int(args.cached_start_index))
        if start_index >= len(dataset):
            raise ValueError(
                f"--cached_start_index {start_index} is out of range for cached dataset of length {len(dataset)}."
            )

        end_index = len(dataset)
        if args.cached_num_samples is not None:
            if args.cached_num_samples <= 0:
                raise ValueError("--cached_num_samples must be positive when provided.")
            end_index = min(len(dataset), start_index + int(args.cached_num_samples))

        indices = list(range(start_index, end_index))
        if not indices:
            raise ValueError("No cached samples selected for inference.")

        if world_size > 1:
            indices = indices[local_rank::world_size]
            if not indices:
                return {"outputs": [], "summary": None}

        dataset = Subset(dataset, indices)
        dataloader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": args.cached_num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": args.cached_num_workers > 0,
        }
        if args.cached_num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = args.cached_prefetch_factor
        dataloader = DataLoader(dataset, **dataloader_kwargs)

        for batch in tqdm(dataloader, desc="Running cached SingleTurn inference"):
            cache_path = _first_item(batch["cache_path"])
            batch_index = int(_first_item(batch["idx"]))
            sample_name = args.output_name or Path(str(cache_path)).stem
            sample_dir_name = f"{batch_index:06d}_{sample_name}"
            sample_output_root = args.output_dir if world_size == 1 else os.path.join(args.output_dir, f"rank{local_rank}")
            sample_output_dir = os.path.join(sample_output_root, sample_dir_name)

            prompt_cache = shared_prompt_cache
            if args.shared_prompt_cache is None:
                prompt_cache = {
                    "prompt_embeds": batch["prompt_embeds"],
                    "prompt_seq_len": int(_first_item(batch["prompt_seq_len"])),
                    "text": _first_item(batch["text"]),
                    "formatted_text": _first_item(batch["formatted_text"]),
                }

            sample = {
                "mask_frame_latent": batch["mask_frame_latent"],
                "source_frame_latent": batch["source_frame_latent"],
                "cache_path": cache_path,
                "source_image": _first_item(batch["source_image"]),
                "bg_image": _first_item(batch["bg_image"]),
                "mask_check_image": _first_item(batch["mask_check_image"]),
                "mask_frame_image": _first_item(batch["mask_frame_image"]),
                "mask_sam_image": _first_item(batch["mask_sam_image"]),
                "used_mask_sam": bool(_first_item(batch["used_mask_sam"])),
                "total_frames": int(_first_item(batch["total_frames"])),
                "cache_mode": _first_item(batch["cache_mode"]),
                "idx": batch_index,
            }

            results.append(
                _run_singleturn_cached_sample(
                    pipeline=pipeline,
                    args=args,
                    weight_dtype=weight_dtype,
                    generator=generator,
                    sample=sample,
                    prompt_cache=prompt_cache,
                    output_dir=sample_output_dir,
                    stem="singleturn",
                )
            )

    summary_root = args.output_dir if world_size == 1 else os.path.join(args.output_dir, f"rank{local_rank}")
    summary_path = os.path.join(summary_root, "cached_infer_manifest.json")
    os.makedirs(summary_root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return {"outputs": results, "summary": summary_path}


def parse_args():
    parser = argparse.ArgumentParser(description="SingleTurn CORNE object-removal inference")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Base Wan model path.")
    parser.add_argument("--image_path", type=str, default=None, help="Source image path for direct image-mode inference.")
    parser.add_argument("--mask_path", type=str, default=None, help="Mask path for direct image-mode inference.")
    parser.add_argument("--prompt", type=str, default=None, help="Deprecated and ignored. Prompt is fixed for CORNE object-removal mode.")
    parser.add_argument("--cached_sample_path", type=str, default=None, help="Path to one cached SingleTurn sample (.pt).")
    parser.add_argument("--cached_data_meta", type=str, default=None, help="Manifest for cached SingleTurn samples.")
    parser.add_argument("--cached_data_dir", type=str, default=None, help="Root directory used to resolve relative cache paths.")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Raw eval directory with img/images and mask/masks folders.")
    parser.add_argument("--raw_selected_triplets", type=str, default=None, help="Optional selected_triplets.json path for raw folder mode.")
    parser.add_argument("--shared_prompt_cache", type=str, default=None, help="Optional shared prompt embedding cache.")
    parser.add_argument("--cached_start_index", type=int, default=0, help="Start index when iterating over cached manifests.")
    parser.add_argument("--cached_num_samples", type=int, default=None, help="Optional limit when iterating over cached manifests.")
    parser.add_argument("--cached_num_workers", type=int, default=2, help="CPU workers used to load cached samples.")
    parser.add_argument("--cached_prefetch_factor", type=int, default=2, help="Prefetch factor for cached sample loading.")
    parser.add_argument("--output_dir", type=str, default="outputs/singleturn", help="Directory for outputs.")
    parser.add_argument("--output_name", type=str, default=None, help="Optional filename stem for outputs.")
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Wan config path.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA checkpoint.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA merge multiplier.")
    parser.add_argument("--enable_refinement", action="store_true", help="Run a one-step refinement pass after the coarse 50-step denoise.")
    parser.add_argument("--refinement_lora_path", type=str, default=None, help="LoRA checkpoint used only for the refinement pass.")
    parser.add_argument("--refinement_lora_alpha", type=float, default=1.0, help="Refinement LoRA merge multiplier.")
    parser.add_argument("--refinement_guidance_scale", type=float, default=1.0, help="Guidance scale used only during the one-step refinement pass.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Optional negative prompt.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument(
        "--sample_size",
        type=int,
        nargs="+",
        default=[480, 832],
        help="Letterboxed sample size used before VAE encode. Pass one value for square or two values for HEIGHT WIDTH.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--reset_seed_per_sample",
        action="store_true",
        help="Recreate the same seeded generator at the start of every raw-folder sample.",
    )
    parser.add_argument("--fps", type=int, default=4, help="GIF playback FPS.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Inference weight dtype.")
    parser.add_argument(
        "--singleturn_total_frames",
        type=int,
        default=None,
        help="Total latent frames used in direct image-mode SingleTurn inference. Defaults to corruption/restoration layout size.",
    )
    parser.add_argument(
        "--singleturn_endpoint_mode",
        action="store_true",
        help="Endpoint mode: skip interpolation frames, use [mask, pred, source, target] 4-frame layout.",
    )
    parser.add_argument(
        "--rollback_cross_attn",
        action="store_true",
        help="Disable decoupled cross-attention; use unified prompt for all frames (RoPE trick remains).",
    )
    parser.add_argument(
        "--rollback_rope",
        action="store_true",
        help="Disable RoPE temporal trick; use standard [0,1,2,...] indices instead of shared positions.",
    )
    parser.add_argument(
        "--singleturn_cache_corruption_frames",
        type=int,
        default=2,
        help="Number of interpolated frames between source and noisy anchor in the cached layout.",
    )
    parser.add_argument(
        "--singleturn_cache_restoration_frames",
        type=int,
        default=5,
        help="Number of interpolated frames between noisy anchor and target in the cached layout.",
    )
    parser.add_argument(
        "--singleturn_cache_interpolation_gamma",
        type=float,
        default=2.0,
        help="Gamma for non-linear interpolation when cached full_latents are absent.",
    )
    parser.add_argument(
        "--singleturn_mask_condition_source",
        type=str,
        default="mask_check",
        choices=["mask_check", "mask_sam"],
        help="Cached inference mask condition latent source.",
    )
    parser.add_argument(
        "--enable_mask_blending",
        action="store_true",
        help="Blend saved last-frame PNGs with the source image using the predicted mask frame as soft alpha.",
    )
    parser.add_argument(
        "--mask_blend_threshold",
        type=float,
        default=0.5,
        help="Threshold applied to the predicted mask frame before ObjectClear-style blend alpha dilation.",
    )
    parser.add_argument(
        "--mask_blend_dilate_kernel_size",
        type=int,
        default=31,
        help="Odd elliptical dilation kernel size used for ObjectClear-style blend alpha.",
    )
    parser.add_argument(
        "--mask_blend_blur_kernel_size",
        type=int,
        default=15,
        help="Odd Gaussian blur kernel size used for ObjectClear-style blend alpha.",
    )
    parser.add_argument(
        "--mask_blend_blur_sigma",
        type=float,
        default=4.0,
        help="Gaussian sigma used for ObjectClear-style blend alpha.",
    )
    parser.add_argument(
        "--enable_uncertainty_viz",
        action="store_true",
        help="Save experimental heatmap/overlay files from the last denoise-step latent update magnitudes.",
    )
    parser.add_argument(
        "--uncertainty_last_steps",
        type=int,
        default=10,
        help="Number of final denoise steps to average for --enable_uncertainty_viz.",
    )
    parser.add_argument(
        "--enable_trajectory_refinement",
        action="store_true",
        help="Apply one test-time latent trajectory interpolation repair before the final denoise steps.",
    )
    parser.add_argument(
        "--trajectory_refinement_remaining_steps",
        type=int,
        default=10,
        help="Apply trajectory refinement when this many denoise steps remain.",
    )
    parser.add_argument(
        "--trajectory_refinement_gamma",
        type=float,
        default=None,
        help="Gamma for trajectory refinement interpolation. Defaults to --singleturn_cache_interpolation_gamma.",
    )
    parser.add_argument(
        "--trajectory_refinement_strength",
        type=float,
        default=1.0,
        help="Blend strength for trajectory refinement interpolation, where 1 fully replaces middle trajectory frames.",
    )
    args = parser.parse_args()

    cache_mode = args.cached_sample_path is not None or args.cached_data_meta is not None
    image_mode = args.image_path is not None or args.mask_path is not None
    raw_mode = args.raw_data_dir is not None
    if sum([cache_mode, image_mode, raw_mode]) != 1:
        raise ValueError("Choose exactly one mode: direct image, cached, or raw folder.")
    if image_mode:
        if args.image_path is None or args.mask_path is None:
            raise ValueError("Image mode requires both --image_path and --mask_path.")
    elif cache_mode:
        if args.cached_sample_path is not None and args.cached_data_meta is not None:
            raise ValueError("Choose either --cached_sample_path or --cached_data_meta, not both.")
        if args.cached_sample_path is None and args.cached_data_meta is None:
            raise ValueError("Cached mode requires --cached_sample_path or --cached_data_meta.")
        if args.cached_num_workers < 0:
            raise ValueError("--cached_num_workers must be non-negative.")
        if args.cached_prefetch_factor <= 0:
            raise ValueError("--cached_prefetch_factor must be positive.")
    else:
        if not os.path.isdir(args.raw_data_dir):
            raise ValueError(f"--raw_data_dir must be an existing directory, got {args.raw_data_dir}.")
        if args.raw_selected_triplets is not None and not os.path.exists(args.raw_selected_triplets):
            raise ValueError(f"--raw_selected_triplets does not exist: {args.raw_selected_triplets}")

    if args.enable_refinement and not args.refinement_lora_path:
        raise ValueError("--enable_refinement requires --refinement_lora_path.")
    if (not args.enable_refinement) and args.refinement_lora_path is not None:
        raise ValueError("--refinement_lora_path requires --enable_refinement.")
    if args.singleturn_cache_corruption_frames < 0 or args.singleturn_cache_restoration_frames < 0:
        raise ValueError("--singleturn_cache_corruption_frames and --singleturn_cache_restoration_frames must be non-negative.")
    if args.singleturn_endpoint_mode:
        args.singleturn_total_frames = SINGLETURN_ENDPOINT_TOTAL_FRAMES
    else:
        expected_singleturn_total_frames = compute_singleturn_object_removal_total_frames(
            args.singleturn_cache_corruption_frames,
            args.singleturn_cache_restoration_frames,
        )
        if args.singleturn_total_frames is None:
            args.singleturn_total_frames = expected_singleturn_total_frames
        elif args.singleturn_total_frames != expected_singleturn_total_frames:
            raise ValueError(
                "--singleturn_total_frames must match --singleturn_cache_corruption_frames/"
                "--singleturn_cache_restoration_frames layout, "
                f"got {args.singleturn_total_frames} vs expected {expected_singleturn_total_frames}."
            )
    if args.singleturn_cache_interpolation_gamma <= 0:
        raise ValueError("--singleturn_cache_interpolation_gamma must be positive.")
    if args.uncertainty_last_steps <= 0:
        raise ValueError("--uncertainty_last_steps must be positive.")
    if args.trajectory_refinement_remaining_steps <= 0:
        raise ValueError("--trajectory_refinement_remaining_steps must be positive.")
    if args.trajectory_refinement_strength < 0:
        raise ValueError("--trajectory_refinement_strength must be non-negative.")
    if args.trajectory_refinement_gamma is None:
        args.trajectory_refinement_gamma = args.singleturn_cache_interpolation_gamma
    elif args.trajectory_refinement_gamma <= 0:
        raise ValueError("--trajectory_refinement_gamma must be positive.")

    args.sample_size = normalize_singleturn_sample_size(args.sample_size)
    if any(dim % 16 != 0 for dim in args.sample_size):
        raise ValueError(f"--sample_size must be divisible by 16, got {args.sample_size}.")
    return args


def get_weight_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16


def main():
    args = parse_args()

    device = _resolve_runtime_device()
    weight_dtype = get_weight_dtype(args.dtype, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    config = OmegaConf.load(args.config_path)

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
            "tokenizer",
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
            "text_encoder",
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()
    vae = AutoencoderKLWan.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
            "vae",
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).eval().to(dtype=weight_dtype)
    transformer = WanTransformer3DModel.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
            "transformer",
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipeline = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    ).to(device)

    if args.lora_path:
        pipeline = merge_lora(
            pipeline,
            args.lora_path,
            args.lora_alpha,
            device=device,
            dtype=weight_dtype,
            transformer_only=True,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    default_prompt_cache = _encode_fixed_prompt(tokenizer, text_encoder, device, weight_dtype)

    if args.raw_data_dir is not None:
        result = _run_singleturn_raw_folder_mode(
            pipeline=pipeline,
            args=args,
            weight_dtype=weight_dtype,
            generator=generator,
        )
    elif args.cached_sample_path is not None or args.cached_data_meta is not None:
        result = _run_singleturn_cached_mode(
            pipeline=pipeline,
            args=args,
            weight_dtype=weight_dtype,
            generator=generator,
            default_prompt_cache=default_prompt_cache,
        )
    else:
        result = _run_singleturn_image_mode(
            pipeline=pipeline,
            args=args,
            weight_dtype=weight_dtype,
            generator=generator,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
