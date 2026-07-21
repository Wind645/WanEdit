#!/usr/bin/env python

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from shutil import copyfile
from typing import Any, Optional

import torch
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

from videox_fun.data.singleturn_dataset import (
    CachedSingleTurnLatentDataset,
    IMAGE_EXTENSIONS,
    load_singleturn_cache_payload,
)
from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel, WanTransformer3DModel
from videox_fun.pipeline import WanPipeline
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.singleturn_utils import (
    CORNE_SINGLETURN_PROMPT,
    SINGLETURN_OBJECT_REMOVAL_CACHE_MODE,
    SINGLETURN_TOTAL_FRAMES,
    compute_singleturn_object_removal_total_frames,
    generate_singleturn_sample,
    generate_singleturn_sample_from_latents,
    normalize_singleturn_sample_size,
    preprocess_singleturn_image,
    preprocess_singleturn_mask_frame,
    preprocess_singleturn_mask,
    refine_singleturn_sample_from_latents,
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


def _resolve_singleturn_total_frames(
    corruption_frame: int,
    restoration_frame: int,
    *,
    source: str,
    cache_mode: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> int:
    try:
        return compute_singleturn_object_removal_total_frames(corruption_frame, restoration_frame)
    except ValueError as exc:
        details = [f"{source}=({int(corruption_frame)}, {int(restoration_frame)})"]
        if cache_mode is not None:
            details.append(f"cache_mode={cache_mode!r}")
        if cache_path:
            details.append(f"cache_path={cache_path}")
        raise ValueError(f"Invalid SingleTurn coarse frame layout for {', '.join(details)}: {exc}") from exc


def _load_shared_prompt_cache(shared_prompt_cache: str):
    payload = load_singleturn_cache_payload(shared_prompt_cache)
    missing = [key for key in ("prompt_embeds", "prompt_seq_len") if key not in payload]
    if missing:
        raise ValueError(f"Shared prompt cache {shared_prompt_cache} is missing keys: {missing}")
    return {
        "prompt_embeds": payload["prompt_embeds"],
        "prompt_seq_len": int(payload["prompt_seq_len"]),
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
    video_format: str,
):
    os.makedirs(output_dir, exist_ok=True)
    return save_singleturn_outputs(
        full_frames=generation["full_frames"],
        tail_frames=generation["tail_frames"],
        output_dir=output_dir,
        stem=stem,
        fps=fps,
        video_format=video_format,
    )


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
    video_format: str,
):
    output_paths = _save_singleturn_variant(
        output_dir=output_dir,
        stem=stem,
        generation=generation,
        fps=fps,
        video_format=video_format,
    )
    metadata_path = _write_singleturn_metadata(
        output_dir=output_dir,
        stem=stem,
        metadata={**metadata, "outputs": output_paths},
    )
    return output_paths, metadata_path


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _resolve_ranked_output_path(output_dir: str, filename: str, *, local_rank: int, world_size: int) -> str:
    if world_size == 1:
        return os.path.join(output_dir, filename)
    stem, suffix = os.path.splitext(filename)
    return os.path.join(output_dir, f"{stem}.rank{local_rank}{suffix}")


def _index_raw_folder(directory: str, extensions: tuple[str, ...]) -> tuple[dict[str, str], dict[str, list[str]]]:
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Expected raw-folder input directory does not exist: {directory}")

    normalized_extensions = {extension.lower() for extension in extensions}
    index: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for candidate in sorted(path.iterdir(), key=lambda item: item.name):
        if not candidate.is_file() or candidate.suffix.lower() not in normalized_extensions:
            continue
        key = candidate.stem
        candidate_str = str(candidate)
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(candidate_str)
            continue
        index[key] = candidate_str
    return index, duplicates


def _discover_raw_folder_samples(
    *,
    source_dir: str,
    mask_dir: str,
    gt_dir: str,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_index, source_duplicates = _index_raw_folder(source_dir, extensions)
    mask_index, mask_duplicates = _index_raw_folder(mask_dir, extensions)
    gt_index, gt_duplicates = _index_raw_folder(gt_dir, extensions)

    source_keys = set(source_index)
    mask_keys = set(mask_index)
    gt_keys = set(gt_index)
    matched_keys = sorted(source_keys & mask_keys & gt_keys)
    all_keys = sorted(source_keys | mask_keys | gt_keys)

    samples = [
        {
            "index": index,
            "key": key,
            "source_path": source_index[key],
            "mask_path": mask_index[key],
            "gt_path": gt_index[key],
        }
        for index, key in enumerate(matched_keys)
    ]

    missing = []
    for key in all_keys:
        missing_in = []
        if key not in source_index:
            missing_in.append("source")
        if key not in mask_index:
            missing_in.append("mask")
        if key not in gt_index:
            missing_in.append("gt")
        if missing_in:
            missing.append(
                {
                    "key": key,
                    "missing_in": missing_in,
                    "source_path": source_index.get(key, ""),
                    "mask_path": mask_index.get(key, ""),
                    "gt_path": gt_index.get(key, ""),
                }
            )

    duplicate_keys = {
        "source": source_duplicates,
        "mask": mask_duplicates,
        "gt": gt_duplicates,
    }
    for directory_name, duplicates in duplicate_keys.items():
        for key, paths in sorted(duplicates.items()):
            _warn(
                f"Duplicate raw-folder basename {key!r} detected in {directory_name}_dir; "
                f"keeping {paths[0]} and ignoring {len(paths) - 1} additional file(s)."
            )

    if missing:
        _warn(
            f"Raw-folder mode found {len(missing)} key(s) outside the source/mask/gt intersection; "
            "see raw_infer_missing_summary.json for details."
        )

    summary = {
        "source_dir": source_dir,
        "mask_dir": mask_dir,
        "gt_dir": gt_dir,
        "extensions": list(extensions),
        "matched_count": len(samples),
        "missing_count": len(missing),
        "duplicate_counts": {
            "source": len(source_duplicates),
            "mask": len(mask_duplicates),
            "gt": len(gt_duplicates),
        },
        "missing": missing,
        "duplicates": duplicate_keys,
    }
    return samples, summary


def _write_json(output_path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def _encode_singleturn_frame_latent(pipeline, frame_tensor: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    return pipeline.vae.encode(
        frame_tensor.to(device=pipeline._execution_device, dtype=weight_dtype).permute(0, 2, 1, 3, 4)
    )[0].mode()


def _sample_generator_for_index(device: torch.device, seed: int, index: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed) + int(index))


def _save_raw_folder_generation(
    *,
    output_dir: str,
    sample_index: int,
    generation: dict,
    fps: int,
    video_format: str,
) -> dict[str, str]:
    videos_dir = os.path.join(output_dir, "generated_videos")
    last_frame_dir = os.path.join(output_dir, "generated_last_frame")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(last_frame_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output_dir, prefix=f".raw_singleturn_{sample_index:06d}_") as tmpdir:
        tmp_outputs = save_singleturn_outputs(
            full_frames=generation["full_frames"],
            tail_frames=generation["tail_frames"],
            output_dir=tmpdir,
            stem="singleturn",
            fps=fps,
            video_format=video_format,
        )
        final_video_path = os.path.join(videos_dir, f"{sample_index:06d}.{video_format}")
        final_last_frame_path = os.path.join(last_frame_dir, f"{sample_index:06d}.png")
        copyfile(tmp_outputs["full_video"], final_video_path)
        copyfile(tmp_outputs["full_last_frame"], final_last_frame_path)

    return {
        "generated_video_path": final_video_path,
        "generated_last_frame_path": final_last_frame_path,
    }


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
    video_format: str,
):
    if refined_generation is None:
        output_paths = _save_singleturn_variant(
            output_dir=output_dir,
            stem=stem,
            generation=coarse_generation,
            fps=fps,
            video_format=video_format,
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
        video_format=video_format,
    )
    refined_output_paths = _save_singleturn_variant(
        output_dir=output_dir,
        stem=f"{stem}_refined",
        generation=refined_generation,
        fps=fps,
        video_format=video_format,
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


def _run_singleturn_image_mode(pipeline, args, weight_dtype, generator):
    total_frames = _resolve_singleturn_total_frames(
        args.corruption_frame,
        args.restoration_frame,
        source="--corruption_frame/--restoration_frame",
    )
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
            mask_sam_tensor=mask_frame_tensor,
            source_tensor=source_tensor,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            weight_dtype=weight_dtype,
            corruption_frame=args.corruption_frame,
            restoration_frame=args.restoration_frame,
        )
        refined_generation = None
        if args.enable_refinement:
            if total_frames != SINGLETURN_TOTAL_FRAMES:
                raise ValueError(
                    "The current refinement stage only supports 8-frame coarse latents. "
                    f"Got singleturn_total_frames={total_frames}."
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
            "corruption_frame": args.corruption_frame,
            "restoration_frame": args.restoration_frame,
            "singleturn_total_frames": total_frames,
            "input_previews": preview_paths,
            "coarse_lora_path": args.lora_path or "",
            "coarse_lora_alpha": args.lora_alpha,
            "refinement_enabled": bool(args.enable_refinement),
            "refinement_lora_path": args.refinement_lora_path or "",
            "refinement_lora_alpha": args.refinement_lora_alpha,
            "refinement_guidance_scale": args.refinement_guidance_scale,
        },
        fps=args.fps,
        video_format=args.video_format,
    )

    return {
        "outputs": coarse_output_paths,
        "refined_outputs": refined_output_paths,
        "metadata": metadata_path,
    }


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
        total_frames = _resolve_singleturn_total_frames(
            args.corruption_frame,
            args.restoration_frame,
            source="--corruption_frame/--restoration_frame",
            cache_mode=sample.get("cache_mode"),
            cache_path=sample.get("cache_path"),
        )
        coarse_generation = generate_singleturn_sample_from_latents(
            pipeline=pipeline,
            mask_sam_latent=sample["mask_sam_latent"],
            source_frame_latent=sample["source_frame_latent"],
            prompt_embeds=prompt_cache["prompt_embeds"],
            prompt_seq_len=prompt_cache["prompt_seq_len"],
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            weight_dtype=weight_dtype,
            corruption_frame=args.corruption_frame,
            restoration_frame=args.restoration_frame,
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
    if sample.get("source_image") and mask_frame_path:
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
            "corruption_frame": args.corruption_frame,
            "restoration_frame": args.restoration_frame,
            "singleturn_total_frames": total_frames,
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
        video_format=args.video_format,
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
        if payload.get("mode") != SINGLETURN_OBJECT_REMOVAL_CACHE_MODE:
            raise ValueError(
                f"Cached SingleTurn sample {cache_path} has unsupported mode={payload.get('mode')!r}. "
                f"Expected {SINGLETURN_OBJECT_REMOVAL_CACHE_MODE!r}. Re-run CORNE object-removal preprocess."
            )
        missing = [
            key
            for key in ("mask_sam_latent", "source_frame_latent", "noisy_anchor_latent", "target_latent")
            if key not in payload
        ]
        if missing:
            raise ValueError(f"Cached SingleTurn sample {cache_path} is missing keys: {missing}.")

        prompt_cache = shared_prompt_cache
        if args.shared_prompt_cache is None and payload.get("shared_prompt_cache") is not None:
            prompt_cache = _load_shared_prompt_cache(_resolve_cache_path(payload["shared_prompt_cache"], cached_data_dir))

        sample = {
            "mask_sam_latent": payload["mask_sam_latent"],
            "source_frame_latent": payload["source_frame_latent"],
            "noisy_anchor_latent": payload["noisy_anchor_latent"],
            "target_latent": payload["target_latent"],
            "cache_path": cache_path,
            "source_image": payload.get("source_image", ""),
            "bg_image": payload.get("bg_image", ""),
            "mask_check_image": payload.get("mask_check_image", ""),
            "mask_frame_image": payload.get("mask_frame_image", ""),
            "mask_sam_image": payload.get("mask_sam_image", ""),
            "used_mask_sam": bool(payload.get("used_mask_sam", False)),
            "singleturn_sample_size": payload.get("singleturn_sample_size", []),
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
        dataset = CachedSingleTurnLatentDataset(args.cached_data_meta, cached_data_dir)
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
                "mask_sam_latent": batch["mask_sam_latent"],
                "source_frame_latent": batch["source_frame_latent"],
                "noisy_anchor_latent": batch["noisy_anchor_latent"],
                "target_latent": batch["target_latent"],
                "cache_path": cache_path,
                "source_image": _first_item(batch["source_image"]),
                "bg_image": _first_item(batch["bg_image"]),
                "mask_check_image": _first_item(batch["mask_check_image"]),
                "mask_frame_image": _first_item(batch["mask_frame_image"]),
                "mask_sam_image": _first_item(batch["mask_sam_image"]),
                "used_mask_sam": bool(_first_item(batch["used_mask_sam"])),
                "singleturn_sample_size": _first_item(batch["singleturn_sample_size"]),
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


def _run_singleturn_raw_folder_mode(
    pipeline,
    args,
    weight_dtype: torch.dtype,
    default_prompt_cache: dict,
):
    total_frames = _resolve_singleturn_total_frames(
        args.corruption_frame,
        args.restoration_frame,
        source="--corruption_frame/--restoration_frame",
    )
    local_rank, world_size = _get_distributed_context()
    all_samples, missing_summary = _discover_raw_folder_samples(
        source_dir=args.raw_source_dir,
        mask_dir=args.raw_mask_dir,
        gt_dir=args.raw_gt_dir,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    missing_summary_path = _write_json(
        _resolve_ranked_output_path(
            args.output_dir,
            "raw_infer_missing_summary.json",
            local_rank=local_rank,
            world_size=world_size,
        ),
        missing_summary,
    )

    if not all_samples:
        raise ValueError(
            "Raw-folder mode found no matched samples across source/mask/gt directories. "
            f"See {missing_summary_path} for missing-key details."
        )

    assigned_samples = all_samples[local_rank::world_size] if world_size > 1 else all_samples
    resized_gt_dir = os.path.join(args.output_dir, "resized_gt")
    resized_mask_dir = os.path.join(args.output_dir, "resized_mask")
    os.makedirs(resized_gt_dir, exist_ok=True)
    os.makedirs(resized_mask_dir, exist_ok=True)

    results = []
    for sample in tqdm(
        assigned_samples,
        desc="Running raw-folder SingleTurn inference",
        disable=not assigned_samples,
    ):
        sample_index = int(sample["index"])
        sample_key = sample["key"]
        resized_gt_path = os.path.join(resized_gt_dir, f"{sample_index:06d}.png")
        resized_mask_path = os.path.join(resized_mask_dir, f"{sample_index:06d}.png")
        record = {
            "index": sample_index,
            "key": sample_key,
            "source_path": sample["source_path"],
            "mask_path": sample["mask_path"],
            "gt_path": sample["gt_path"],
            "resized_gt_path": resized_gt_path,
            "resized_mask_path": resized_mask_path,
            "generated_video_path": "",
            "generated_last_frame_path": "",
            "corruption_frame": args.corruption_frame,
            "restoration_frame": args.restoration_frame,
            "singleturn_total_frames": total_frames,
            "status": "pending",
        }

        try:
            source_tensor = preprocess_singleturn_image(sample["source_path"], args.sample_size)
            gt_tensor = preprocess_singleturn_image(sample["gt_path"], args.sample_size)
            mask_tensor = preprocess_singleturn_mask(sample["mask_path"], args.sample_size)
            mask_frame_tensor = preprocess_singleturn_mask_frame(sample["mask_path"], args.sample_size)

            _tensor_image_to_pil(gt_tensor[0, 0]).save(resized_gt_path)
            _tensor_mask_to_pil(mask_tensor[0, 0]).save(resized_mask_path)

            sample_generator = _sample_generator_for_index(pipeline._execution_device, args.seed, sample_index)
            with torch.no_grad():
                coarse_generation = generate_singleturn_sample_from_latents(
                    pipeline=pipeline,
                    mask_sam_latent=_encode_singleturn_frame_latent(pipeline, mask_frame_tensor, weight_dtype),
                    source_frame_latent=_encode_singleturn_frame_latent(pipeline, source_tensor, weight_dtype),
                    prompt_embeds=default_prompt_cache["prompt_embeds"],
                    prompt_seq_len=default_prompt_cache["prompt_seq_len"],
                    negative_prompt=args.negative_prompt,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=sample_generator,
                    weight_dtype=weight_dtype,
                    corruption_frame=args.corruption_frame,
                    restoration_frame=args.restoration_frame,
                )
                final_generation = coarse_generation
                if args.enable_refinement:
                    if total_frames != SINGLETURN_TOTAL_FRAMES:
                        raise ValueError(
                            "The current refinement stage only supports 8-frame coarse latents. "
                            f"Got singleturn_total_frames={total_frames} for raw sample key={sample_key!r}."
                        )
                    pipeline = _enable_refinement_lora(
                        pipeline,
                        args,
                        pipeline._execution_device,
                        weight_dtype,
                    )
                    try:
                        final_generation = refine_singleturn_sample_from_latents(
                            pipeline=pipeline,
                            coarse_latents=coarse_generation["latents"],
                            prompt_embeds=default_prompt_cache["prompt_embeds"],
                            prompt_seq_len=default_prompt_cache["prompt_seq_len"],
                            negative_prompt=args.negative_prompt,
                            guidance_scale=args.refinement_guidance_scale,
                            weight_dtype=weight_dtype,
                        )
                    finally:
                        pipeline = _restore_coarse_lora(
                            pipeline,
                            args,
                            pipeline._execution_device,
                            weight_dtype,
                        )

            output_paths = _save_raw_folder_generation(
                output_dir=args.output_dir,
                sample_index=sample_index,
                generation=final_generation,
                fps=args.fps,
                video_format=args.video_format,
            )
            record.update(output_paths)
            record["status"] = "ok"
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            _warn(f"Raw-folder inference failed for key={sample_key!r} index={sample_index}: {exc}")

        results.append(record)

    manifest_path = _write_json(
        _resolve_ranked_output_path(
            args.output_dir,
            "raw_infer_manifest.json",
            local_rank=local_rank,
            world_size=world_size,
        ),
        results,
    )
    return {
        "outputs": results,
        "summary": manifest_path,
        "missing_summary": missing_summary_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="SingleTurn CORNE object-removal inference")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Base Wan model path.")
    parser.add_argument("--image_path", type=str, default=None, help="Source image path for direct image-mode inference.")
    parser.add_argument("--mask_path", type=str, default=None, help="Mask path for direct image-mode inference.")
    parser.add_argument("--prompt", type=str, default=None, help="Deprecated and ignored. Prompt is fixed for CORNE object-removal mode.")
    parser.add_argument("--cached_sample_path", type=str, default=None, help="Path to one cached SingleTurn sample (.pt).")
    parser.add_argument("--cached_data_meta", type=str, default=None, help="Manifest for cached SingleTurn samples.")
    parser.add_argument("--cached_data_dir", type=str, default=None, help="Root directory used to resolve relative cache paths.")
    parser.add_argument("--shared_prompt_cache", type=str, default=None, help="Optional shared prompt embedding cache.")
    parser.add_argument("--raw_source_dir", type=str, default=None, help="Raw-folder source image directory.")
    parser.add_argument("--raw_mask_dir", type=str, default=None, help="Raw-folder mask directory.")
    parser.add_argument("--raw_gt_dir", type=str, default=None, help="Raw-folder ground-truth directory.")
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
    parser.add_argument("--fps", type=int, default=4, help="GIF playback FPS.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Inference weight dtype.")
    parser.add_argument(
        "--corruption_frame",
        type=int,
        default=2,
        help="Number of interpolation frames inserted between source and noisy-anchor.",
    )
    parser.add_argument(
        "--restoration_frame",
        type=int,
        default=14,
        help="Number of interpolation frames inserted between noisy-anchor and target.",
    )
    parser.add_argument(
        "--video_format",
        type=str,
        default="gif",
        choices=["gif", "mp4"],
        help="Saved preview video format. mp4 uses H.264 via ffmpeg/imageio.",
    )
    args = parser.parse_args()

    cache_mode = args.cached_sample_path is not None or args.cached_data_meta is not None
    image_mode = args.image_path is not None or args.mask_path is not None
    raw_mode = args.raw_source_dir is not None or args.raw_mask_dir is not None or args.raw_gt_dir is not None
    selected_mode_count = int(cache_mode) + int(image_mode) + int(raw_mode)
    if selected_mode_count != 1:
        raise ValueError(
            "Choose exactly one inference mode: direct image mode, cached mode, or raw-folder mode."
        )

    if raw_mode:
        if args.raw_source_dir is None or args.raw_mask_dir is None or args.raw_gt_dir is None:
            raise ValueError(
                "Raw-folder mode requires --raw_source_dir, --raw_mask_dir, and --raw_gt_dir together."
            )
    elif image_mode:
        if args.image_path is None or args.mask_path is None:
            raise ValueError("Image mode requires both --image_path and --mask_path.")
    else:
        if args.cached_sample_path is not None and args.cached_data_meta is not None:
            raise ValueError("Choose either --cached_sample_path or --cached_data_meta, not both.")
        if args.cached_sample_path is None and args.cached_data_meta is None:
            raise ValueError("Cached mode requires --cached_sample_path or --cached_data_meta.")
        if args.cached_num_workers < 0:
            raise ValueError("--cached_num_workers must be non-negative.")
        if args.cached_prefetch_factor <= 0:
            raise ValueError("--cached_prefetch_factor must be positive.")

    if args.enable_refinement and not args.refinement_lora_path:
        raise ValueError("--enable_refinement requires --refinement_lora_path.")
    if (not args.enable_refinement) and args.refinement_lora_path is not None:
        raise ValueError("--refinement_lora_path requires --enable_refinement.")
    _resolve_singleturn_total_frames(
        args.corruption_frame,
        args.restoration_frame,
        source="--corruption_frame/--restoration_frame",
    )

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

    if args.raw_source_dir is not None or args.raw_mask_dir is not None or args.raw_gt_dir is not None:
        result = _run_singleturn_raw_folder_mode(
            pipeline=pipeline,
            args=args,
            weight_dtype=weight_dtype,
            default_prompt_cache=default_prompt_cache,
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
