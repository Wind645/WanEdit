#!/usr/bin/env python

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageSequence
from tqdm.auto import tqdm

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.data.singleturn_dataset import load_singleturn_cache_payload
from videox_fun.models import AutoencoderKLWan
from videox_fun.utils.singleturn_utils import (
    CORNE_SINGLETURN_PROMPT,
    SINGLETURN_TOTAL_FRAMES,
    build_singleturn_refinement_target_latents,
    normalize_singleturn_sample_size,
    preprocess_singleturn_image,
)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Build SingleTurn refinement caches from completed coarse inference outputs.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Base Wan model path.")
    parser.add_argument("--coarse_output_dir", type=str, required=True, help="Root directory containing completed Stage-1 inference sample folders.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write refinement cache tensors and manifest.")
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Wan config path.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Cache tensor dtype.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing refinement cache files.")
    return parser.parse_args()


def get_weight_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16


def _resolve_saved_output_path(meta_dir: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    meta_relative = meta_dir / candidate
    if meta_relative.exists():
        return meta_relative.resolve()
    basename_relative = meta_dir / path.name
    if basename_relative.exists():
        return basename_relative.resolve()
    return path


def _iter_completed_meta_files(coarse_output_dir: Path):
    for meta_path in sorted(coarse_output_dir.rglob("singleturn_meta.json")):
        yield meta_path


def _load_gif_frames(gif_path: Path) -> list[Image.Image]:
    with Image.open(gif_path) as image:
        return [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(image)]


def _encode_full_sequence_latents(
    *,
    vae,
    frames: list[Image.Image],
    sample_size: tuple[int, int],
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    frame_latents = []
    with torch.no_grad():
        for frame in frames:
            frame_tensor = preprocess_singleturn_image(
                frame,
                sample_size,
                add_batch_dim=True,
                add_frame_dim=True,
            ).to(device=device, dtype=weight_dtype)
            frame_latent = vae.encode(frame_tensor.permute(0, 2, 1, 3, 4))[0].mode()
            frame_latents.append(frame_latent)
    return torch.cat(frame_latents, dim=2)


def _resolve_shared_prompt_source(cache_path: Path, payload: dict) -> Path | None:
    shared_prompt_cache = payload.get("shared_prompt_cache")
    if not shared_prompt_cache:
        return None
    if os.path.isabs(shared_prompt_cache):
        return Path(shared_prompt_cache)
    cache_root = cache_path.parent.parent
    return (cache_root / shared_prompt_cache).resolve()


def _copy_shared_prompt_cache_if_needed(output_dir: Path, prompt_cache_source: Path | None) -> str | None:
    if prompt_cache_source is None:
        return None
    destination = output_dir / prompt_cache_source.name
    if not destination.exists():
        shutil.copy2(prompt_cache_source, destination)
    return destination.name


def _build_manifest_entry(cache_path: Path, output_dir: Path, payload: dict) -> dict:
    return {
        "cache_path": str(cache_path.relative_to(output_dir)),
        "mode": "singleturn_object_removal_refine_v1",
        "coarse_output_dir": payload["coarse_output_dir"],
        "coarse_meta_path": payload["coarse_meta_path"],
        "source_cache_path": payload["source_cache_path"],
        "source_image": payload["source_image"],
        "bg_image": payload["bg_image"],
        "mask_check_image": payload["mask_check_image"],
        "mask_frame_image": payload["mask_frame_image"],
        "mask_sam_image": payload.get("mask_sam_image", ""),
        "used_mask_sam": bool(payload["used_mask_sam"]),
        "singleturn_sample_size": payload["singleturn_sample_size"],
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    metadata_path = output_dir / "metadata.json"

    coarse_output_dir = Path(args.coarse_output_dir)
    if not coarse_output_dir.is_dir():
        raise FileNotFoundError(f"Coarse output directory does not exist: {coarse_output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = get_weight_dtype(args.dtype, device)
    config = OmegaConf.load(args.config_path)
    vae = AutoencoderKLWan.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
            "vae",
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).eval().to(device, dtype=weight_dtype)

    manifest = []
    processed = 0
    skipped = {
        "missing_cache_path": 0,
        "missing_outputs": 0,
        "missing_cache_payload": 0,
        "incomplete_frames": 0,
        "unsupported_mode": 0,
        "bad_sample_size": 0,
    }

    for meta_path in tqdm(_iter_completed_meta_files(coarse_output_dir), desc="Building refinement cache"):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cache_path_value = meta.get("cache_path")
        if not cache_path_value:
            skipped["missing_cache_path"] += 1
            continue

        outputs = meta.get("outputs") or meta.get("coarse_outputs") or {}
        full_gif_value = outputs.get("full_gif")
        if not full_gif_value:
            skipped["missing_outputs"] += 1
            continue

        full_gif_path = _resolve_saved_output_path(meta_path.parent, full_gif_value)
        if not full_gif_path.is_file():
            skipped["missing_outputs"] += 1
            continue

        cache_path = Path(cache_path_value)
        if not cache_path.is_file():
            skipped["missing_cache_payload"] += 1
            continue

        source_payload = load_singleturn_cache_payload(str(cache_path))
        if source_payload.get("mode") != "singleturn_object_removal_v2":
            skipped["unsupported_mode"] += 1
            continue

        try:
            sample_size = normalize_singleturn_sample_size(source_payload["singleturn_sample_size"])
        except Exception:
            skipped["bad_sample_size"] += 1
            continue

        frames = _load_gif_frames(full_gif_path)
        if len(frames) != SINGLETURN_TOTAL_FRAMES:
            skipped["incomplete_frames"] += 1
            continue

        coarse_latents = _encode_full_sequence_latents(
            vae=vae,
            frames=frames,
            sample_size=sample_size,
            device=device,
            weight_dtype=weight_dtype,
        )
        gt_last_latent = source_payload["full_latents"][:, -1:].unsqueeze(0).to(device=device, dtype=weight_dtype)
        target_latents = build_singleturn_refinement_target_latents(coarse_latents, gt_last_latent)

        prompt_cache_source = _resolve_shared_prompt_source(cache_path, source_payload)
        shared_prompt_cache = _copy_shared_prompt_cache_if_needed(output_dir, prompt_cache_source)

        cache_name = f"{meta_path.parent.name}.pt"
        refine_cache_path = cache_dir / cache_name
        payload = {
            "mode": "singleturn_object_removal_refine_v1",
            "dataset_type": "corne_object_removal_refine",
            "input_latents": coarse_latents[0].detach().cpu().to(weight_dtype),
            "target_latents": target_latents[0].detach().cpu().to(weight_dtype),
            "coarse_output_dir": str(meta_path.parent.resolve()),
            "coarse_meta_path": str(meta_path.resolve()),
            "coarse_full_gif": str(full_gif_path.resolve()),
            "source_cache_path": str(cache_path.resolve()),
            "source_image": source_payload.get("source_image", meta.get("source_image", "")),
            "bg_image": source_payload.get("bg_image", meta.get("bg_image", "")),
            "mask_check_image": source_payload.get("mask_check_image", meta.get("mask_check_image", "")),
            "mask_frame_image": source_payload.get("mask_frame_image", meta.get("mask_frame_image", "")),
            "mask_sam_image": source_payload.get("mask_sam_image", meta.get("mask_sam_image", "")),
            "used_mask_sam": bool(source_payload.get("used_mask_sam", meta.get("used_mask_sam", False))),
            "singleturn_sample_size": list(sample_size),
            "text": source_payload.get("text", CORNE_SINGLETURN_PROMPT),
            "formatted_text": source_payload.get("formatted_text", source_payload.get("text", CORNE_SINGLETURN_PROMPT)),
        }
        if shared_prompt_cache is not None:
            payload["shared_prompt_cache"] = shared_prompt_cache
        else:
            for key in ("prompt_embeds", "prompt_seq_len"):
                if key in source_payload:
                    payload[key] = source_payload[key]

        if refine_cache_path.exists() and not args.overwrite:
            manifest.append(_build_manifest_entry(refine_cache_path, output_dir, payload))
            processed += 1
            continue

        torch.save(payload, refine_cache_path)
        manifest.append(_build_manifest_entry(refine_cache_path, output_dir, payload))
        processed += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    metadata = {
        "mode": "singleturn_object_removal_refine_v1",
        "coarse_output_dir": str(coarse_output_dir.resolve()),
        "processed": processed,
        "skipped": skipped,
        "manifest_path": str(manifest_path.resolve()),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
