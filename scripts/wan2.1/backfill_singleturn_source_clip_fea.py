#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm.auto import tqdm


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from videox_fun.models import AutoencoderKLWan, CLIPModel
from videox_fun.utils.singleturn_utils import (
    normalize_singleturn_sample_size,
    preprocess_singleturn_image,
    preprocess_singleturn_mask,
)


DEFAULT_CACHE_ROOT = (
    "/mnt/cpfs/jiachengliu/dataset/CORNE/cache/"
    "singleturn_object_removal_wan2.1_1.3b_sam_strict_keyframe_cache_v1"
)
DEFAULT_CLIP_CKPT = (
    "/mnt/cpfs/jiachengliu/pretrained_models/Wan-AI/Wan2.1-Fun-14B-InP/"
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
)
DEFAULT_VAE_CKPT = "/mnt/cpfs/jiachengliu/pretrained_models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill source_clip_fea and target_mask_latent into SingleTurn cache .pt files in place."
    )
    parser.add_argument("--cache_root", type=str, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cache_subdir", type=str, default="cache")
    parser.add_argument("--clip_checkpoint", type=str, default=DEFAULT_CLIP_CKPT)
    parser.add_argument("--vae_checkpoint", type=str, default=DEFAULT_VAE_CKPT)
    parser.add_argument("--clip_key", type=str, default="source_clip_fea")
    parser.add_argument("--target_mask_latent_key", type=str, default="target_mask_latent")
    parser.add_argument("--sample_size", type=int, nargs=2, default=(480, 832), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--overwrite", action="store_true", help="Recompute keys even if they already exist.")
    parser.add_argument("--dry_run", action="store_true", help="Only list work; do not load models or modify files.")
    return parser.parse_args()


def load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32" or device.type != "cuda":
        return torch.float32
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def collect_cache_files(cache_root: Path, cache_subdir: str, rank: int, world_size: int):
    cache_dir = cache_root / cache_subdir
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
    files = sorted(cache_dir.glob("*.pt"))
    if world_size <= 0:
        raise ValueError("--world_size must be positive.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"--rank must be in [0, {world_size}), got {rank}.")
    return [path for index, path in enumerate(files) if index % world_size == rank]


def tensor_for_clip(image_path: str, sample_size: tuple[int, int]) -> torch.Tensor:
    normalized = preprocess_singleturn_image(
        image_path,
        sample_size,
        add_batch_dim=False,
        add_frame_dim=True,
    )
    first_frame = normalized[0].permute(1, 2, 0).contiguous()
    clip_pixel = (first_frame * 0.5 + 0.5).mul(255.0).clamp(0, 255)
    clip_image = Image.fromarray(np.uint8(clip_pixel.cpu().numpy()))
    clip_tensor = TF.to_tensor(clip_image).sub_(0.5).div_(0.5)
    return clip_tensor[:, None, :, :].contiguous()


def tensor_for_target_mask_latent(
    target_image_path: str,
    mask_image_path: str,
    sample_size: tuple[int, int],
) -> torch.Tensor:
    target = preprocess_singleturn_image(
        target_image_path,
        sample_size,
        add_batch_dim=False,
        add_frame_dim=True,
    )
    mask = preprocess_singleturn_mask(
        mask_image_path,
        sample_size,
        add_batch_dim=False,
        add_frame_dim=True,
    )
    return target * (1.0 - mask)


def save_payload_in_place(path: Path, payload: dict):
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def flush_batch(
    batch,
    clip_encoder,
    vae,
    sample_size,
    device,
    weight_dtype,
    clip_key: str,
    target_mask_latent_key: str,
    overwrite: bool,
    dry_run: bool,
):
    if not batch:
        return 0, 0

    pending = []
    skipped = 0
    for path in batch:
        payload = load_pt(path)
        needs_clip = overwrite or clip_key not in payload
        needs_target_mask_latent = overwrite or target_mask_latent_key not in payload
        if not needs_clip and not needs_target_mask_latent:
            skipped += 1
            continue
        source_image = payload.get("source_image")
        if not source_image:
            raise KeyError(f"{path} does not contain source_image.")
        if not os.path.exists(source_image):
            raise FileNotFoundError(f"source_image does not exist for {path}: {source_image}")
        target_image = payload.get("bg_image", payload.get("target_image"))
        if not target_image:
            raise KeyError(f"{path} does not contain bg_image or target_image.")
        if not os.path.exists(target_image):
            raise FileNotFoundError(f"target image does not exist for {path}: {target_image}")
        mask_check_image = payload.get("mask_check_image")
        if not mask_check_image:
            raise KeyError(f"{path} does not contain mask_check_image.")
        if not os.path.exists(mask_check_image):
            raise FileNotFoundError(f"mask_check_image does not exist for {path}: {mask_check_image}")
        pending.append((path, payload, source_image, target_image, mask_check_image, needs_clip, needs_target_mask_latent))

    if not pending:
        return 0, skipped
    if dry_run:
        return len(pending), skipped

    with torch.no_grad():
        clip_indices = [index for index, item in enumerate(pending) if item[5]]
        source_clip_fea = None
        if clip_indices:
            videos = [
                tensor_for_clip(pending[index][2], sample_size).to(device=device, dtype=weight_dtype, non_blocking=True)
                for index in clip_indices
            ]
            source_clip_fea = clip_encoder(videos).detach().cpu().to(weight_dtype)

        target_mask_indices = [index for index, item in enumerate(pending) if item[6]]
        target_mask_latents = None
        if target_mask_indices:
            target_mask_pixels = torch.stack(
                [
                    tensor_for_target_mask_latent(pending[index][3], pending[index][4], sample_size)
                    for index in target_mask_indices
                ],
                dim=0,
            )
            target_mask_pixels = target_mask_pixels.permute(0, 2, 1, 3, 4).to(
                device=device,
                dtype=weight_dtype,
                non_blocking=True,
            )
            target_mask_latents = vae.encode(target_mask_pixels)[0].mode().detach().cpu().to(weight_dtype)

    clip_offsets = {pending_index: offset for offset, pending_index in enumerate(clip_indices)}
    target_mask_offsets = {pending_index: offset for offset, pending_index in enumerate(target_mask_indices)}
    for index, (path, payload, *_rest) in enumerate(pending):
        if index in clip_offsets:
            payload[clip_key] = source_clip_fea[clip_offsets[index]]
        if index in target_mask_offsets:
            payload[target_mask_latent_key] = target_mask_latents[target_mask_offsets[index]]
        save_payload_in_place(path, payload)

    return len(pending), skipped


def main():
    args = parse_args()
    sample_size = normalize_singleturn_sample_size(args.sample_size)
    cache_root = Path(args.cache_root)
    files = collect_cache_files(cache_root, args.cache_subdir, args.rank, args.world_size)

    print(
        f"rank {args.rank}/{args.world_size}: {len(files)} cache files under "
        f"{cache_root / args.cache_subdir}"
    )
    if args.dry_run:
        pending = 0
        skipped = 0
        for path in tqdm(files, desc="dry-run"):
            payload = load_pt(path)
            if args.clip_key in payload and args.target_mask_latent_key in payload and not args.overwrite:
                skipped += 1
            else:
                if "source_image" not in payload:
                    raise KeyError(f"{path} does not contain source_image.")
                if "mask_check_image" not in payload:
                    raise KeyError(f"{path} does not contain mask_check_image.")
                if "bg_image" not in payload and "target_image" not in payload:
                    raise KeyError(f"{path} does not contain bg_image or target_image.")
                pending += 1
        print(f"dry-run done: pending={pending}, skipped_existing={skipped}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = get_dtype(args.dtype, device)
    clip_encoder = CLIPModel.from_pretrained(args.clip_checkpoint).eval().to(device=device, dtype=weight_dtype)
    clip_encoder.requires_grad_(False)
    vae = AutoencoderKLWan.from_pretrained(
        args.vae_checkpoint,
        additional_kwargs={
            "temporal_compression_ratio": 4,
            "spatial_compression_ratio": 8,
        },
    ).eval().to(device=device, dtype=weight_dtype)
    vae.requires_grad_(False)

    patched = 0
    skipped = 0
    progress = tqdm(range(0, len(files), args.batch_size), desc=f"rank {args.rank}")
    for start in progress:
        batch = files[start : start + args.batch_size]
        num_patched, num_skipped = flush_batch(
            batch,
            clip_encoder,
            vae,
            sample_size,
            device,
            weight_dtype,
            args.clip_key,
            args.target_mask_latent_key,
            overwrite=args.overwrite,
            dry_run=False,
        )
        patched += num_patched
        skipped += num_skipped
        progress.set_postfix(patched=patched, skipped=skipped)

    print(f"done: patched={patched}, skipped_existing={skipped}")


if __name__ == "__main__":
    main()
