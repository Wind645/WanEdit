#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

import torch
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
from videox_fun.utils.singleturn_utils import (
    build_singleturn_refinement_weight_map,
    normalize_singleturn_sample_size,
    preprocess_singleturn_mask,
    resize_singleturn_mask_to_latent_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill refinement_loss_weight_map into existing SingleTurn refine cache .pt files."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory containing refine cache .pt files, usually <refine_cache_root>/cache.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional refine cache manifest.json used to resolve relative cache paths.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional root used to resolve relative cache paths from manifest entries.",
    )
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Recompute and overwrite refinement_loss_weight_map even if it already exists.",
    )
    parser.add_argument(
        "--force_recompute_all",
        action="store_true",
        help="Alias of --overwrite_existing. Recompute weight maps for every refine cache sample.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Stored tensor dtype for the generated weight map.",
    )
    return parser.parse_args()


def get_storage_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16


def iter_cache_paths(args) -> list[Path]:
    cache_paths: list[Path] = []
    if args.manifest_path is not None:
        manifest_path = Path(args.manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, list):
            raise ValueError(f"Expected manifest to be a list, got {type(manifest)}")
        base_root = Path(args.data_root) if args.data_root is not None else manifest_path.parent
        for entry in manifest:
            cache_path = Path(entry["cache_path"])
            if not cache_path.is_absolute():
                cache_path = base_root / cache_path
            cache_paths.append(cache_path.resolve())
        return cache_paths

    if args.cache_dir is None:
        raise ValueError("Provide either --cache_dir or --manifest_path.")

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")
    return sorted(cache_dir.glob("*.pt"))


def main():
    args = parse_args()
    if args.force_recompute_all:
        args.overwrite_existing = True
    storage_dtype = get_storage_dtype(args.dtype)
    cache_paths = iter_cache_paths(args)
    if not cache_paths:
        raise ValueError("No cache files found to patch.")

    processed = 0
    skipped = {
        "already_present": 0,
        "unsupported_mode": 0,
        "missing_mask_path": 0,
        "missing_sample_size": 0,
        "bad_input_latents": 0,
    }

    progress = tqdm(cache_paths, total=len(cache_paths), desc="Backfilling refine weight maps", dynamic_ncols=True)
    for cache_path in progress:
        payload = load_singleturn_cache_payload(str(cache_path))
        if payload.get("mode") != "singleturn_object_removal_refine_v1":
            skipped["unsupported_mode"] += 1
            progress.set_postfix(processed=processed, already_present=skipped["already_present"], skipped=sum(skipped.values()) - skipped["already_present"])
            continue
        if (not args.overwrite_existing) and ("refinement_loss_weight_map" in payload):
            skipped["already_present"] += 1
            progress.set_postfix(processed=processed, already_present=skipped["already_present"], skipped=sum(skipped.values()) - skipped["already_present"])
            continue

        mask_check_image = payload.get("mask_check_image")
        if not mask_check_image or not os.path.exists(mask_check_image):
            skipped["missing_mask_path"] += 1
            progress.set_postfix(processed=processed, already_present=skipped["already_present"], skipped=sum(skipped.values()) - skipped["already_present"])
            continue

        sample_size_value = payload.get("singleturn_sample_size")
        if sample_size_value is None:
            skipped["missing_sample_size"] += 1
            progress.set_postfix(processed=processed, already_present=skipped["already_present"], skipped=sum(skipped.values()) - skipped["already_present"])
            continue
        sample_size = normalize_singleturn_sample_size(sample_size_value)

        input_latents = payload.get("input_latents")
        if not torch.is_tensor(input_latents) or input_latents.ndim != 4:
            skipped["bad_input_latents"] += 1
            progress.set_postfix(processed=processed, already_present=skipped["already_present"], skipped=sum(skipped.values()) - skipped["already_present"])
            continue

        latent_frame = input_latents[:, :1].unsqueeze(0).float()
        mask_check_tensor = preprocess_singleturn_mask(
            mask_check_image,
            sample_size,
            add_batch_dim=True,
            add_frame_dim=True,
        ).float()
        mask_check_latent = resize_singleturn_mask_to_latent_grid(mask_check_tensor, latent_frame)
        refinement_loss_weight_map = build_singleturn_refinement_weight_map(mask_check_latent)[0].to(storage_dtype)

        payload["refinement_loss_weight_map"] = refinement_loss_weight_map.cpu()
        torch.save(payload, cache_path)
        processed += 1
        progress.set_postfix(processed=processed, already_present=skipped["already_present"], skipped=sum(skipped.values()) - skipped["already_present"])

    summary = {
        "processed": processed,
        "skipped": skipped,
        "num_cache_files": len(cache_paths),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
