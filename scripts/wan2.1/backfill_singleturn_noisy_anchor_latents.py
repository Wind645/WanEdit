#!/usr/bin/env python

import argparse
import json
import os
import shutil
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
    SINGLETURN_OBJECT_REMOVAL_CACHE_MODE,
    build_singleturn_noisy_anchor_latents,
    normalize_singleturn_sample_size,
    preprocess_singleturn_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild noisy_anchor_latent for cached SingleTurn samples by resizing the loose mask to latent space "
            "and replacing the masked latent region with Gaussian noise."
        )
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory containing cached .pt files, usually <cache_root>/cache.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional manifest.json used to resolve relative cache paths and rewrite manifest entries.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional root used to resolve relative cache paths from manifest entries.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output root. If omitted, patch files in place.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output files when --output_dir is set.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Base seed added to each sample global_index to deterministically rebuild noisy_anchor_latent.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of cache samples to patch.",
    )
    return parser.parse_args()


def _parse_global_index(value: object, fallback_name: str) -> int:
    if value is not None:
        return int(value)
    prefix = str(fallback_name).split("_", 1)[0]
    return int(prefix)


def _resolve_paths(args):
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path is not None else None
    manifest = None
    source_root = None
    cache_paths = []
    manifest_entry_map = None

    if manifest_path is not None:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, list):
            raise ValueError(f"Expected manifest to be a list, got {type(manifest)}")
        source_root = Path(args.data_root).resolve() if args.data_root is not None else manifest_path.parent.resolve()
        for entry in manifest:
            cache_path = Path(entry["cache_path"])
            if not cache_path.is_absolute():
                cache_path = source_root / cache_path
            cache_paths.append(cache_path.resolve())
        manifest_entry_map = {Path(entry["cache_path"]).name: entry for entry in manifest}
    else:
        if args.cache_dir is None:
            raise ValueError("Provide either --cache_dir or --manifest_path.")
        cache_dir = Path(args.cache_dir).resolve()
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")
        source_root = cache_dir.parent
        cache_paths = sorted(cache_dir.glob("*.pt"))

    if not cache_paths:
        raise ValueError("No cache files found to patch.")
    if args.max_samples is not None:
        cache_paths = cache_paths[: args.max_samples]

    if args.output_dir is None:
        output_root = source_root
    else:
        output_root = Path(args.output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    cache_output_dir = output_root / "cache"
    cache_output_dir.mkdir(parents=True, exist_ok=True)
    return manifest_path, manifest, manifest_entry_map, source_root, cache_paths, output_root, cache_output_dir


def _copy_auxiliary_files_if_needed(source_root: Path, output_root: Path, metadata: dict | None) -> None:
    if source_root == output_root:
        return
    if metadata is None:
        return
    shared_prompt_cache = metadata.get("shared_prompt_cache")
    if not shared_prompt_cache:
        return
    src = source_root / shared_prompt_cache
    dst = output_root / shared_prompt_cache
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _build_generator(base_seed: int, global_index: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(base_seed) + int(global_index))
    return generator


def main():
    args = parse_args()
    (
        manifest_path,
        manifest,
        manifest_entry_map,
        source_root,
        cache_paths,
        output_root,
        cache_output_dir,
    ) = _resolve_paths(args)

    metadata_path = source_root / "metadata.json"
    source_metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            source_metadata = json.load(f)
    _copy_auxiliary_files_if_needed(source_root, output_root, source_metadata)

    processed = 0
    skipped = {
        "unsupported_mode": 0,
        "missing_mask_path": 0,
        "missing_sample_size": 0,
        "missing_source_latent": 0,
        "already_exists": 0,
    }
    manifest_entries = []

    progress = tqdm(cache_paths, total=len(cache_paths), desc="Backfilling noisy anchor latents", dynamic_ncols=True)
    for cache_path in progress:
        payload = load_singleturn_cache_payload(str(cache_path))
        manifest_entry = manifest_entry_map.get(cache_path.name) if manifest_entry_map is not None else None

        if payload.get("mode") != SINGLETURN_OBJECT_REMOVAL_CACHE_MODE:
            skipped["unsupported_mode"] += 1
            progress.set_postfix(processed=processed, skipped=sum(skipped.values()))
            continue

        mask_check_image = payload.get("mask_check_image", "")
        if not mask_check_image or not os.path.exists(mask_check_image):
            skipped["missing_mask_path"] += 1
            progress.set_postfix(processed=processed, skipped=sum(skipped.values()))
            continue

        sample_size_value = payload.get("singleturn_sample_size")
        if sample_size_value is None:
            skipped["missing_sample_size"] += 1
            progress.set_postfix(processed=processed, skipped=sum(skipped.values()))
            continue
        sample_size = normalize_singleturn_sample_size(sample_size_value)

        source_frame_latent = payload.get("source_frame_latent")
        if not torch.is_tensor(source_frame_latent) or source_frame_latent.ndim != 4:
            skipped["missing_source_latent"] += 1
            progress.set_postfix(processed=processed, skipped=sum(skipped.values()))
            continue

        global_index = _parse_global_index(
            payload.get("global_index", manifest_entry.get("global_index") if manifest_entry is not None else None),
            cache_path.name,
        )
        generator = _build_generator(args.base_seed, global_index)
        mask_check_tensor = preprocess_singleturn_mask(
            mask_check_image,
            sample_size,
            add_batch_dim=True,
            add_frame_dim=True,
        ).float()
        rebuilt_noisy_anchor = build_singleturn_noisy_anchor_latents(
            source_frame_latent.unsqueeze(0).float(),
            mask_check_tensor,
            generator=generator,
        )[0].to(dtype=source_frame_latent.dtype)

        updated_payload = dict(payload)
        updated_payload["noisy_anchor_latent"] = rebuilt_noisy_anchor.cpu()
        updated_payload["noisy_anchor_build_mode"] = "latent_space_mask_replace_gaussian_noise"
        updated_payload["noisy_anchor_seed"] = int(args.base_seed) + int(global_index)

        output_cache_path = cache_output_dir / cache_path.name
        if output_root != source_root and output_cache_path.exists() and not args.overwrite:
            skipped["already_exists"] += 1
            progress.set_postfix(processed=processed, skipped=sum(skipped.values()))
            continue

        torch.save(updated_payload, output_cache_path if output_root != source_root else cache_path)
        processed += 1

        if manifest_entry is not None:
            rewritten_entry = dict(manifest_entry)
        else:
            rewritten_entry = {
                "cache_path": str((output_cache_path if output_root != source_root else cache_path).relative_to(output_root)),
                "mode": payload.get("mode", ""),
            }
        target_cache_path = output_cache_path if output_root != source_root else cache_path
        rewritten_entry["cache_path"] = str(target_cache_path.relative_to(output_root))
        manifest_entries.append(rewritten_entry)
        progress.set_postfix(processed=processed, skipped=sum(skipped.values()))

    progress.close()

    if manifest is not None:
        manifest_out_path = output_root / "manifest.json"
        with open(manifest_out_path, "w", encoding="utf-8") as f:
            json.dump(manifest_entries, f, indent=2)
    else:
        manifest_out_path = None

    metadata = dict(source_metadata or {})
    metadata.update(
        {
            "noisy_anchor_build_mode": "latent_space_mask_replace_gaussian_noise",
            "noisy_anchor_base_seed": int(args.base_seed),
            "noisy_anchor_rebuilt_sample_count": processed,
        }
    )
    if manifest_out_path is not None:
        metadata["manifest_path"] = str(manifest_out_path.relative_to(output_root))
    metadata_out_path = output_root / "metadata.json"
    with open(metadata_out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "processed": processed,
        "skipped": skipped,
        "num_cache_files": len(cache_paths),
        "manifest_path": str(manifest_out_path) if manifest_out_path is not None else "",
        "metadata_path": str(metadata_out_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
