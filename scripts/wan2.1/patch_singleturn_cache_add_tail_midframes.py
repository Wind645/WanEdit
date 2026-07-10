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


OLD_TOTAL_FRAMES = 8
NEW_TOTAL_FRAMES = 11
PATCHED_MODE = "singleturn_object_removal_v3_tail_interp11"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Patch cached SingleTurn object-removal samples from the original 8-frame tail layout "
            "to an 11-frame version by inserting three midpoint frames between F5-F6, F6-F7, and F7-F8."
        )
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory containing the original .pt cache files, usually <cache_root>/cache.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional manifest.json for resolving relative cache paths and copying metadata.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional root used to resolve relative cache paths from the manifest.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write patched cache/manifest/metadata. Required unless --inplace is used.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the existing cache files and manifest/metadata in place.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting patched files in --output_dir or existing files when --inplace is used.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of cache files to patch.",
    )
    return parser.parse_args()


def _resolve_paths(args):
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path is not None else None

    if manifest_path is not None:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, list):
            raise ValueError(f"Expected manifest to be a list, got {type(manifest)}")
        source_root = Path(args.data_root).resolve() if args.data_root is not None else manifest_path.parent.resolve()
        cache_paths = []
        for entry in manifest:
            cache_path = Path(entry["cache_path"])
            if not cache_path.is_absolute():
                cache_path = source_root / cache_path
            cache_paths.append(cache_path.resolve())
    else:
        if args.cache_dir is None:
            raise ValueError("Provide either --cache_dir or --manifest_path.")
        source_root = Path(args.cache_dir).resolve().parent
        manifest = None
        cache_paths = sorted(Path(args.cache_dir).resolve().glob("*.pt"))

    if not cache_paths:
        raise ValueError("No cache files found to patch.")
    if args.max_samples is not None:
        cache_paths = cache_paths[: args.max_samples]

    if args.inplace:
        output_root = source_root
    else:
        if args.output_dir is None:
            raise ValueError("Provide --output_dir unless using --inplace.")
        output_root = Path(args.output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    cache_output_dir = output_root / "cache"
    cache_output_dir.mkdir(parents=True, exist_ok=True)

    return manifest_path, manifest, source_root, cache_paths, output_root, cache_output_dir


def _insert_tail_midframes(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.ndim < 2:
        raise ValueError(f"Expected at least 2 dims with frame axis in dim=1, got {tuple(sequence.shape)}")
    if sequence.shape[1] != OLD_TOTAL_FRAMES:
        raise ValueError(f"Expected {OLD_TOTAL_FRAMES} frames before patching, got {sequence.shape[1]}")

    f5 = sequence[:, 4:5]
    f6 = sequence[:, 5:6]
    f7 = sequence[:, 6:7]
    f8 = sequence[:, 7:8]
    mid_56 = 0.5 * (f5 + f6)
    mid_67 = 0.5 * (f6 + f7)
    mid_78 = 0.5 * (f7 + f8)
    return torch.cat(
        [
            sequence[:, :5],
            mid_56,
            f6,
            mid_67,
            f7,
            mid_78,
            f8,
        ],
        dim=1,
    )


def _patch_payload(payload: dict) -> dict:
    if payload.get("mode") != "singleturn_object_removal_v2":
        raise ValueError(f"Unsupported mode {payload.get('mode')!r}; expected singleturn_object_removal_v2")
    if "full_latents" not in payload:
        raise ValueError("Missing full_latents in cache payload.")

    full_latents = payload["full_latents"]
    if not torch.is_tensor(full_latents) or full_latents.ndim != 4:
        raise ValueError(f"Expected full_latents to be a 4D tensor, got {type(full_latents)} / {getattr(full_latents, 'shape', None)}")
    if full_latents.shape[1] != OLD_TOTAL_FRAMES:
        raise ValueError(f"Expected full_latents to have {OLD_TOTAL_FRAMES} frames, got {full_latents.shape[1]}")

    patched = dict(payload)
    patched["full_latents"] = _insert_tail_midframes(full_latents)
    if "edge_weight_map" in payload:
        edge_weight_map = payload["edge_weight_map"]
        if not torch.is_tensor(edge_weight_map) or edge_weight_map.ndim != 4:
            raise ValueError(
                f"Expected edge_weight_map to be a 4D tensor with shape (1, T, H, W), got {type(edge_weight_map)} / {getattr(edge_weight_map, 'shape', None)}"
            )
        if edge_weight_map.shape[1] != OLD_TOTAL_FRAMES:
            raise ValueError(f"Expected edge_weight_map to have {OLD_TOTAL_FRAMES} frames, got {edge_weight_map.shape[1]}")
        patched["edge_weight_map"] = _insert_tail_midframes(edge_weight_map)

    patched["mode"] = PATCHED_MODE
    patched["total_frames"] = NEW_TOTAL_FRAMES
    patched["tail_patch"] = {
        "source_mode": "singleturn_object_removal_v2",
        "target_mode": PATCHED_MODE,
        "source_total_frames": OLD_TOTAL_FRAMES,
        "target_total_frames": NEW_TOTAL_FRAMES,
        "strategy": "insert_midpoints_between_F5_F6_F6_F7_F7_F8",
    }
    return patched


def _copy_shared_prompt_cache_if_needed(source_root: Path, output_root: Path, metadata: dict | None) -> None:
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


def main():
    args = parse_args()
    manifest_path, manifest, source_root, cache_paths, output_root, cache_output_dir = _resolve_paths(args)
    manifest_entry_map = None
    if manifest is not None:
        manifest_entry_map = {Path(entry["cache_path"]).name: entry for entry in manifest}

    metadata_path = source_root / "metadata.json"
    source_metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            source_metadata = json.load(f)

    _copy_shared_prompt_cache_if_needed(source_root, output_root, source_metadata)

    manifest_entries = []
    processed = 0
    skipped = {"already_exists": 0}

    progress = tqdm(cache_paths, total=len(cache_paths), desc="Patching SingleTurn cache", dynamic_ncols=True)
    for cache_path in progress:
        output_cache_path = cache_output_dir / cache_path.name
        manifest_entry = manifest_entry_map.get(cache_path.name) if manifest_entry_map is not None else None
        if output_cache_path.exists() and not args.overwrite:
            skipped["already_exists"] += 1
            if manifest_entry is not None:
                patched_entry = dict(manifest_entry)
                patched_entry["cache_path"] = str(output_cache_path.relative_to(output_root))
                patched_entry["mode"] = PATCHED_MODE
                patched_entry["tail_patch"] = {
                    "source_mode": "singleturn_object_removal_v2",
                    "target_mode": PATCHED_MODE,
                    "source_total_frames": OLD_TOTAL_FRAMES,
                    "target_total_frames": NEW_TOTAL_FRAMES,
                    "strategy": "insert_midpoints_between_F5_F6_F6_F7_F7_F8",
                }
                manifest_entries.append(patched_entry)
            else:
                manifest_entries.append(
                    {
                        "cache_path": str(output_cache_path.relative_to(output_root)),
                        "mode": PATCHED_MODE,
                    }
                )
            progress.set_postfix(processed=processed, skipped=skipped["already_exists"])
            continue

        payload = load_singleturn_cache_payload(str(cache_path))
        patched_payload = _patch_payload(payload)
        torch.save(patched_payload, output_cache_path)

        if manifest_entry is not None:
            patched_entry = dict(manifest_entry)
            patched_entry["cache_path"] = str(output_cache_path.relative_to(output_root))
            patched_entry["mode"] = PATCHED_MODE
            patched_entry["tail_patch"] = patched_payload["tail_patch"]
            manifest_entries.append(patched_entry)
        else:
            manifest_entries.append(
                {
                    "cache_path": str(output_cache_path.relative_to(output_root)),
                    "mode": PATCHED_MODE,
                    "tail_patch": patched_payload["tail_patch"],
                }
            )

        processed += 1
        progress.set_postfix(processed=processed, skipped=skipped["already_exists"])

    progress.close()

    manifest_out_path = output_root / "manifest.json"
    with open(manifest_out_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    metadata = dict(source_metadata or {})
    metadata.update(
        {
            "mode": PATCHED_MODE,
            "dataset_type": metadata.get("dataset_type", "corne_object_removal"),
            "conditioning_format": "11-frame 2-prefix mask-latent + clean-source-latent + densified tail interpolation",
            "prefix_frames": metadata.get("prefix_frames", 2),
            "total_frames": NEW_TOTAL_FRAMES,
            "source_total_frames": OLD_TOTAL_FRAMES,
            "tail_patch": {
                "source_mode": "singleturn_object_removal_v2",
                "target_mode": PATCHED_MODE,
                "source_total_frames": OLD_TOTAL_FRAMES,
                "target_total_frames": NEW_TOTAL_FRAMES,
                "strategy": "insert_midpoints_between_F5_F6_F6_F7_F7_F8",
            },
            "num_samples_total": len(manifest_entries),
            "manifest_path": str(manifest_out_path.relative_to(output_root)),
        }
    )
    metadata_out_path = output_root / "metadata.json"
    with open(metadata_out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "processed": processed,
        "skipped": skipped,
        "num_cache_files": len(cache_paths),
        "manifest_path": str(manifest_out_path),
        "metadata_path": str(metadata_out_path),
        "target_mode": PATCHED_MODE,
        "target_total_frames": NEW_TOTAL_FRAMES,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
