#!/usr/bin/env python

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple CORNE singleturn cache directories into one output root.")
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Preprocess output directories to merge. Each must contain manifest.json, metadata.json, cache/, and shared_prompt_embeds.pt.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Merged output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_empty_output(output_dir: Path, overwrite: bool):
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to reuse it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cache").mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    args = parse_args()

    input_dirs = [Path(path) for path in args.input_dirs]
    output_dir = Path(args.output_dir)
    _ensure_empty_output(output_dir, args.overwrite)

    merged_manifest = []
    metadata_list = []
    sample_sizes = set()
    prompt_texts = set()
    prompt_cache_copied = False

    for input_dir in input_dirs:
        manifest_path = input_dir / "manifest.json"
        metadata_path = input_dir / "metadata.json"
        prompt_cache_path = input_dir / "shared_prompt_embeds.pt"
        cache_dir = input_dir / "cache"

        if not manifest_path.is_file() or not metadata_path.is_file() or not prompt_cache_path.is_file() or not cache_dir.is_dir():
            raise FileNotFoundError(f"Incomplete preprocess output: {input_dir}")

        manifest = _load_json(manifest_path)
        metadata = _load_json(metadata_path)
        if not isinstance(manifest, list):
            raise ValueError(f"Manifest must be a list: {manifest_path}")
        if metadata.get("dataset_type") != "corne_object_removal":
            raise ValueError(f"Unexpected dataset_type in {metadata_path}: {metadata.get('dataset_type')}")

        metadata_list.append(metadata)
        sample_sizes.add(tuple(metadata.get("singleturn_sample_size", [])))
        prompt_texts.add(metadata.get("shared_prompt_text"))

        if not prompt_cache_copied:
            shutil.copy2(prompt_cache_path, output_dir / "shared_prompt_embeds.pt")
            prompt_cache_copied = True

        for entry in manifest:
            rel_cache_path = Path(entry["cache_path"])
            src_cache_path = input_dir / rel_cache_path
            dst_cache_path = output_dir / rel_cache_path
            if not src_cache_path.is_file():
                raise FileNotFoundError(f"Missing cache payload referenced by manifest: {src_cache_path}")
            _link_or_copy(src_cache_path, dst_cache_path)
            merged_manifest.append(dict(entry))

    if len(sample_sizes) != 1:
        raise ValueError(f"Mismatched sample sizes across inputs: {sorted(sample_sizes)}")
    if len(prompt_texts) != 1:
        raise ValueError(f"Mismatched shared prompts across inputs: {sorted(prompt_texts)}")

    merged_manifest.sort(key=lambda item: (int(item.get("global_index", 0)), str(item.get("cache_path", ""))))
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(merged_manifest, f, indent=2)

    merged_metadata = {
        "mode": metadata_list[0].get("mode", "singleturn_object_removal_v2"),
        "dataset_type": "corne_object_removal",
        "conditioning_format": metadata_list[0].get("conditioning_format", ""),
        "prefix_frames": metadata_list[0].get("prefix_frames"),
        "total_frames": metadata_list[0].get("total_frames"),
        "pixel_space_source_masking": metadata_list[0].get("pixel_space_source_masking", False),
        "max_samples_with_mask_sam": sum(int(meta.get("max_samples_with_mask_sam", 0)) for meta in metadata_list),
        "max_samples_without_mask_sam": sum(int(meta.get("max_samples_without_mask_sam", 0)) for meta in metadata_list),
        "skip_samples_with_mask_sam": sum(int(meta.get("skip_samples_with_mask_sam", 0)) for meta in metadata_list),
        "skip_samples_without_mask_sam": sum(int(meta.get("skip_samples_without_mask_sam", 0)) for meta in metadata_list),
        "num_samples_with_mask_sam": sum(int(meta.get("num_samples_with_mask_sam", 0)) for meta in metadata_list),
        "num_samples_without_mask_sam": sum(int(meta.get("num_samples_without_mask_sam", 0)) for meta in metadata_list),
        "stopped_early_when_quotas_met": all(bool(meta.get("stopped_early_when_quotas_met", False)) for meta in metadata_list),
        "num_samples_total": len(merged_manifest),
        "singleturn_sample_size": list(sample_sizes.pop()),
        "shared_prompt_text": prompt_texts.pop(),
        "shared_prompt_cache": "shared_prompt_embeds.pt",
        "manifest_path": "manifest.json",
        "merged_from": [str(path) for path in input_dirs],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(merged_metadata, f, indent=2)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest_path": str(output_dir / "manifest.json"),
                "metadata_path": str(output_dir / "metadata.json"),
                "num_samples_total": len(merged_manifest),
                "num_samples_with_mask_sam": merged_metadata["num_samples_with_mask_sam"],
                "num_samples_without_mask_sam": merged_metadata["num_samples_without_mask_sam"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
