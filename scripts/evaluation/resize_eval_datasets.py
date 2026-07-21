#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_OUTPUT_ROOT = Path("/home/data/nas_hdd/eval_datasets")
REMOVALBENCH_ROOT = Path("/home/data/nas_hdd/eval_datasets/RemovalBench")
OMNIPAINT_PART1_ROOT = Path("/home/data/nas_hdd/eval_datasets/OmniPaint-Bench/extracted/OmniPaint-Bench-Part1")
RORD_VAL_ROOT = Path("/home/data/nas_hdd/eval_datasets/RORD/extracted/RORD/val")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MASK_DIR_NAMES = {"mask", "masks"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize evaluation datasets under /home/data/nas_hdd/eval_datasets.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--square-size", type=int, default=480)
    parser.add_argument("--rord-sample-count", type=int, default=100)
    parser.add_argument("--rord-width", type=int, default=832)
    parser.add_argument("--rord-height", type=int, default=468)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def is_mask_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if parts & MASK_DIR_NAMES:
        return True
    name = path.stem.lower()
    return name.startswith("mask") or "_mask" in name


def iter_image_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def resize_one_image(src_path: Path, dst_path: Path, size: tuple[int, int], *, overwrite: bool) -> bool:
    if dst_path.exists() and not overwrite:
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as image:
        image = ImageOps.exif_transpose(image)
        resample = Image.Resampling.NEAREST if is_mask_path(src_path) else Image.Resampling.LANCZOS
        resized = image.resize(size, resample=resample)
        save_kwargs = {}
        if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update({"quality": 95, "subsampling": 0})
        resized.save(dst_path, **save_kwargs)
    return True


def process_square_root(
    input_root: Path,
    output_root: Path,
    output_name: str,
    square_size: int,
    *,
    overwrite: bool,
) -> tuple[int, int]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    output_dir = output_root / output_name
    resized_count = 0
    skipped_count = 0

    for src_path in iter_image_files(input_root):
        dst_path = output_dir / src_path.relative_to(input_root)
        if resize_one_image(src_path, dst_path, (square_size, square_size), overwrite=overwrite):
            resized_count += 1
        else:
            skipped_count += 1

    return resized_count, skipped_count


def _collect_rord_triplets(input_root: Path) -> list[dict[str, Path]]:
    img_root = input_root / "img"
    gt_root = input_root / "gt"
    mask_root = input_root / "mask"
    for directory in (img_root, gt_root, mask_root):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing RORD val directory: {directory}")

    triplets: dict[Path, dict[str, Path]] = {}
    for kind, base in (("img", img_root), ("gt", gt_root), ("mask", mask_root)):
        for src_path in iter_image_files(base):
            rel_key = src_path.relative_to(base).with_suffix("")
            if kind == "mask" and rel_key.name.endswith("_M"):
                rel_key = rel_key.with_name(rel_key.name[:-2])
            key = rel_key
            triplets.setdefault(key, {})[kind] = src_path

    return [
        {"key": key, "img": files["img"], "gt": files["gt"], "mask": files["mask"]}
        for key, files in sorted(triplets.items(), key=lambda item: str(item[0]))
        if {"img", "gt", "mask"} <= files.keys()
    ]


def process_rord_val(
    input_root: Path,
    output_root: Path,
    *,
    sample_count: int,
    target_size: tuple[int, int],
    seed: int,
    overwrite: bool,
) -> tuple[int, int, Path, int]:
    triplets = _collect_rord_triplets(input_root)
    if not triplets:
        raise ValueError(f"No valid img/gt/mask triplets found under {input_root}")

    rng = random.Random(seed)
    if sample_count < len(triplets):
        selected = rng.sample(triplets, sample_count)
    else:
        selected = triplets

    output_dir = output_root / f"RORD_val_{len(selected)}_{target_size[0]}x{target_size[1]}"
    resized_count = 0
    skipped_count = 0
    selected_keys: list[str] = []

    for triplet in selected:
        selected_keys.append(str(triplet["key"]))
        for kind in ("img", "gt", "mask"):
            src_path = triplet[kind]
            dst_path = output_dir / src_path.relative_to(input_root)
            if resize_one_image(src_path, dst_path, target_size, overwrite=overwrite):
                resized_count += 1
            else:
                skipped_count += 1

    manifest_path = output_dir / "selected_triplets.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "input_root": str(input_root),
                "sample_count": len(selected),
                "seed": seed,
                "target_size": list(target_size),
                "selected_triplets": selected_keys,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return resized_count, skipped_count, output_dir, len(selected)


def main() -> None:
    args = parse_args()

    square_tasks = [
        (REMOVALBENCH_ROOT, "RemovalBench_480"),
        (OMNIPAINT_PART1_ROOT, "OmniPaint-Bench-Part1_480"),
    ]

    total_resized = 0
    total_skipped = 0

    for input_root, output_name in square_tasks:
        resized, skipped = process_square_root(
            input_root,
            args.output_root,
            output_name,
            args.square_size,
            overwrite=args.overwrite,
        )
        total_resized += resized
        total_skipped += skipped
        print(f"{input_root} -> {args.output_root / output_name}: resized={resized}, skipped={skipped}")

    rord_resized, rord_skipped, rord_output_dir, rord_selected = process_rord_val(
        RORD_VAL_ROOT,
        args.output_root,
        sample_count=args.rord_sample_count,
        target_size=(args.rord_width, args.rord_height),
        seed=args.seed,
        overwrite=args.overwrite,
    )
    total_resized += rord_resized
    total_skipped += rord_skipped
    print(f"{RORD_VAL_ROOT} -> {rord_output_dir}: resized={rord_resized}, skipped={rord_skipped}, sampled={rord_selected}")
    print(f"done: resized={total_resized}, skipped={total_skipped}")


if __name__ == "__main__":
    main()
