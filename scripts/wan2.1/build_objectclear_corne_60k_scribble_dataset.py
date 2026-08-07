#!/usr/bin/env python

import argparse
import importlib.util
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


DEFAULT_OBJECTCLEAR_ROOT = Path("/mnt/cpfs/jiachengliu/dataset/ObjectClear/extracted/train/captured")
DEFAULT_CORNE_ROOT = Path("/mnt/cpfs/jiachengliu/dataset/CORNE")
DEFAULT_OUTPUT_ROOT = Path("/mnt/cpfs/jiachengliu/dataset/ObjectClear_CORNE_60k_scribble_v1")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _load_simulate_user_scribble():
    script_path = Path(__file__).with_name("simulate_user_scribble.py")
    spec = importlib.util.spec_from_file_location("simulate_user_scribble_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load simulate_user_scribble from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simulate_user_scribble


simulate_user_scribble = _load_simulate_user_scribble()


@dataclass(frozen=True)
class Sample:
    source_kind: str
    stem: str
    input_path: Path
    gt_path: Path
    mask_rem_path: Path
    exact_mask_path: Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a 60k mixed ObjectClear/CORNE dataset with scribble masks.")
    parser.add_argument("--objectclear_root", type=str, default=str(DEFAULT_OBJECTCLEAR_ROOT))
    parser.add_argument("--corne_root", type=str, default=str(DEFAULT_CORNE_ROOT))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--target_total", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _iter_image_files(directory: Path) -> Iterable[Path]:
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _index_image_files(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _iter_image_files(directory):
        index.setdefault(path.stem, path)
    return index


def _resolve_companion_file(directory: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _discover_objectclear_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    input_index = _index_image_files(root / "input")
    gt_index = _index_image_files(root / "gt")
    effect_index = _index_image_files(root / "object_effect_mask")
    exact_index = _index_image_files(root / "object_mask")
    common_stems = sorted(set(input_index) & set(gt_index) & set(effect_index) & set(exact_index))
    for stem in common_stems:
        samples.append(
            Sample(
                source_kind="objectclear",
                stem=stem,
                input_path=input_index[stem],
                gt_path=gt_index[stem],
                mask_rem_path=effect_index[stem],
                exact_mask_path=exact_index[stem],
            )
        )
    return samples


def _discover_corne_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    exact_dir = root / "mask_sam"
    if not exact_dir.is_dir():
        return samples
    exact_index = _index_image_files(exact_dir)
    shot_dir = root / "shot"
    bg_dir = root / "bg"
    effect_dir = root / "mask-check"
    for stem in sorted(exact_index):
        shot_path = _resolve_companion_file(shot_dir, stem)
        gt_path = _resolve_companion_file(bg_dir, stem)
        mask_rem_path = _resolve_companion_file(effect_dir, stem)
        exact_mask_path = exact_index[stem]
        if shot_path is None or gt_path is None or mask_rem_path is None:
            continue
        samples.append(
            Sample(
                source_kind="corne",
                stem=stem,
                input_path=shot_path,
                gt_path=gt_path,
                mask_rem_path=mask_rem_path,
                exact_mask_path=exact_mask_path,
            )
        )
    return samples


def _balanced_labels(total: int, labels: list[str], rng: random.Random) -> list[str]:
    base = total // len(labels)
    remainder = total % len(labels)
    out: list[str] = []
    for idx, label in enumerate(labels):
        count = base + (1 if idx < remainder else 0)
        out.extend([label] * count)
    rng.shuffle(out)
    return out


def _make_link(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.is_symlink():
        if os.path.realpath(dst) == str(src.resolve()):
            return
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")
        dst.unlink()
    elif dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src.resolve())


def _load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if mask.max() == 0:
        return mask
    return (mask > 0).astype(np.uint8)


def _scribble_worker(task: tuple[str, str, int]) -> str:
    mask_path, output_path, seed = task
    mask = _load_binary_mask(Path(mask_path))
    scribble, _ = simulate_user_scribble(mask, seed=seed)
    Image.fromarray(scribble, mode="L").save(output_path)
    return output_path


def main():
    args = parse_args()
    objectclear_root = Path(args.objectclear_root).resolve()
    corne_root = Path(args.corne_root).resolve()
    output_root = Path(args.output_root).resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)

    output_input_dir = output_root / "input"
    output_gt_dir = output_root / "gt"
    output_mask_rem_dir = output_root / "mask_rem"
    output_mask_inp_dir = output_root / "mask_inp"
    for d in [output_input_dir, output_gt_dir, output_mask_rem_dir, output_mask_inp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    objectclear_samples = _discover_objectclear_samples(objectclear_root)
    corne_samples = _discover_corne_samples(corne_root)
    if len(objectclear_samples) == 0:
        raise ValueError(f"No valid ObjectClear samples found under {objectclear_root}")
    if len(corne_samples) == 0:
        raise ValueError(f"No valid CORNE samples with mask_sam found under {corne_root}")
    if len(objectclear_samples) > args.target_total:
        raise ValueError(
            f"ObjectClear sample count {len(objectclear_samples)} exceeds target_total {args.target_total}. "
            "Increase target_total or filter the input root."
        )

    fill_count = args.target_total - len(objectclear_samples)
    if fill_count > len(corne_samples):
        raise ValueError(
            f"Need {fill_count} CORNE samples with mask_sam, but only found {len(corne_samples)}."
        )

    rng.shuffle(corne_samples)
    selected_samples = list(objectclear_samples) + corne_samples[:fill_count]
    mode_labels = _balanced_labels(len(selected_samples), ["exact", "effect", "scribble"], rng)
    scribble_bases = _balanced_labels(mode_labels.count("scribble"), ["exact", "effect"], rng)

    scribble_tasks: list[tuple[str, str, int]] = []
    manifest_path = output_root / "manifest.jsonl"
    summary_path = output_root / "summary.json"
    mode_counts = {"exact": 0, "effect": 0, "scribble": 0}
    source_counts = {"objectclear": 0, "corne": 0}
    scribble_base_counts = {"exact": 0, "effect": 0}
    scribble_base_index = 0

    started = time.perf_counter()
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for idx, (sample, mode) in enumerate(tqdm(list(zip(selected_samples, mode_labels)), desc="prepare", dynamic_ncols=True)):
            mode_counts[mode] += 1
            source_counts[sample.source_kind] += 1

            input_dst = output_input_dir / sample.input_path.name
            gt_dst = output_gt_dir / sample.gt_path.name
            mask_rem_dst = output_mask_rem_dir / f"{sample.stem}.png"
            mask_inp_dst = output_mask_inp_dir / f"{sample.stem}.png"

            _make_link(sample.input_path, input_dst, args.overwrite)
            _make_link(sample.gt_path, gt_dst, args.overwrite)
            _make_link(sample.mask_rem_path, mask_rem_dst, args.overwrite)

            if mode == "scribble":
                scribble_base = scribble_bases[scribble_base_index]
                scribble_base_index += 1
                scribble_base_counts[scribble_base] += 1
                base_mask_path = sample.exact_mask_path if scribble_base == "exact" else sample.mask_rem_path
                scribble_tasks.append((str(base_mask_path), str(mask_inp_dst), args.seed + idx))
                mask_inp_source = str(base_mask_path)
            elif mode == "exact":
                _make_link(sample.exact_mask_path, mask_inp_dst, args.overwrite)
                mask_inp_source = str(sample.exact_mask_path)
            else:
                _make_link(sample.mask_rem_path, mask_inp_dst, args.overwrite)
                mask_inp_source = str(sample.mask_rem_path)

            record = {
                "index": idx,
                "mode": mode,
                "source_kind": sample.source_kind,
                "source_stem": sample.stem,
                "input_path": str(input_dst),
                "gt_path": str(gt_dst),
                "mask_rem_path": str(mask_rem_dst),
                "mask_inp_path": str(mask_inp_dst),
                "source_input_path": str(sample.input_path),
                "source_gt_path": str(sample.gt_path),
                "source_mask_rem_path": str(sample.mask_rem_path),
                "source_exact_mask_path": str(sample.exact_mask_path),
                "mask_inp_source_path": mask_inp_source,
            }
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    if scribble_tasks:
        max_workers = max(1, min(int(args.num_workers), os.cpu_count() or int(args.num_workers)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(_scribble_worker, scribble_tasks), total=len(scribble_tasks), desc="scribble", dynamic_ncols=True))

    elapsed = time.perf_counter() - started
    summary = {
        "output_root": str(output_root),
        "target_total": args.target_total,
        "selected_total": len(selected_samples),
        "objectclear_count": len(objectclear_samples),
        "corne_count": len(corne_samples[:fill_count]),
        "mode_counts": mode_counts,
        "source_counts": source_counts,
        "scribble_base_counts": scribble_base_counts,
        "seed": args.seed,
        "num_workers": max(1, min(int(args.num_workers), os.cpu_count() or int(args.num_workers))),
        "elapsed_sec": elapsed,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
