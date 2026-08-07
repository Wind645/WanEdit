#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
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

from videox_fun.utils.singleturn_utils import normalize_singleturn_sample_size, preprocess_singleturn_image


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Patch existing SingleTurn inference outputs by re-blending the source image with the final generated "
            "frame, using the second frame of the full mp4 as ObjectClear-style binary/dilated/blurred alpha."
        )
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Root directory containing rank*/raw_infer_manifest.json files.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional single raw_infer_manifest.json to patch.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        nargs="+",
        default=[480, 832],
        help="Letterboxed sample size used by the original inference run.",
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
        "--inplace",
        action="store_true",
        help="Overwrite the existing full_last_frame / tail_last_frame files in place.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="edgeblur",
        help="Suffix appended to patched PNGs when --inplace is not set.",
    )
    return parser.parse_args()


def _find_manifest_paths(args) -> list[Path]:
    if args.manifest_path is not None:
        return [Path(args.manifest_path).resolve()]
    if args.run_dir is None:
        raise ValueError("Provide either --run_dir or --manifest_path.")

    run_dir = Path(args.run_dir).resolve()
    manifests = sorted(run_dir.rglob("raw_infer_manifest.json"))
    if not manifests:
        raise ValueError(f"No raw_infer_manifest.json files found under {run_dir}")
    return manifests


def _iter_entries(manifest_path: Path) -> Iterable[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, list):
        raise ValueError(f"Expected {manifest_path} to contain a list, got {type(manifest)}")
    for entry in manifest:
        yield entry


def _load_video_rgb(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return np.stack(frames, axis=0)


def _load_source_frame_for_blending(source_path: str, *, height: int, width: int, sample_size: tuple[int, int]) -> torch.Tensor:
    if (height, width) == tuple(sample_size):
        source = preprocess_singleturn_image(source_path, sample_size, add_batch_dim=False, add_frame_dim=False)
        source = ((source.float() + 1.0) / 2.0).clamp(0, 1)
    else:
        with Image.open(source_path) as image:
            image = image.convert("RGB").resize((width, height), resample=Image.BILINEAR)
            source = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return source.unsqueeze(0).unsqueeze(2)


def _objectclear_style_blend_alpha(alpha: torch.Tensor) -> torch.Tensor:
    if alpha.ndim != 5 or alpha.shape[1] != 1 or alpha.shape[2] != 1:
        raise ValueError(f"alpha must have shape (B, 1, 1, H, W), got {tuple(alpha.shape)}")

    alpha_2d = alpha[:, 0, 0].detach().cpu().numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    softened = []
    for sample in alpha_2d:
        binary = (sample >= 0.5).astype(np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1).astype(np.float32)
        blurred = cv2.GaussianBlur(dilated, (9, 9), sigmaX=2)
        merged = np.maximum(binary.astype(np.float32), blurred)
        softened.append(torch.from_numpy(merged))

    softened_alpha = torch.stack(softened, dim=0).unsqueeze(1).unsqueeze(2)
    return softened_alpha.to(device=alpha.device, dtype=alpha.dtype)


def _resolve_source_path(entry: dict, manifest_path: Path) -> str:
    for key in ("source_image", "image_path"):
        value = entry.get(key)
        if value:
            return str(value)

    metadata_path = entry.get("metadata")
    if metadata_path:
        metadata_path = Path(metadata_path)
        if not metadata_path.is_absolute():
            metadata_path = manifest_path.parent / metadata_path
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for key in ("source_image", "image_path"):
                value = metadata.get(key)
                if value:
                    return str(value)
    return ""


def _patched_output_path(path: str, suffix: str, *, inplace: bool) -> str:
    if inplace:
        return path
    file_path = Path(path)
    return str(file_path.with_name(f"{file_path.stem}_{suffix}{file_path.suffix}"))


def _validate_odd_kernel_size(name: str, value: int) -> int:
    value = int(value)
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}")
    return value


def _blend_case(
    *,
    source_path: str,
    full_video_path: str,
    full_last_frame_path: str,
    tail_last_frame_path: str,
    sample_size: tuple[int, int],
    mask_blend_threshold: float,
    mask_blend_dilate_kernel_size: int,
    mask_blend_blur_kernel_size: int,
    mask_blend_blur_sigma: float,
    inplace: bool,
    output_suffix: str,
) -> dict:
    frames_rgb = _load_video_rgb(full_video_path)
    if frames_rgb.shape[0] <= 1:
        raise ValueError(f"Video must contain at least 2 frames: {full_video_path}")

    height, width = frames_rgb.shape[1:3]
    source_frame = _load_source_frame_for_blending(
        source_path,
        height=height,
        width=width,
        sample_size=sample_size,
    ).to(dtype=torch.float32)

    alpha = torch.from_numpy(frames_rgb[1].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    alpha = alpha.mean(dim=1, keepdim=True).clamp(0, 1)
    if not (0.0 <= mask_blend_threshold <= 1.0):
        raise ValueError(f"mask_blend_threshold must be in [0, 1], got {mask_blend_threshold}")
    mask_blend_dilate_kernel_size = _validate_odd_kernel_size(
        "mask_blend_dilate_kernel_size",
        mask_blend_dilate_kernel_size,
    )
    mask_blend_blur_kernel_size = _validate_odd_kernel_size(
        "mask_blend_blur_kernel_size",
        mask_blend_blur_kernel_size,
    )
    if mask_blend_blur_sigma <= 0:
        raise ValueError(f"mask_blend_blur_sigma must be positive, got {mask_blend_blur_sigma}")

    alpha_2d = alpha[:, 0, 0].detach().cpu().numpy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (mask_blend_dilate_kernel_size, mask_blend_dilate_kernel_size),
    )
    softened = []
    for sample in alpha_2d:
        binary = (sample >= mask_blend_threshold).astype(np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1).astype(np.float32)
        blurred = cv2.GaussianBlur(
            dilated,
            (mask_blend_blur_kernel_size, mask_blend_blur_kernel_size),
            sigmaX=float(mask_blend_blur_sigma),
        )
        merged = np.maximum(binary.astype(np.float32), blurred)
        softened.append(torch.from_numpy(merged))
    alpha = torch.stack(softened, dim=0).unsqueeze(1).unsqueeze(2).to(device=alpha.device, dtype=alpha.dtype)

    last_frame = torch.from_numpy(frames_rgb[-1].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    blended_last = alpha * last_frame + (1.0 - alpha) * source_frame
    frame = blended_last[0, :, 0].permute(1, 2, 0).clamp(0, 1).numpy()
    image = Image.fromarray((frame * 255).astype(np.uint8))

    blended_full_path = _patched_output_path(full_last_frame_path, output_suffix, inplace=inplace)
    blended_tail_path = _patched_output_path(tail_last_frame_path, output_suffix, inplace=inplace)
    os.makedirs(os.path.dirname(blended_full_path), exist_ok=True)
    os.makedirs(os.path.dirname(blended_tail_path), exist_ok=True)
    image.save(blended_full_path)
    image.save(blended_tail_path)

    return {
        "source_path": source_path,
        "full_video_path": full_video_path,
        "full_last_frame_path": blended_full_path,
        "tail_last_frame_path": blended_tail_path,
    }


def main():
    args = parse_args()
    if (args.run_dir is None) == (args.manifest_path is None):
        raise ValueError("Provide exactly one of --run_dir or --manifest_path.")
    sample_size = normalize_singleturn_sample_size(args.sample_size)
    manifest_paths = _find_manifest_paths(args)

    summary = {
        "sample_size": list(sample_size),
        "mask_blend_threshold": args.mask_blend_threshold,
        "mask_blend_dilate_kernel_size": args.mask_blend_dilate_kernel_size,
        "mask_blend_blur_kernel_size": args.mask_blend_blur_kernel_size,
        "mask_blend_blur_sigma": args.mask_blend_blur_sigma,
        "inplace": bool(args.inplace),
        "output_suffix": args.output_suffix,
        "manifests": [],
    }

    total_cases = 0
    processed_cases = 0
    skipped_cases = 0

    for manifest_path in manifest_paths:
        manifest_entries = list(_iter_entries(manifest_path))
        manifest_summary = {
            "manifest_path": str(manifest_path),
            "num_entries": len(manifest_entries),
            "processed": 0,
            "skipped": 0,
        }

        for entry in tqdm(manifest_entries, desc=manifest_path.parent.name, dynamic_ncols=True):
            total_cases += 1
            output_paths = entry.get("output_paths") or {}
            full_video_path = str(output_paths.get("full_video", ""))
            full_last_frame_path = str(output_paths.get("full_last_frame", ""))
            tail_last_frame_path = str(output_paths.get("tail_last_frame", ""))
            source_path = _resolve_source_path(entry, manifest_path)

            if not full_video_path or not os.path.exists(full_video_path):
                skipped_cases += 1
                manifest_summary["skipped"] += 1
                continue
            if not full_last_frame_path or not tail_last_frame_path:
                skipped_cases += 1
                manifest_summary["skipped"] += 1
                continue
            if not source_path or not os.path.exists(source_path):
                skipped_cases += 1
                manifest_summary["skipped"] += 1
                continue

            try:
                _blend_case(
                    source_path=source_path,
                    full_video_path=full_video_path,
                    full_last_frame_path=full_last_frame_path,
                    tail_last_frame_path=tail_last_frame_path,
                    sample_size=sample_size,
                    mask_blend_threshold=args.mask_blend_threshold,
                    mask_blend_dilate_kernel_size=args.mask_blend_dilate_kernel_size,
                    mask_blend_blur_kernel_size=args.mask_blend_blur_kernel_size,
                    mask_blend_blur_sigma=args.mask_blend_blur_sigma,
                    inplace=args.inplace,
                    output_suffix=args.output_suffix,
                )
            except Exception as exc:
                skipped_cases += 1
                manifest_summary["skipped"] += 1
                manifest_summary.setdefault("errors", []).append(
                    {
                        "full_video_path": full_video_path,
                        "source_path": source_path,
                        "error": str(exc),
                    }
                )
                continue

            processed_cases += 1
            manifest_summary["processed"] += 1

        summary["manifests"].append(manifest_summary)

    summary["total_cases"] = total_cases
    summary["processed_cases"] = processed_cases
    summary["skipped_cases"] = skipped_cases

    if args.run_dir is not None:
        summary_path = Path(args.run_dir).resolve() / "mask_blend_patch_summary.json"
    else:
        summary_path = Path(args.manifest_path).resolve().parent / "mask_blend_patch_summary.json"
    summary["summary_path"] = str(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
