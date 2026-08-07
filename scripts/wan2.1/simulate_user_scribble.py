#!/usr/bin/env python

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.interpolate import splprep, splev
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate user scribbles from binary masks.")
    parser.add_argument("--mask_path", type=str, default=None, help="Single binary mask path.")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory of binary masks.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for scribble masks.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--num_samples", type=int, default=None, help="Optional cap on number of masks to process.")
    parser.add_argument("--preview", action="store_true", help="Also save contact-sheet previews.")
    parser.add_argument("--preview_cols", type=int, default=3, help="Columns in preview sheets.")
    return parser.parse_args()


def _iter_mask_paths(args) -> list[Path]:
    if args.mask_path is not None:
        return [Path(args.mask_path).resolve()]
    if args.mask_dir is not None:
        mask_dir = Path(args.mask_dir).resolve()
        mask_paths = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
        if args.num_samples is not None:
            mask_paths = mask_paths[: args.num_samples]
        return mask_paths
    raise ValueError("Provide either --mask_path or --mask_dir.")


def _load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if mask.max() == 0:
        return mask
    return (mask > 0).astype(np.uint8)


def _extract_contour_points(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0, :].astype(np.float32)
    return points


def _bbox_and_centroid(mask: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        h, w = mask.shape[:2]
        bbox = (0, 0, w, h)
        centroid = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        return bbox, centroid
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    centroid = np.array([xs.mean(), ys.mean()], dtype=np.float32)
    return bbox, centroid


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / norm).astype(np.float32)


def _perpendicular(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=np.float32)


def _sample_contour_point(contour: np.ndarray, rng: random.Random) -> np.ndarray:
    idx = rng.randrange(len(contour))
    return contour[idx].astype(np.float32)


def _fit_curve(points: np.ndarray, num_samples: int = 96) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 4:
        raise ValueError("Need at least 4 control points for spline fitting.")
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    unique = np.unique(pts, axis=0)
    if unique.shape[0] < 4:
        return np.repeat(unique[:1], num_samples, axis=0)

    x = unique[:, 0]
    y = unique[:, 1]
    k = min(3, unique.shape[0] - 1)
    try:
        tck, _ = splprep([x, y], s=0.0, k=k)
        u = np.linspace(0.0, 1.0, num_samples)
        out = splev(u, tck)
        return np.stack(out, axis=1).astype(np.float32)
    except Exception:
        idx = np.linspace(0, unique.shape[0] - 1, num_samples)
        left = np.floor(idx).astype(int)
        right = np.clip(left + 1, 0, unique.shape[0] - 1)
        alpha = idx - left
        return (unique[left] * (1.0 - alpha[:, None]) + unique[right] * alpha[:, None]).astype(np.float32)


def _draw_variable_width_curve(canvas: np.ndarray, curve: np.ndarray, width_start: float, width_end: float) -> None:
    h, w = canvas.shape[:2]
    n = len(curve)
    if n == 0:
        return
    for i, p in enumerate(curve):
        t = 0.0 if n == 1 else i / float(n - 1)
        radius = max(1, int(round((width_start * (1.0 - t) + width_end * t) / 2.0)))
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), radius, 255, thickness=-1)


def simulate_user_scribble(mask: np.ndarray, seed: int = 0) -> tuple[np.ndarray, dict]:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.shape}")

    rng = random.Random(seed)
    contour = _extract_contour_points(mask)
    bbox, centroid = _bbox_and_centroid(mask)
    bbox_short_side = max(1.0, float(min(bbox[2], bbox[3])))
    major_axis = max(1.0, float(max(bbox[2], bbox[3])))

    scribble = np.zeros_like(mask, dtype=np.uint8)
    if contour.shape[0] == 0:
        return scribble, {
            "seed": seed,
            "num_strokes": 0,
            "bbox": list(bbox),
            "bbox_short_side": bbox_short_side,
            "major_axis": major_axis,
            "empty_mask": True,
        }

    num_strokes = rng.randint(2, 5)
    attempts_per_stroke = 64
    successful = 0

    for _ in range(num_strokes):
        found = False
        for _attempt in range(attempts_per_stroke):
            A = _sample_contour_point(contour, rng)
            B = _sample_contour_point(contour, rng)
            if np.linalg.norm(A - B) <= 0.4 * major_axis:
                continue

            dir_A = _normalize(A - centroid)
            dir_B = _normalize(B - centroid)
            if not np.any(dir_A):
                dir_A = _normalize(A - np.array([mask.shape[1] / 2.0, mask.shape[0] / 2.0], dtype=np.float32))
            if not np.any(dir_B):
                dir_B = _normalize(B - np.array([mask.shape[1] / 2.0, mask.shape[0] / 2.0], dtype=np.float32))

            extend_A = rng.uniform(0.05, 0.15) * bbox_short_side
            extend_B = rng.uniform(0.05, 0.15) * bbox_short_side
            A_ext = A + dir_A * extend_A
            B_ext = B + dir_B * extend_B

            chord = B_ext - A_ext
            perp = _perpendicular(_normalize(chord))
            if not np.any(perp):
                perp = np.array([0.0, 1.0], dtype=np.float32)

            p1 = (1.0 / 3.0) * A_ext + (2.0 / 3.0) * B_ext
            p2 = (2.0 / 3.0) * A_ext + (1.0 / 3.0) * B_ext
            offset_scale1 = rng.uniform(-0.2, 0.2) * float(np.linalg.norm(chord))
            offset_scale2 = rng.uniform(-0.2, 0.2) * float(np.linalg.norm(chord))
            P1 = p1 + perp * offset_scale1
            P2 = p2 + perp * offset_scale2

            ctrl = np.stack([A_ext, P1, P2, B_ext], axis=0)
            curve = _fit_curve(ctrl, num_samples=96)

            width_start = rng.uniform(0.05, 0.12) * bbox_short_side
            width_end = rng.uniform(0.05, 0.12) * bbox_short_side
            _draw_variable_width_curve(scribble, curve, width_start, width_end)
            found = True
            successful += 1
            break

        if not found:
            continue

    return scribble, {
        "seed": seed,
        "num_strokes": num_strokes,
        "successful_strokes": successful,
        "bbox": list(bbox),
        "bbox_short_side": bbox_short_side,
        "major_axis": major_axis,
        "empty_mask": False,
    }


def _make_preview(mask: np.ndarray, scribble: np.ndarray) -> Image.Image:
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("RGB")
    scribble_img = Image.fromarray(scribble, mode="L").convert("RGB")
    canvas = Image.new("RGB", (mask_img.width * 2, mask_img.height), "white")
    canvas.paste(mask_img, (0, 0))
    canvas.paste(scribble_img, (mask_img.width, 0))
    return canvas


def main():
    args = parse_args()
    mask_paths = _iter_mask_paths(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview"
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for idx, mask_path in enumerate(tqdm(mask_paths, desc="scribble", dynamic_ncols=True)):
        mask = _load_binary_mask(mask_path)
        scribble, meta = simulate_user_scribble(mask, seed=args.seed + idx)

        out_path = output_dir / f"{mask_path.stem}_scribble.png"
        Image.fromarray(scribble, mode="L").save(out_path)
        meta["mask_path"] = str(mask_path)
        meta["scribble_path"] = str(out_path)

        if args.preview:
            preview = _make_preview(mask, scribble)
            preview_path = preview_dir / f"{mask_path.stem}_preview.png"
            preview.save(preview_path)
            meta["preview_path"] = str(preview_path)

        summaries.append(meta)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(json.dumps(summaries[: min(5, len(summaries))], indent=2))


if __name__ == "__main__":
    main()
