#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect and visualize sampled SingleTurn refine weight maps.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory containing refine cache .pt files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for sampled visualizations.")
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 199, 599, 999],
        help="Sample indices in the sorted .pt file list.",
    )
    parser.add_argument("--frame_index", type=int, default=5, help="0-based frame index to visualize, default is F6.")
    return parser.parse_args()


def make_overview(mask_img: Image.Image, weight_img: Image.Image, out_path: Path, frame_label: str):
    mask_rgb = ImageOps.contain(mask_img.convert("RGB"), (320, 184))
    weight_rgb = ImageOps.contain(weight_img.convert("RGB"), (320, 184))
    canvas = Image.new("RGB", (640, 220), "white")
    canvas.paste(mask_rgb, ((320 - mask_rgb.width) // 2, 8))
    canvas.paste(weight_rgb, (320 + (320 - weight_rgb.width) // 2, 8))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 192), "mask_check", fill="black")
    draw.text((330, 192), frame_label, fill="black")
    canvas.save(out_path)


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_files = sorted(cache_dir.glob("*.pt"))
    if not cache_files:
        raise ValueError(f"No .pt files found in {cache_dir}")

    selected_indices = sorted({min(max(0, idx), len(cache_files) - 1) for idx in args.indices})
    selected_files = [cache_files[idx] for idx in selected_indices]

    summary = []
    for path in selected_files:
        payload = torch.load(path, map_location="cpu")
        if "refinement_loss_weight_map" not in payload:
            raise ValueError(f"Sample {path} is missing refinement_loss_weight_map")

        weight_map = payload["refinement_loss_weight_map"].float()  # (1, T, H, W)
        if weight_map.ndim != 4:
            raise ValueError(f"Unexpected weight map shape for {path}: {tuple(weight_map.shape)}")

        sample_dir = output_dir / path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        unique_vals, counts = torch.unique(weight_map, return_counts=True)
        unique_values = {float(v.item()): int(c.item()) for v, c in zip(unique_vals, counts)}
        prefix_zero = bool(torch.all(weight_map[:, :5] == 0).item())
        active_map = weight_map[:, 5:]
        active_nonzero = active_map[active_map > 0]
        active_min = float(active_nonzero.min().item()) if active_nonzero.numel() else 0.0
        active_max = float(active_nonzero.max().item()) if active_nonzero.numel() else 0.0

        frame_index = min(max(0, args.frame_index), weight_map.shape[1] - 1)
        frame_map = weight_map[0, frame_index].numpy()
        vmax = max(float(frame_map.max()), 1.0)
        frame_norm = (frame_map / vmax * 255).clip(0, 255).astype(np.uint8)
        weight_img = Image.fromarray(frame_norm, mode="L").convert("RGB")
        weight_img.save(sample_dir / f"weight_map_f{frame_index + 1}.png")

        mask_check_path = payload.get("mask_check_image")
        if mask_check_path and Path(mask_check_path).exists():
            mask_img = Image.open(mask_check_path).convert("L")
            mask_img.save(sample_dir / "mask_check.png")
        else:
            mask_img = Image.new("L", (832, 480), 0)

        make_overview(mask_img, weight_img, sample_dir / "overview.png", f"weight_map_f{frame_index + 1}")

        sample_summary = {
            "cache_path": str(path),
            "weight_map_shape": list(weight_map.shape),
            "prefix_zero_f1_f5": prefix_zero,
            "active_min": active_min,
            "active_max": active_max,
            "unique_values": unique_values,
            "mask_check_image": str(mask_check_path),
            "preview_dir": str(sample_dir),
        }
        summary.append(sample_summary)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
