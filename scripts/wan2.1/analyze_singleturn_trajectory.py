#!/usr/bin/env python

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

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
from videox_fun.models import AutoencoderKLWan


def resolve_model_path(model_root: str, subpath: str, default_subpath: str) -> str:
    subpath = str(subpath or default_subpath)
    if os.path.isabs(subpath):
        return subpath

    model_root_candidate = os.path.join(model_root, subpath)
    if os.path.exists(model_root_candidate):
        return model_root_candidate

    sibling_candidate = os.path.join(os.path.dirname(os.path.abspath(model_root)), subpath)
    if os.path.exists(sibling_candidate):
        return sibling_candidate

    return subpath


def get_weight_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16


def load_video_rgb(video_path: str) -> np.ndarray:
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


def encode_video_frames_individually(
    vae: AutoencoderKLWan,
    frames_rgb: np.ndarray,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    latents = []
    with torch.no_grad():
        for frame in frames_rgb:
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)
            tensor = tensor.mul(2.0).sub(1.0)[None, :, None].to(device=device, dtype=weight_dtype)
            latent = vae.encode(tensor)[0].mode()
            latents.append(latent[0].detach().cpu().float())
    return torch.cat(latents, dim=1)


def _latent_frame(payload: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    latent = payload[key].detach().cpu().float()
    if latent.ndim != 4 or latent.shape[1] != 1:
        raise ValueError(f"{key} must have shape (C, 1, H, W), got {tuple(latent.shape)}")
    return latent


def build_test_time_interpolation_latents(
    payload: Dict[str, torch.Tensor],
    generated_latents: torch.Tensor,
    *,
    use_mask_sam: bool,
    test_time_anchor: torch.Tensor,
    target_latent: torch.Tensor,
    target_role: str,
    corruption_frames: int,
    restoration_frames: int,
    interpolation_gamma: float,
) -> Tuple[torch.Tensor, List[Dict[str, float | str]], Dict[str, torch.Tensor]]:
    mask_frame_key = "mask_sam_latent" if use_mask_sam and "mask_sam_latent" in payload else "mask_check_latent"
    mask_frame_latent = _latent_frame(payload, mask_frame_key)
    source_latent = _latent_frame(payload, "source_frame_latent")
    generated_mask_prediction = generated_latents[:, 1:2]

    frames = [mask_frame_latent, generated_mask_prediction, source_latent]
    roles: List[Dict[str, float | str]] = [
        {"role": "mask_condition", "segment": "condition", "gt_alpha": 0.0},
        {"role": "generated_mask_prediction", "segment": "condition", "gt_alpha": 0.0},
        {"role": "source", "segment": "condition", "gt_alpha": 0.0},
    ]

    for frame_idx in range(1, corruption_frames + 1):
        alpha = (float(frame_idx) / float(corruption_frames + 1)) ** float(interpolation_gamma)
        frames.append((1.0 - alpha) * source_latent + alpha * test_time_anchor)
        roles.append({"role": "corruption_interp", "segment": "source_to_noisy_anchor", "gt_alpha": alpha})

    frames.append(test_time_anchor)
    roles.append({"role": "test_time_anchor", "segment": "source_to_noisy_anchor", "gt_alpha": 1.0})

    for frame_idx in range(1, restoration_frames + 1):
        progress = float(frame_idx) / float(restoration_frames + 1)
        alpha = 1.0 - (1.0 - progress) ** float(interpolation_gamma)
        frames.append((1.0 - alpha) * test_time_anchor + alpha * target_latent)
        roles.append({"role": "restoration_interp", "segment": "noisy_anchor_to_target", "gt_alpha": alpha})

    frames.append(target_latent)
    roles.append({"role": target_role, "segment": "noisy_anchor_to_target", "gt_alpha": 1.0})
    anchors = {
        "source": source_latent[:, 0],
        "noisy_anchor": test_time_anchor[:, 0],
        "target": target_latent[:, 0],
    }
    return torch.cat(frames, dim=1), roles, anchors


def compute_offpath_metrics(
    generated_latents: torch.Tensor,
    reference_latents: torch.Tensor,
    roles: List[Dict[str, float | str]],
    anchors: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    source_latent = anchors["source"]
    noisy_anchor = anchors["noisy_anchor"]
    target_latent = anchors["target"]

    offpath_frames = []
    progress_delta_frames = []
    for frame_idx, role in enumerate(roles):
        generated = generated_latents[:, frame_idx]
        segment = str(role["segment"])
        gt_alpha = float(role["gt_alpha"])
        if segment == "source_to_noisy_anchor":
            start, end = source_latent, noisy_anchor
        elif segment == "noisy_anchor_to_target":
            start, end = noisy_anchor, target_latent
        else:
            offpath_frames.append((generated - reference_latents[:, frame_idx]).norm(p=2, dim=0))
            progress_delta_frames.append(torch.zeros_like(offpath_frames[-1]))
            continue

        direction = end - start
        denom = direction.pow(2).sum(dim=0).clamp_min(1e-8)
        alpha_hat = ((generated - start) * direction).sum(dim=0) / denom
        projected = start + alpha_hat.clamp(0.0, 1.0)[None] * direction
        offpath_frames.append((generated - projected).norm(p=2, dim=0))
        progress_delta_frames.append(alpha_hat - gt_alpha)

    return torch.stack(offpath_frames, dim=0), torch.stack(progress_delta_frames, dim=0)


def load_masks(mask_path: str, image_size: Tuple[int, int], latent_size: Tuple[int, int], threshold: float) -> Tuple[np.ndarray, torch.Tensor]:
    image_width, image_height = image_size
    latent_height, latent_width = latent_size
    with Image.open(mask_path) as image:
        mask_image = image.convert("L")
        full_mask = np.asarray(mask_image.resize((image_width, image_height), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
        latent_mask = np.asarray(mask_image.resize((latent_width, latent_height), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    full_mask = full_mask >= float(threshold)
    latent_mask = torch.from_numpy(latent_mask >= float(threshold))
    if not bool(latent_mask.any()):
        raise ValueError(f"Mask is empty after resizing: {mask_path}")
    return full_mask, latent_mask


def _robust_scale(metric: torch.Tensor, latent_mask: torch.Tensor) -> float:
    values = metric[:, latent_mask].detach().cpu().numpy().reshape(-1)
    if values.size == 0:
        return 1.0
    scale = float(np.quantile(values, 0.99))
    return max(scale, 1e-8)


def save_metric_visuals(
    metric: torch.Tensor,
    frames_rgb: np.ndarray,
    full_mask: np.ndarray,
    output_dir: str,
    prefix: str,
    *,
    overlay_alpha: float,
) -> Tuple[List[str], List[str], str]:
    os.makedirs(output_dir, exist_ok=True)
    frame_count, image_height, image_width = frames_rgb.shape[:3]
    resized = F.interpolate(
        metric[:, None].float(),
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    mask = full_mask.astype(np.float32)
    scale = max(float(np.quantile(resized[:, full_mask].numpy().reshape(-1), 0.99)), 1e-8)

    heatmap_paths = []
    overlay_paths = []
    overlay_frames = []
    for frame_idx in range(frame_count):
        heat = (resized[frame_idx].numpy() / scale).clip(0.0, 1.0) * mask
        heatmap = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        heatmap[..., 0] = (heat * 255).astype(np.uint8)

        base = frames_rgb[frame_idx].astype(np.uint8)
        red = np.zeros_like(base)
        red[..., 0] = 255
        alpha = (float(overlay_alpha) * heat)[..., None]
        overlay = (base.astype(np.float32) * (1.0 - alpha) + red.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
        overlay = _draw_label(overlay, f"{prefix} frame {frame_idx:02d}")

        heatmap_path = os.path.join(output_dir, f"{prefix}_frame{frame_idx:02d}_heatmap.png")
        overlay_path = os.path.join(output_dir, f"{prefix}_frame{frame_idx:02d}_overlay.png")
        Image.fromarray(heatmap).save(heatmap_path)
        Image.fromarray(overlay).save(overlay_path)
        heatmap_paths.append(heatmap_path)
        overlay_paths.append(overlay_path)
        overlay_frames.append(overlay)

    video_path = os.path.join(output_dir, f"{prefix}_overlay.mp4")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        4.0,
        (image_width, image_height),
    )
    for frame in overlay_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return heatmap_paths, overlay_paths, video_path


def _draw_label(frame_rgb: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 230, 34), fill=(0, 0, 0))
    draw.text((14, 14), text, fill=(255, 255, 255))
    return np.asarray(image)


def save_mask_offpath_line_plot(rows: List[Dict[str, float | int | str]], output_dir: str) -> str:
    path = os.path.join(output_dir, "trajectory_mask_offpath_line.png")
    width, height = 960, 420
    left, right, top, bottom = 72, 28, 34, 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(row["mask_offpath_mean"]) for row in rows]
    frames = [int(row["frame"]) for row in rows]
    y_max = max(max(values) * 1.12, 1e-6)
    x_min, x_max = min(frames), max(frames)
    x_span = max(x_max - x_min, 1)

    def point(frame: int, value: float) -> tuple[int, int]:
        x = left + int(round((frame - x_min) / x_span * plot_width))
        y = top + int(round((1.0 - value / y_max) * plot_height))
        return x, y

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    axis_color = (35, 35, 35)
    grid_color = (225, 225, 225)
    line_color = (205, 34, 34)
    anchor_color = (42, 96, 190)

    for tick_idx in range(6):
        value = y_max * tick_idx / 5.0
        y = top + int(round((1.0 - tick_idx / 5.0) * plot_height))
        draw.line((left, y, width - right, y), fill=grid_color, width=1)
        draw.text((8, y - 7), f"{value:.2f}", fill=axis_color)

    draw.line((left, top, left, height - bottom), fill=axis_color, width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill=axis_color, width=2)
    for frame in frames:
        x, _ = point(frame, 0.0)
        draw.line((x, height - bottom, x, height - bottom + 5), fill=axis_color, width=1)
        draw.text((x - 8, height - bottom + 12), str(frame), fill=axis_color)

    points = [point(frame, value) for frame, value in zip(frames, values)]
    if len(points) > 1:
        draw.line(points, fill=line_color, width=4)
    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=line_color)

    for row in rows:
        if str(row["role"]) in ("test_time_anchor", "own_noisy_anchor", "noisy_anchor"):
            x, _ = point(int(row["frame"]), 0.0)
            draw.line((x, top, x, height - bottom), fill=anchor_color, width=2)
            draw.text((x + 6, top + 6), "anchor", fill=anchor_color)

    draw.text((left, 10), "Mask off-path distance over frames", fill=axis_color)
    draw.text((width // 2 - 42, height - 28), "frame", fill=axis_color)
    draw.text((8, 10), "L2", fill=axis_color)
    image.save(path)
    return path


def summarize_metrics(
    l2_error: torch.Tensor,
    offpath_error: torch.Tensor,
    progress_delta: torch.Tensor,
    latent_mask: torch.Tensor,
    roles: List[Dict[str, float | str]],
) -> List[Dict[str, float | int | str]]:
    rows = []
    outside_mask = ~latent_mask
    for frame_idx, role in enumerate(roles):
        mask_l2 = l2_error[frame_idx][latent_mask]
        outside_l2 = l2_error[frame_idx][outside_mask]
        mask_offpath = offpath_error[frame_idx][latent_mask]
        mask_progress_delta = progress_delta[frame_idx][latent_mask]
        rows.append(
            {
                "frame": frame_idx,
                "role": str(role["role"]),
                "segment": str(role["segment"]),
                "gt_alpha": float(role["gt_alpha"]),
                "mask_l2_mean": float(mask_l2.mean().item()),
                "mask_l2_p95": float(torch.quantile(mask_l2, 0.95).item()),
                "mask_l2_max": float(mask_l2.max().item()),
                "outside_l2_mean": float(outside_l2.mean().item()) if outside_l2.numel() else 0.0,
                "mask_offpath_mean": float(mask_offpath.mean().item()),
                "mask_offpath_p95": float(torch.quantile(mask_offpath, 0.95).item()),
                "mask_progress_delta_mean": float(mask_progress_delta.mean().item()),
                "mask_progress_delta_abs_mean": float(mask_progress_delta.abs().mean().item()),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SingleTurn generated latent trajectory against a training-style interpolation reference.")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to singleturn_meta.json.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=os.environ.get("MODEL_NAME", "/mnt/cpfs/jiachengliu/pretrained_models/Wan-AI/Wan2.1-T2V-14B"),
        help="Base Wan model path used to load the VAE.",
    )
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Wan config path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to <sample_dir>/trajectory_compare_test_time_anchor[_cache_target].")
    parser.add_argument("--corruption_frames", type=int, default=4, help="source->noisy_anchor interpolation frame count.")
    parser.add_argument("--restoration_frames", type=int, default=5, help="noisy_anchor->target interpolation frame count.")
    parser.add_argument("--interpolation_gamma", type=float, default=1.2, help="Interpolation gamma used for this experiment.")
    parser.add_argument("--anchor_frame", type=int, default=None, help="Generated frame used as the test-time anchor. Defaults to 3 + corruption_frames.")
    parser.add_argument(
        "--target_source",
        type=str,
        default="generated_final",
        choices=["generated_final", "cache_target"],
        help="Final reference target: generated final frame latent or cached target_latent.",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="VAE dtype on CUDA.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Threshold for the mask-region metrics.")
    parser.add_argument("--overlay_alpha", type=float, default=0.55, help="Max red overlay alpha.")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    video_path = metadata["outputs"]["full_video"]
    cache_path = metadata["cache_path"]
    mask_path = metadata.get("mask_frame_image") or metadata.get("mask_sam_image") or metadata.get("mask_check_image")
    default_dirname = "trajectory_compare_test_time_anchor"
    if args.target_source == "cache_target":
        default_dirname = "trajectory_compare_test_time_anchor_cache_target"
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.metadata_path), default_dirname)
    os.makedirs(output_dir, exist_ok=True)

    frames_rgb = load_video_rgb(video_path)
    payload = load_singleturn_cache_payload(cache_path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    weight_dtype = get_weight_dtype(args.dtype, device)

    config = OmegaConf.load(args.config_path)
    vae = AutoencoderKLWan.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
            "vae",
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).eval().to(device=device, dtype=weight_dtype)

    generated_latents = encode_video_frames_individually(vae, frames_rgb, device, weight_dtype)
    anchor_frame = int(args.anchor_frame) if args.anchor_frame is not None else 3 + int(args.corruption_frames)
    if anchor_frame <= 2 or anchor_frame >= generated_latents.shape[1] - 1:
        raise ValueError(
            f"--anchor_frame must be between source frame 2 and final frame {generated_latents.shape[1] - 1}, got {anchor_frame}."
        )
    test_time_anchor = generated_latents[:, anchor_frame : anchor_frame + 1]
    if args.target_source == "cache_target":
        target_latent = _latent_frame(payload, "target_latent")
        target_role = "cache_target"
    else:
        target_latent = generated_latents[:, -1:]
        target_role = "generated_target"
    reference_latents, roles, anchors = build_test_time_interpolation_latents(
        payload,
        generated_latents,
        use_mask_sam=bool(metadata.get("used_mask_sam", False)),
        test_time_anchor=test_time_anchor,
        target_latent=target_latent,
        target_role=target_role,
        corruption_frames=args.corruption_frames,
        restoration_frames=args.restoration_frames,
        interpolation_gamma=args.interpolation_gamma,
    )
    np.save(os.path.join(output_dir, "test_time_anchor.npy"), test_time_anchor.numpy())
    np.save(os.path.join(output_dir, f"{args.target_source}_target.npy"), target_latent.numpy())
    if frames_rgb.shape[0] != reference_latents.shape[1]:
        raise ValueError(f"Video has {frames_rgb.shape[0]} frames but reference trajectory has {reference_latents.shape[1]} frames.")
    if generated_latents.shape != reference_latents.shape:
        raise ValueError(
            f"Generated latents have shape {tuple(generated_latents.shape)} "
            f"but reference has {tuple(reference_latents.shape)}."
        )

    latent_height, latent_width = int(reference_latents.shape[-2]), int(reference_latents.shape[-1])
    image_height, image_width = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    full_mask, latent_mask = load_masks(
        mask_path,
        image_size=(image_width, image_height),
        latent_size=(latent_height, latent_width),
        threshold=args.mask_threshold,
    )

    l2_error = (generated_latents - reference_latents).norm(p=2, dim=0)
    offpath_error, progress_delta = compute_offpath_metrics(generated_latents, reference_latents, roles, anchors)
    rows = summarize_metrics(l2_error, offpath_error, progress_delta, latent_mask, roles)

    np.save(os.path.join(output_dir, "trajectory_reference_latents.npy"), reference_latents.numpy())
    np.save(os.path.join(output_dir, "trajectory_l2_error.npy"), l2_error.numpy())
    np.save(os.path.join(output_dir, "trajectory_offpath_error.npy"), offpath_error.numpy())
    np.save(os.path.join(output_dir, "trajectory_progress_delta.npy"), progress_delta.numpy())

    l2_heatmaps, l2_overlays, l2_video = save_metric_visuals(
        l2_error,
        frames_rgb,
        full_mask,
        output_dir,
        "trajectory_l2",
        overlay_alpha=args.overlay_alpha,
    )
    offpath_heatmaps, offpath_overlays, offpath_video = save_metric_visuals(
        offpath_error,
        frames_rgb,
        full_mask,
        output_dir,
        "trajectory_offpath",
        overlay_alpha=args.overlay_alpha,
    )

    csv_path = os.path.join(output_dir, "trajectory_metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    line_plot_path = save_mask_offpath_line_plot(rows, output_dir)

    summary = {
        "metadata_path": args.metadata_path,
        "video_path": video_path,
        "cache_path": cache_path,
        "mask_path": mask_path,
        "output_dir": output_dir,
        "corruption_frames": args.corruption_frames,
        "restoration_frames": args.restoration_frames,
        "interpolation_gamma": args.interpolation_gamma,
        "reference_mode": "test_time_generated_frame_anchor",
        "anchor_frame": anchor_frame,
        "target_source": args.target_source,
        "generated_latents_shape": list(generated_latents.shape),
        "reference_latents_shape": list(reference_latents.shape),
        "mask_latent_pixels": int(latent_mask.sum().item()),
        "metrics": rows,
        "outputs": {
            "metrics_csv": csv_path,
            "trajectory_mask_offpath_line": line_plot_path,
            "trajectory_reference_latents_npy": os.path.join(output_dir, "trajectory_reference_latents.npy"),
            "test_time_anchor_npy": os.path.join(output_dir, "test_time_anchor.npy"),
            "target_latent_npy": os.path.join(output_dir, f"{args.target_source}_target.npy"),
            "trajectory_l2_error_npy": os.path.join(output_dir, "trajectory_l2_error.npy"),
            "trajectory_offpath_error_npy": os.path.join(output_dir, "trajectory_offpath_error.npy"),
            "trajectory_progress_delta_npy": os.path.join(output_dir, "trajectory_progress_delta.npy"),
            "trajectory_l2_heatmaps": l2_heatmaps,
            "trajectory_l2_overlays": l2_overlays,
            "trajectory_l2_overlay_video": l2_video,
            "trajectory_offpath_heatmaps": offpath_heatmaps,
            "trajectory_offpath_overlays": offpath_overlays,
            "trajectory_offpath_overlay_video": offpath_video,
        },
    }
    summary_path = os.path.join(output_dir, "trajectory_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(summary_path)
    print(csv_path)
    print(l2_video)
    print(offpath_video)


if __name__ == "__main__":
    main()
