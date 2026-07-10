#!/usr/bin/env python

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm


FRAME_LABELS = [f"F{i}" for i in range(1, 9)]
TRANSITION_LABELS = [f"F{i}->F{i+1}" for i in range(1, 8)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze latent-trajectory statistics for SingleTurn refine cache samples."
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory containing refine cache .pt files.")
    parser.add_argument("--manifest_path", type=str, default=None, help="Optional refine manifest.json path.")
    parser.add_argument("--data_root", type=str, default=None, help="Optional root used with --manifest_path.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on the number of samples to scan.")
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=1,
        help="Only analyze every N-th sample from the sorted cache list / manifest.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Computation device, e.g. cpu or cuda:0.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to write JSON summary.")
    return parser.parse_args()


def summarize_vector(values: list[float]) -> dict:
    if not values:
        return {"count": 0}

    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "max": float(tensor.max().item()),
    }


def summarize_matrix(values: list[list[float]], labels: list[str]) -> dict:
    if not values:
        return {label: {"count": 0} for label in labels}

    tensor = torch.tensor(values, dtype=torch.float64)
    return {label: summarize_vector(tensor[:, idx].tolist()) for idx, label in enumerate(labels)}


def rms_per_frame(latents: torch.Tensor) -> list[float]:
    return latents.square().mean(dim=(0, 2, 3)).sqrt().tolist()


def mean_per_frame(latents: torch.Tensor) -> list[float]:
    return latents.mean(dim=(0, 2, 3)).tolist()


def std_per_frame(latents: torch.Tensor) -> list[float]:
    return latents.std(dim=(0, 2, 3), unbiased=False).tolist()


def transition_rms(latents: torch.Tensor) -> list[float]:
    return (latents[:, 1:] - latents[:, :-1]).square().mean(dim=(0, 2, 3)).sqrt().tolist()


def diff_rms_by_frame(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    return (a - b).square().mean(dim=(0, 2, 3)).sqrt().tolist()


def masked_transition_rms(latents: torch.Tensor, mask_2d: torch.Tensor) -> tuple[list[float], list[float]]:
    editable_vals = []
    background_vals = []
    editable_count = float(mask_2d.sum().item())
    total_count = float(mask_2d.numel())
    background_count = total_count - editable_count

    frame_diff = latents[:, 1:] - latents[:, :-1]
    for frame_idx in range(frame_diff.shape[1]):
        frame_sq = frame_diff[:, frame_idx].square().mean(dim=0)
        if editable_count > 0:
            editable_rms = math.sqrt(float(frame_sq[mask_2d].mean().item()))
        else:
            editable_rms = 0.0
        if background_count > 0:
            background_rms = math.sqrt(float(frame_sq[~mask_2d].mean().item()))
        else:
            background_rms = 0.0
        editable_vals.append(editable_rms)
        background_vals.append(background_rms)
    return editable_vals, background_vals


def masked_rms_by_frame(diff: torch.Tensor, mask_2d: torch.Tensor) -> tuple[list[float], list[float]]:
    editable_vals = []
    background_vals = []
    editable_count = float(mask_2d.sum().item())
    total_count = float(mask_2d.numel())
    background_count = total_count - editable_count

    for frame_idx in range(diff.shape[1]):
        frame_sq = diff[:, frame_idx].square().mean(dim=0)
        if editable_count > 0:
            editable_rms = math.sqrt(float(frame_sq[mask_2d].mean().item()))
        else:
            editable_rms = 0.0
        if background_count > 0:
            background_rms = math.sqrt(float(frame_sq[~mask_2d].mean().item()))
        else:
            background_rms = 0.0
        editable_vals.append(editable_rms)
        background_vals.append(background_rms)
    return editable_vals, background_vals


def distance_to_final_frame_rms(latents: torch.Tensor, mask_2d: torch.Tensor) -> tuple[list[float], list[float]]:
    editable_vals = []
    background_vals = []
    editable_count = float(mask_2d.sum().item())
    total_count = float(mask_2d.numel())
    background_count = total_count - editable_count

    final_frame = latents[:, -1:]
    diff = latents - final_frame
    for frame_idx in range(diff.shape[1]):
        frame_sq = diff[:, frame_idx].square().mean(dim=0)
        if editable_count > 0:
            editable_rms = math.sqrt(float(frame_sq[mask_2d].mean().item()))
        else:
            editable_rms = 0.0
        if background_count > 0:
            background_rms = math.sqrt(float(frame_sq[~mask_2d].mean().item()))
        else:
            background_rms = 0.0
        editable_vals.append(editable_rms)
        background_vals.append(background_rms)
    return editable_vals, background_vals


def interpolation_progress_metrics(
    latents: torch.Tensor,
    anchor_frame: torch.Tensor,
    final_frame: torch.Tensor,
    mask_2d: torch.Tensor,
) -> tuple[list[float], list[float]]:
    mask_flat = mask_2d.reshape(-1)
    if int(mask_flat.sum().item()) == 0:
        return [0.0] * latents.shape[1], [0.0] * latents.shape[1]

    anchor_masked = anchor_frame[:, mask_2d]
    final_masked = final_frame[:, mask_2d]
    basis = (final_masked - anchor_masked).reshape(-1)
    basis_norm_sq = float(torch.dot(basis, basis).item())
    if basis_norm_sq <= 1e-12:
        return [0.0] * latents.shape[1], [0.0] * latents.shape[1]

    alphas = []
    orthogonal_rms = []
    for frame_idx in range(latents.shape[1]):
        frame_masked = latents[:, frame_idx][:, mask_2d]
        vec = (frame_masked - anchor_masked).reshape(-1)
        alpha = float(torch.dot(vec, basis).item() / basis_norm_sq)
        residual = vec - alpha * basis
        alphas.append(alpha)
        orthogonal_rms.append(math.sqrt(float(residual.square().mean().item())))
    return alphas, orthogonal_rms


def resolve_cache_files(args) -> list[Path]:
    cache_files = []
    if args.cache_dir:
        cache_files.extend(sorted(Path(args.cache_dir).glob("*.pt")))
    elif args.manifest_path:
        manifest_path = Path(args.manifest_path)
        manifest = json.loads(manifest_path.read_text())
        data_root = Path(args.data_root) if args.data_root else manifest_path.parent
        for entry in manifest:
            cache_path = Path(entry["cache_path"])
            if not cache_path.is_absolute():
                cache_path = data_root / cache_path
            cache_files.append(cache_path)
    else:
        raise ValueError("Provide either --cache_dir or --manifest_path.")

    if args.sample_stride > 1:
        cache_files = cache_files[:: args.sample_stride]
    if args.max_samples is not None:
        cache_files = cache_files[: args.max_samples]
    if not cache_files:
        raise ValueError("No refine cache .pt files matched the requested inputs.")
    return cache_files


def main():
    args = parse_args()
    cache_files = resolve_cache_files(args)
    device = torch.device(args.device)

    input_frame_mean_stats = []
    input_frame_std_stats = []
    input_frame_rms_stats = []
    input_transition_stats = []
    target_transition_stats = []
    input_transition_editable_stats = []
    input_transition_background_stats = []
    target_transition_editable_stats = []
    target_transition_background_stats = []
    refine_delta_rms_stats = []
    refine_delta_editable_stats = []
    refine_delta_background_stats = []
    input_to_final_editable_stats = []
    input_to_final_background_stats = []
    target_to_final_editable_stats = []
    target_to_final_background_stats = []
    input_interp_alpha_stats = []
    input_interp_orthogonal_stats = []
    target_interp_alpha_stats = []
    target_interp_orthogonal_stats = []
    editable_fraction_stats = []
    edge_fraction_stats = []
    max_weight_stats = []
    per_sample_rankings = []
    encountered_weight_sets = {}

    latent_shape = None

    for path in tqdm(cache_files, desc="analyze_refine_cache"):
        payload = torch.load(path, map_location="cpu")
        input_latents = payload["input_latents"].float().to(device=device)
        target_latents = payload["target_latents"].float().to(device=device)
        weight_map = payload["refinement_loss_weight_map"].float().to(device=device)

        if input_latents.ndim != 4 or target_latents.ndim != 4:
            raise ValueError(f"Expected 4D CxTxHxW latents in {path}, got {tuple(input_latents.shape)} / {tuple(target_latents.shape)}")
        if weight_map.ndim != 4 or weight_map.shape[0] != 1:
            raise ValueError(f"Expected 4D weight_map with shape (1,T,H,W) in {path}, got {tuple(weight_map.shape)}")

        latent_shape = latent_shape or list(input_latents.shape)
        diff = target_latents - input_latents

        active_frame_weights = weight_map[0, 5]
        unique_weights = sorted(float(v.item()) for v in torch.unique(active_frame_weights))
        encountered_weight_sets[tuple(unique_weights)] = encountered_weight_sets.get(tuple(unique_weights), 0) + 1
        positive_weights = [weight for weight in unique_weights if weight > 0]
        background_weight = min(positive_weights) if positive_weights else 0.0
        max_weight = max(unique_weights) if unique_weights else 0.0

        editable_mask = active_frame_weights > (background_weight + 1e-6)
        edge_mask = active_frame_weights >= (max_weight - 1e-6)

        editable_fraction = float(editable_mask.float().mean().item())
        edge_fraction = float(edge_mask.float().mean().item())

        input_frame_mean = mean_per_frame(input_latents)
        input_frame_std = std_per_frame(input_latents)
        input_frame_rms = rms_per_frame(input_latents)
        input_transition = transition_rms(input_latents)
        target_transition = transition_rms(target_latents)
        input_transition_editable, input_transition_background = masked_transition_rms(input_latents, editable_mask)
        target_transition_editable, target_transition_background = masked_transition_rms(target_latents, editable_mask)
        refine_delta_rms = diff_rms_by_frame(target_latents, input_latents)
        refine_delta_editable, refine_delta_background = masked_rms_by_frame(diff, editable_mask)
        input_to_final_editable, input_to_final_background = distance_to_final_frame_rms(input_latents, editable_mask)
        target_to_final_editable, target_to_final_background = distance_to_final_frame_rms(target_latents, editable_mask)
        anchor_frame = target_latents[:, 4]
        final_frame = target_latents[:, 7]
        input_interp_alpha, input_interp_orthogonal = interpolation_progress_metrics(
            input_latents,
            anchor_frame,
            final_frame,
            editable_mask,
        )
        target_interp_alpha, target_interp_orthogonal = interpolation_progress_metrics(
            target_latents,
            anchor_frame,
            final_frame,
            editable_mask,
        )

        input_frame_mean_stats.append(input_frame_mean)
        input_frame_std_stats.append(input_frame_std)
        input_frame_rms_stats.append(input_frame_rms)
        input_transition_stats.append(input_transition)
        target_transition_stats.append(target_transition)
        input_transition_editable_stats.append(input_transition_editable)
        input_transition_background_stats.append(input_transition_background)
        target_transition_editable_stats.append(target_transition_editable)
        target_transition_background_stats.append(target_transition_background)
        refine_delta_rms_stats.append(refine_delta_rms)
        refine_delta_editable_stats.append(refine_delta_editable)
        refine_delta_background_stats.append(refine_delta_background)
        input_to_final_editable_stats.append(input_to_final_editable)
        input_to_final_background_stats.append(input_to_final_background)
        target_to_final_editable_stats.append(target_to_final_editable)
        target_to_final_background_stats.append(target_to_final_background)
        input_interp_alpha_stats.append(input_interp_alpha)
        input_interp_orthogonal_stats.append(input_interp_orthogonal)
        target_interp_alpha_stats.append(target_interp_alpha)
        target_interp_orthogonal_stats.append(target_interp_orthogonal)
        editable_fraction_stats.append(editable_fraction)
        edge_fraction_stats.append(edge_fraction)
        max_weight_stats.append(max_weight)

        per_sample_rankings.append(
            {
                "cache_path": str(path),
                "editable_fraction": editable_fraction,
                "edge_fraction": edge_fraction,
                "refine_delta_rms_F8": refine_delta_rms[7],
                "refine_delta_editable_F8": refine_delta_editable[7],
                "refine_delta_background_F8": refine_delta_background[7],
                "transition_input_F2_to_F3": input_transition[1],
                "transition_input_F5_to_F6": input_transition[4],
                "transition_input_F7_to_F8": input_transition[6],
                "input_interp_alpha_F6": input_interp_alpha[5],
                "input_interp_alpha_F7": input_interp_alpha[6],
                "input_interp_alpha_F8": input_interp_alpha[7],
                "input_interp_orthogonal_F8": input_interp_orthogonal[7],
            }
        )

    per_sample_rankings.sort(key=lambda item: item["refine_delta_editable_F8"], reverse=True)

    summary = {
        "num_samples": len(cache_files),
        "latent_shape_CxTxHxW": latent_shape,
        "frame_labels": FRAME_LABELS,
        "transition_labels": TRANSITION_LABELS,
        "encountered_weight_sets": {str(key): value for key, value in encountered_weight_sets.items()},
        "editable_fraction": summarize_vector(editable_fraction_stats),
        "edge_fraction": summarize_vector(edge_fraction_stats),
        "max_weight": summarize_vector(max_weight_stats),
        "input_frame_mean": summarize_matrix(input_frame_mean_stats, FRAME_LABELS),
        "input_frame_std": summarize_matrix(input_frame_std_stats, FRAME_LABELS),
        "input_frame_rms": summarize_matrix(input_frame_rms_stats, FRAME_LABELS),
        "input_transition_rms": summarize_matrix(input_transition_stats, TRANSITION_LABELS),
        "target_transition_rms": summarize_matrix(target_transition_stats, TRANSITION_LABELS),
        "input_transition_editable_rms": summarize_matrix(input_transition_editable_stats, TRANSITION_LABELS),
        "input_transition_background_rms": summarize_matrix(input_transition_background_stats, TRANSITION_LABELS),
        "target_transition_editable_rms": summarize_matrix(target_transition_editable_stats, TRANSITION_LABELS),
        "target_transition_background_rms": summarize_matrix(target_transition_background_stats, TRANSITION_LABELS),
        "refine_delta_rms": summarize_matrix(refine_delta_rms_stats, FRAME_LABELS),
        "refine_delta_editable_rms": summarize_matrix(refine_delta_editable_stats, FRAME_LABELS),
        "refine_delta_background_rms": summarize_matrix(refine_delta_background_stats, FRAME_LABELS),
        "input_to_final_editable_rms": summarize_matrix(input_to_final_editable_stats, FRAME_LABELS),
        "input_to_final_background_rms": summarize_matrix(input_to_final_background_stats, FRAME_LABELS),
        "target_to_final_editable_rms": summarize_matrix(target_to_final_editable_stats, FRAME_LABELS),
        "target_to_final_background_rms": summarize_matrix(target_to_final_background_stats, FRAME_LABELS),
        "input_interp_alpha": summarize_matrix(input_interp_alpha_stats, FRAME_LABELS),
        "input_interp_orthogonal_rms": summarize_matrix(input_interp_orthogonal_stats, FRAME_LABELS),
        "target_interp_alpha": summarize_matrix(target_interp_alpha_stats, FRAME_LABELS),
        "target_interp_orthogonal_rms": summarize_matrix(target_interp_orthogonal_stats, FRAME_LABELS),
        "top_samples_by_editable_F8_delta": per_sample_rankings[:20],
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
