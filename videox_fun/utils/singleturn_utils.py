import contextlib
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from videox_fun.pipeline.pipeline_wan import retrieve_timesteps
from videox_fun.utils.utils import save_videos_grid

CORNE_SINGLETURN_PROMPT = "Remove the masked object and its side effect"
SINGLETURN_TOTAL_FRAMES = 8
SINGLETURN_OBJECT_REMOVAL_DENSIFIED_TOTAL_FRAMES = 11
SINGLETURN_OBJECT_REMOVAL_MODE_TO_TOTAL_FRAMES = {
    "singleturn_object_removal_v2": 8,
    "singleturn_object_removal_v3_tail_interp11": 11,
}
SINGLETURN_FIRST_FRAME_FIXED_PREFIX_FRAMES = 2
SINGLETURN_REFINEMENT_FIXED_PREFIX_FRAMES = 5
SINGLETURN_TAIL_START = 2


def format_singleturn_prompt(prompt: Optional[str] = None, prompt_template: Optional[str] = None) -> str:
    del prompt
    del prompt_template
    return CORNE_SINGLETURN_PROMPT


def normalize_singleturn_sample_size(sample_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(sample_size, int):
        height = width = int(sample_size)
    else:
        values = [int(value) for value in sample_size]
        if len(values) == 1:
            height = width = values[0]
        elif len(values) == 2:
            height, width = values
        else:
            raise ValueError(f"sample_size must be an int or a sequence of length 1 or 2, got {sample_size}")

    if height <= 0 or width <= 0:
        raise ValueError(f"sample_size must be positive, got {(height, width)}")

    return height, width


def sample_latent_from_posterior_moments(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if mean.shape != logvar.shape:
        raise ValueError(f"mean and logvar must share the same shape, got {tuple(mean.shape)} and {tuple(logvar.shape)}")
    dtype = dtype or mean.dtype
    std = torch.exp(0.5 * logvar.to(dtype=torch.float32)).to(dtype=dtype)
    noise = torch.randn(mean.shape, device=mean.device, generator=generator, dtype=dtype)
    return mean.to(dtype=dtype) + std * noise


def _load_pil_image(image_source) -> Image.Image:
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    return Image.open(image_source).convert("RGB")


def _load_pil_mask(mask_source) -> Image.Image:
    if isinstance(mask_source, Image.Image):
        return mask_source.convert("L")
    return Image.open(mask_source).convert("L")


def _letterbox_geometry(width: int, height: int, sample_size: tuple[int, int]) -> tuple[int, int, int, int]:
    target_height, target_width = sample_size
    scale = min(target_width / float(width), target_height / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    return resized_width, resized_height, left, top


def _letterbox_rgb_tensor(image: Image.Image, sample_size: tuple[int, int]) -> torch.Tensor:
    target_height, target_width = sample_size
    resized_width, resized_height, left, top = _letterbox_geometry(image.width, image.height, sample_size)
    resized = image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _letterbox_mask_tensor(mask: Image.Image, sample_size: tuple[int, int]) -> torch.Tensor:
    target_height, target_width = sample_size
    resized_width, resized_height, left, top = _letterbox_geometry(mask.width, mask.height, sample_size)
    resized = mask.resize((resized_width, resized_height), resample=Image.NEAREST)
    canvas = Image.new("L", (target_width, target_height), 0)
    canvas.paste(resized, (left, top))
    array = (np.asarray(canvas, dtype=np.float32) / 255.0) >= 0.5
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(0).contiguous()


def _normalize_rgb_tensor(rgb_tensor: torch.Tensor) -> torch.Tensor:
    return rgb_tensor * 2.0 - 1.0


def _add_singleturn_dims(tensor: torch.Tensor, *, add_batch_dim: bool, add_frame_dim: bool) -> torch.Tensor:
    if add_frame_dim:
        tensor = tensor.unsqueeze(0)
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)
    return tensor


def preprocess_singleturn_image(
    image_source,
    sample_size: int | Sequence[int],
    *,
    add_batch_dim: bool = True,
    add_frame_dim: bool = True,
) -> torch.Tensor:
    sample_size = normalize_singleturn_sample_size(sample_size)
    rgb_tensor = _letterbox_rgb_tensor(_load_pil_image(image_source), sample_size)
    tensor = _normalize_rgb_tensor(rgb_tensor)
    return _add_singleturn_dims(tensor, add_batch_dim=add_batch_dim, add_frame_dim=add_frame_dim)


def preprocess_singleturn_mask(
    mask_source,
    sample_size: int | Sequence[int],
    *,
    add_batch_dim: bool = True,
    add_frame_dim: bool = True,
) -> torch.Tensor:
    sample_size = normalize_singleturn_sample_size(sample_size)
    mask_tensor = _letterbox_mask_tensor(_load_pil_mask(mask_source), sample_size)
    return _add_singleturn_dims(mask_tensor, add_batch_dim=add_batch_dim, add_frame_dim=add_frame_dim)


def preprocess_singleturn_mask_frame(
    mask_source,
    sample_size: int | Sequence[int],
    *,
    add_batch_dim: bool = True,
    add_frame_dim: bool = True,
) -> torch.Tensor:
    sample_size = normalize_singleturn_sample_size(sample_size)
    mask_tensor = _letterbox_mask_tensor(_load_pil_mask(mask_source), sample_size)
    rgb_tensor = mask_tensor.repeat(3, 1, 1)
    tensor = _normalize_rgb_tensor(rgb_tensor)
    return _add_singleturn_dims(tensor, add_batch_dim=add_batch_dim, add_frame_dim=add_frame_dim)


def _ensure_singleturn_latent_shape(name: str, latent: torch.Tensor) -> None:
    if latent.ndim != 5:
        raise ValueError(f"{name} must have shape (B, C, T, H, W), got {tuple(latent.shape)}")
    if latent.shape[2] != 1:
        raise ValueError(f"{name} must contain exactly one latent frame, got {latent.shape[2]}")


def _ensure_matching_shapes(reference: torch.Tensor, other: torch.Tensor, other_name: str) -> None:
    if reference.shape != other.shape:
        raise ValueError(
            f"Expected matching latent shapes, got {tuple(reference.shape)} and {tuple(other.shape)} for {other_name}."
        )


def _normalize_singleturn_prefix_latent(name: str, latent: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(latent):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(latent)}")
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    elif latent.ndim != 5:
        raise ValueError(f"{name} must have shape (B, C, 1, H, W) or (C, 1, H, W), got {tuple(latent.shape)}")
    _ensure_singleturn_latent_shape(name, latent)
    return latent


def _normalize_singleturn_prompt_context(
    prompt_embeds: torch.Tensor | List[torch.Tensor],
    prompt_seq_len: Optional[int | Sequence[int]] = None,
) -> List[torch.Tensor]:
    if isinstance(prompt_embeds, list):
        if prompt_seq_len is None:
            return [embed for embed in prompt_embeds]
        if isinstance(prompt_seq_len, int):
            seq_lens = [int(prompt_seq_len)] * len(prompt_embeds)
        else:
            seq_lens = [int(value) for value in prompt_seq_len]
            if len(seq_lens) != len(prompt_embeds):
                raise ValueError(
                    f"prompt_seq_len has batch size {len(seq_lens)}, but prompt_embeds has batch size {len(prompt_embeds)}."
                )
        return [embed[:seq_len] for embed, seq_len in zip(prompt_embeds, seq_lens)]

    if not torch.is_tensor(prompt_embeds):
        raise TypeError(f"prompt_embeds must be a tensor or list of tensors, got {type(prompt_embeds)}")

    if prompt_embeds.ndim == 2:
        seq_len = int(prompt_seq_len) if prompt_seq_len is not None else prompt_embeds.shape[0]
        return [prompt_embeds[:seq_len]]

    if prompt_embeds.ndim == 3:
        if prompt_seq_len is None:
            return [embed for embed in prompt_embeds]
        if isinstance(prompt_seq_len, int):
            seq_lens = [int(prompt_seq_len)] * prompt_embeds.shape[0]
        else:
            seq_lens = [int(value) for value in prompt_seq_len]
            if len(seq_lens) != prompt_embeds.shape[0]:
                raise ValueError(
                    f"prompt_seq_len has batch size {len(seq_lens)}, but prompt_embeds has batch size {prompt_embeds.shape[0]}."
                )
        return [embed[:seq_len] for embed, seq_len in zip(prompt_embeds, seq_lens)]

    raise ValueError(
        "prompt_embeds must have shape (seq_len, hidden_dim) or (B, seq_len, hidden_dim), "
        f"got {tuple(prompt_embeds.shape)}"
    )


def _interp(start: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1.0 - alpha) * start + alpha * end


def get_singleturn_object_removal_total_frames_for_mode(
    mode: Optional[str],
    *,
    fallback: Optional[int] = None,
) -> int:
    if mode in SINGLETURN_OBJECT_REMOVAL_MODE_TO_TOTAL_FRAMES:
        return SINGLETURN_OBJECT_REMOVAL_MODE_TO_TOTAL_FRAMES[mode]
    if fallback is not None:
        return int(fallback)
    raise ValueError(
        f"Unsupported SingleTurn object-removal mode {mode!r}. "
        f"Supported modes: {sorted(SINGLETURN_OBJECT_REMOVAL_MODE_TO_TOTAL_FRAMES)}"
    )


def is_supported_singleturn_object_removal_mode(mode: Optional[str]) -> bool:
    return mode in SINGLETURN_OBJECT_REMOVAL_MODE_TO_TOTAL_FRAMES


def resize_singleturn_mask_to_latent_grid(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 5:
        if mask.shape[2] != 1:
            raise ValueError(f"mask must contain exactly one frame, got {tuple(mask.shape)}")
        mask_2d = mask[:, :, 0]
    elif mask.ndim == 4:
        mask_2d = mask
    else:
        raise ValueError(f"mask must have shape (B, 1, H, W) or (B, 1, 1, H, W), got {tuple(mask.shape)}")

    if latent.ndim != 5 or latent.shape[2] != 1:
        raise ValueError(f"latent must have shape (B, C, 1, H, W), got {tuple(latent.shape)}")
    if mask_2d.shape[0] != latent.shape[0]:
        raise ValueError(f"mask batch size {mask_2d.shape[0]} does not match latent batch size {latent.shape[0]}")

    resized = F.interpolate(mask_2d.float(), size=latent.shape[-2:], mode="nearest")
    return (resized >= 0.5).to(dtype=latent.dtype).unsqueeze(2)


def build_singleturn_object_removal_latents(
    mask_frame_latent: torch.Tensor,
    source_frame_latent: torch.Tensor,
    bg_latent: torch.Tensor,
    mask_check_latent: torch.Tensor,
    noise_latent: Optional[torch.Tensor] = None,
    *,
    total_frames: int = SINGLETURN_TOTAL_FRAMES,
) -> torch.Tensor:
    _ensure_singleturn_latent_shape("mask_frame_latent", mask_frame_latent)
    _ensure_singleturn_latent_shape("source_frame_latent", source_frame_latent)
    _ensure_singleturn_latent_shape("bg_latent", bg_latent)
    _ensure_matching_shapes(source_frame_latent, mask_frame_latent, "mask_frame_latent")
    _ensure_matching_shapes(source_frame_latent, bg_latent, "bg_latent")

    if mask_check_latent.ndim != 5 or mask_check_latent.shape[1] != 1 or mask_check_latent.shape[2] != 1:
        raise ValueError(
            "mask_check_latent must have shape (B, 1, 1, H, W), "
            f"got {tuple(mask_check_latent.shape)}"
        )
    if mask_check_latent.shape[0] != source_frame_latent.shape[0] or mask_check_latent.shape[-2:] != source_frame_latent.shape[-2:]:
        raise ValueError(
            "mask_check_latent must match source_frame_latent batch/spatial size, "
            f"got {tuple(mask_check_latent.shape)} and {tuple(source_frame_latent.shape)}"
        )

    if noise_latent is None:
        noise_latent = torch.randn_like(source_frame_latent)
    else:
        _ensure_singleturn_latent_shape("noise_latent", noise_latent)
        _ensure_matching_shapes(source_frame_latent, noise_latent, "noise_latent")

    mask = mask_check_latent.to(dtype=torch.bool).expand_as(source_frame_latent)
    noisy_anchor = torch.where(mask, noise_latent, source_frame_latent)

    if total_frames < 8:
        raise ValueError(f"total_frames must be >= 8 for SingleTurn object removal, got {total_frames}")
    tail_interp_steps = total_frames - 5
    frames = [
        mask_frame_latent,
        source_frame_latent,
        _interp(source_frame_latent, noisy_anchor, 1.0 / 3.0),
        _interp(source_frame_latent, noisy_anchor, 2.0 / 3.0),
        noisy_anchor,
    ]
    for step_idx in range(1, tail_interp_steps + 1):
        frames.append(_interp(noisy_anchor, bg_latent, float(step_idx) / float(tail_interp_steps)))
    return torch.cat(frames, dim=2)


def build_singleturn_edge_weight_map(
    mask_check_latent: torch.Tensor,
    *,
    total_frames: int = SINGLETURN_TOTAL_FRAMES,
    prefix_frames: int = SINGLETURN_FIRST_FRAME_FIXED_PREFIX_FRAMES,
    peak_weight: float = 1.2,
    radius: int = 2,
) -> torch.Tensor:
    if mask_check_latent.ndim != 5 or mask_check_latent.shape[1] != 1 or mask_check_latent.shape[2] != 1:
        raise ValueError(
            "mask_check_latent must have shape (B, 1, 1, H, W), "
            f"got {tuple(mask_check_latent.shape)}"
        )
    if peak_weight < 1.0:
        raise ValueError(f"peak_weight must be >= 1.0, got {peak_weight}")
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    if prefix_frames < 0 or prefix_frames >= total_frames:
        raise ValueError(f"prefix_frames must be in [0, {total_frames - 1}], got {prefix_frames}")

    mask_2d = (mask_check_latent[:, :, 0] >= 0.5).float()
    dilated = F.max_pool2d(mask_2d, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask_2d, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded > 0).float()

    weights_2d = torch.ones_like(mask_2d)
    for distance in range(radius):
        if distance == 0:
            band = boundary
        else:
            band = F.max_pool2d(boundary, kernel_size=2 * distance + 1, stride=1, padding=distance)
        band_weight = 1.0 + (peak_weight - 1.0) * float(radius - distance) / float(radius)
        weights_2d = torch.maximum(weights_2d, torch.where(band > 0, torch.full_like(weights_2d, band_weight), torch.ones_like(weights_2d)))

    edge_weight_map = weights_2d.unsqueeze(2).repeat(1, 1, total_frames, 1, 1)
    edge_weight_map[:, :, :prefix_frames] = 0
    return edge_weight_map.to(dtype=mask_check_latent.dtype)


def build_singleturn_prefix_inference_latents(
    mask_frame_latent: torch.Tensor,
    source_frame_latent: torch.Tensor,
    tail_noise: torch.Tensor,
    *,
    total_frames: int = SINGLETURN_TOTAL_FRAMES,
) -> torch.Tensor:
    _ensure_singleturn_latent_shape("mask_frame_latent", mask_frame_latent)
    _ensure_singleturn_latent_shape("source_frame_latent", source_frame_latent)
    _ensure_matching_shapes(source_frame_latent, mask_frame_latent, "mask_frame_latent")
    if tail_noise.ndim != 5:
        raise ValueError(f"tail_noise must have shape (B, C, T, H, W), got {tuple(tail_noise.shape)}")
    expected_tail_frames = total_frames - SINGLETURN_FIRST_FRAME_FIXED_PREFIX_FRAMES
    if tail_noise.shape[2] != expected_tail_frames:
        raise ValueError(
            "tail_noise must contain exactly "
            f"{expected_tail_frames} latent frames, "
            f"got {tail_noise.shape[2]}"
        )
    if source_frame_latent.shape[:2] + source_frame_latent.shape[3:] != tail_noise.shape[:2] + tail_noise.shape[3:]:
        raise ValueError(
            "tail_noise must match source_frame_latent batch/channel/spatial shape. "
            f"Got {tuple(source_frame_latent.shape)} and {tuple(tail_noise.shape)}."
        )
    return torch.cat([mask_frame_latent, source_frame_latent, tail_noise], dim=2)


def build_singleturn_loss_mask_like(
    latents: torch.Tensor,
    *,
    prefix_frames: int = SINGLETURN_FIRST_FRAME_FIXED_PREFIX_FRAMES,
) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError(f"latents must have shape (B, C, T, H, W), got {tuple(latents.shape)}")
    total_frames = latents.shape[2]
    if prefix_frames <= 0 or prefix_frames >= total_frames:
        raise ValueError(f"prefix_frames must be in [1, {total_frames - 1}], got {prefix_frames}")
    mask = torch.ones_like(latents)
    mask[:, :, :prefix_frames] = 0
    return mask


def build_singleturn_refinement_target_latents(
    coarse_latents: torch.Tensor,
    gt_last_latent: torch.Tensor,
    *,
    prefix_frames: int = SINGLETURN_REFINEMENT_FIXED_PREFIX_FRAMES,
) -> torch.Tensor:
    if coarse_latents.ndim != 5 or coarse_latents.shape[2] != SINGLETURN_TOTAL_FRAMES:
        raise ValueError(
            f"coarse_latents must have shape (B, C, {SINGLETURN_TOTAL_FRAMES}, H, W), got {tuple(coarse_latents.shape)}"
        )
    gt_last_latent = _normalize_singleturn_prefix_latent("gt_last_latent", gt_last_latent)
    if coarse_latents.shape[0] != gt_last_latent.shape[0]:
        raise ValueError(
            "coarse_latents batch size does not match gt_last_latent batch size: "
            f"{coarse_latents.shape[0]} vs {gt_last_latent.shape[0]}"
        )
    if coarse_latents.shape[1] != gt_last_latent.shape[1] or coarse_latents.shape[-2:] != gt_last_latent.shape[-2:]:
        raise ValueError(
            "coarse_latents and gt_last_latent must share channel and spatial shape. "
            f"Got {tuple(coarse_latents.shape)} and {tuple(gt_last_latent.shape)}."
        )
    if prefix_frames <= 0 or prefix_frames >= SINGLETURN_TOTAL_FRAMES:
        raise ValueError(f"prefix_frames must be in [1, {SINGLETURN_TOTAL_FRAMES - 1}], got {prefix_frames}")

    target_latents = coarse_latents.clone()
    anchor = coarse_latents[:, :, prefix_frames - 1 : prefix_frames]
    target_latents[:, :, prefix_frames : prefix_frames + 1] = _interp(anchor, gt_last_latent, 1.0 / 3.0)
    target_latents[:, :, prefix_frames + 1 : prefix_frames + 2] = _interp(anchor, gt_last_latent, 2.0 / 3.0)
    target_latents[:, :, prefix_frames + 2 : prefix_frames + 3] = gt_last_latent
    return target_latents


def build_singleturn_refinement_weight_map(
    mask_check_latent: torch.Tensor,
    *,
    total_frames: int = SINGLETURN_TOTAL_FRAMES,
    prefix_frames: int = SINGLETURN_REFINEMENT_FIXED_PREFIX_FRAMES,
    background_weight: float = 0.1,
    mask_weight: float = 3.0,
    edge_weight: float = 5.0,
    edge_radius: int = 2,
) -> torch.Tensor:
    if mask_check_latent.ndim != 5 or mask_check_latent.shape[1] != 1 or mask_check_latent.shape[2] != 1:
        raise ValueError(
            "mask_check_latent must have shape (B, 1, 1, H, W), "
            f"got {tuple(mask_check_latent.shape)}"
        )
    if edge_radius < 1:
        raise ValueError(f"edge_radius must be >= 1, got {edge_radius}")
    if background_weight < 0.0:
        raise ValueError(f"background_weight must be >= 0.0, got {background_weight}")
    if edge_weight < mask_weight or mask_weight < background_weight:
        raise ValueError(
            "Expected edge_weight >= mask_weight >= background_weight, "
            f"got edge_weight={edge_weight}, mask_weight={mask_weight}, background_weight={background_weight}"
        )

    mask_2d = (mask_check_latent[:, :, 0] >= 0.5).float()
    dilated = F.max_pool2d(mask_2d, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask_2d, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded > 0).float()

    weights_2d = torch.full_like(mask_2d, background_weight)
    weights_2d = torch.where(mask_2d > 0, torch.full_like(weights_2d, mask_weight), weights_2d)

    expanded_boundary = boundary
    for distance in range(1, edge_radius):
        expanded_boundary = torch.maximum(
            expanded_boundary,
            F.max_pool2d(boundary, kernel_size=2 * distance + 1, stride=1, padding=distance),
        )
    weights_2d = torch.where(expanded_boundary > 0, torch.full_like(weights_2d, edge_weight), weights_2d)

    weight_map = weights_2d.unsqueeze(2).repeat(1, 1, total_frames, 1, 1)
    weight_map[:, :, :prefix_frames] = 0
    return weight_map.to(dtype=mask_check_latent.dtype)


def prepare_singleturn_noisy_latents(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    prefix_frames: int = SINGLETURN_FIRST_FRAME_FIXED_PREFIX_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if latents.shape != noise.shape:
        raise ValueError(f"latents and noise must share the same shape, got {tuple(latents.shape)} and {tuple(noise.shape)}")
    if latents.ndim != 5:
        raise ValueError(f"latents must have shape (B, C, T, H, W), got {tuple(latents.shape)}")

    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    noisy_latents[:, :, :prefix_frames] = latents[:, :, :prefix_frames]
    target = noise - latents
    loss_mask = build_singleturn_loss_mask_like(latents, prefix_frames=prefix_frames)
    return noisy_latents, target, loss_mask


def zero_singleturn_prefix_prediction(
    model_pred: torch.Tensor,
    *,
    prefix_frames: int = SINGLETURN_FIRST_FRAME_FIXED_PREFIX_FRAMES,
) -> torch.Tensor:
    model_pred = model_pred.clone()
    model_pred[:, :, :prefix_frames] = 0
    return model_pred


def restore_singleturn_prefix(
    updated_latents: torch.Tensor,
    frozen_prefix: torch.Tensor,
    *,
    prefix_frames: int = SINGLETURN_FIRST_FRAME_FIXED_PREFIX_FRAMES,
) -> torch.Tensor:
    updated_latents = updated_latents.clone()
    updated_latents[:, :, :prefix_frames] = frozen_prefix
    return updated_latents


def apply_singleturn_refinement_prediction(
    coarse_latents: torch.Tensor,
    refinement_pred: torch.Tensor,
    *,
    prefix_frames: int = SINGLETURN_REFINEMENT_FIXED_PREFIX_FRAMES,
) -> torch.Tensor:
    if coarse_latents.shape != refinement_pred.shape:
        raise ValueError(
            f"coarse_latents and refinement_pred must share the same shape, got "
            f"{tuple(coarse_latents.shape)} and {tuple(refinement_pred.shape)}"
        )
    frozen_prefix = coarse_latents[:, :, :prefix_frames].clone()
    refinement_pred = zero_singleturn_prefix_prediction(refinement_pred, prefix_frames=prefix_frames)
    refined_latents = coarse_latents + refinement_pred
    return restore_singleturn_prefix(refined_latents, frozen_prefix, prefix_frames=prefix_frames)


def compute_wan_seq_len_from_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> int:
    if latents.ndim != 5:
        raise ValueError(f"latents must have shape (B, C, T, H, W), got {tuple(latents.shape)}")
    _, _, num_frames, height, width = latents.shape
    return int(np.ceil((height * width) / (patch_size[1] * patch_size[2]) * num_frames))


def _run_singleturn_generation(
    pipeline,
    mask_frame_latent: torch.Tensor,
    source_frame_latent: torch.Tensor,
    prompt_context: List[torch.Tensor],
    negative_prompt_context: Optional[List[torch.Tensor]] = None,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    weight_dtype: Optional[torch.dtype] = None,
    total_frames: int = SINGLETURN_TOTAL_FRAMES,
):
    device = pipeline._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    weight_dtype = weight_dtype or getattr(pipeline.transformer, "dtype", torch.float32)
    mask_frame_latent = _normalize_singleturn_prefix_latent("mask_frame_latent", mask_frame_latent).to(
        device=device,
        dtype=weight_dtype,
    )
    source_frame_latent = _normalize_singleturn_prefix_latent("source_frame_latent", source_frame_latent).to(
        device=device,
        dtype=weight_dtype,
    )
    _ensure_matching_shapes(source_frame_latent, mask_frame_latent, "mask_frame_latent")
    prompt_context = [embed.to(device=device, dtype=weight_dtype) for embed in prompt_context]
    if not prompt_context:
        raise ValueError("prompt_context must not be empty.")
    if source_frame_latent.shape[0] != len(prompt_context):
        raise ValueError(
            "source_frame_latent batch size "
            f"{source_frame_latent.shape[0]} does not match prompt_context size {len(prompt_context)}."
        )

    if do_classifier_free_guidance:
        if negative_prompt_context is None:
            raise ValueError("negative_prompt_context is required when guidance_scale > 1.0.")
        negative_prompt_context = [embed.to(device=device, dtype=weight_dtype) for embed in negative_prompt_context]
        if len(negative_prompt_context) != len(prompt_context):
            raise ValueError(
                "negative_prompt_context must have the same batch size as prompt_context when guidance is enabled."
            )
        context = negative_prompt_context + prompt_context
    else:
        context = prompt_context

    timesteps, _ = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device=device,
        mu=1,
    )
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta=0.0)
    tail_noise = torch.randn(
        (
            source_frame_latent.shape[0],
            source_frame_latent.shape[1],
            total_frames - SINGLETURN_FIRST_FRAME_FIXED_PREFIX_FRAMES,
            source_frame_latent.shape[3],
            source_frame_latent.shape[4],
        ),
        device=device,
        generator=generator,
        dtype=weight_dtype,
    )
    latents = build_singleturn_prefix_inference_latents(
        mask_frame_latent,
        source_frame_latent,
        tail_noise,
        total_frames=total_frames,
    )
    seq_len = compute_wan_seq_len_from_latents(latents, pipeline.transformer.config.patch_size)

    def autocast_context():
        if device.type == "cuda" and weight_dtype != torch.float32:
            return torch.autocast("cuda", dtype=weight_dtype)
        return contextlib.nullcontext()

    for timestep in timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        if hasattr(pipeline.scheduler, "scale_model_input"):
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timestep)

        with autocast_context():
            noise_pred = pipeline.transformer(
                x=latent_model_input,
                context=context,
                t=timestep.expand(latent_model_input.shape[0]),
                seq_len=seq_len,
            )

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        frozen_prefix = latents[:, :, :SINGLETURN_FIRST_FRAME_FIXED_PREFIX_FRAMES].clone()
        noise_pred = zero_singleturn_prefix_prediction(noise_pred)
        latents = pipeline.scheduler.step(
            noise_pred,
            timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
        )[0]
        latents = restore_singleturn_prefix(latents, frozen_prefix)

    full_frames = decode_singleturn_latent_frames(pipeline.vae, latents, decode_dtype=weight_dtype).cpu()
    tail_frames = full_frames[:, :, SINGLETURN_TAIL_START:].contiguous()
    return {
        "formatted_prompt": CORNE_SINGLETURN_PROMPT,
        "full_frames": full_frames,
        "tail_frames": tail_frames,
        "latents": latents.detach().cpu(),
    }


def generate_singleturn_sample(
    pipeline,
    mask_frame_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    prompt: Optional[str] = None,
    prompt_template: Optional[str] = None,
    negative_prompt: str = "",
    guidance_scale: float = 5.0,
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    weight_dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 512,
    total_frames: int = SINGLETURN_OBJECT_REMOVAL_DENSIFIED_TOTAL_FRAMES,
):
    del prompt
    del prompt_template
    device = pipeline._execution_device
    weight_dtype = weight_dtype or getattr(pipeline.transformer, "dtype", torch.float32)

    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt=CORNE_SINGLETURN_PROMPT,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=guidance_scale > 1.0,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=weight_dtype,
    )
    return _run_singleturn_generation(
        pipeline=pipeline,
        mask_frame_latent=pipeline.vae.encode(mask_frame_tensor.permute(0, 2, 1, 3, 4))[0].mode(),
        source_frame_latent=pipeline.vae.encode(source_tensor.permute(0, 2, 1, 3, 4))[0].mode(),
        prompt_context=prompt_embeds,
        negative_prompt_context=negative_prompt_embeds,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        weight_dtype=weight_dtype,
        total_frames=total_frames,
    )


def generate_singleturn_sample_from_latents(
    pipeline,
    mask_frame_latent: torch.Tensor,
    source_frame_latent: torch.Tensor,
    prompt_embeds: torch.Tensor | List[torch.Tensor],
    prompt_seq_len: Optional[int | Sequence[int]] = None,
    negative_prompt: str = "",
    guidance_scale: float = 5.0,
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    weight_dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 512,
    total_frames: int = SINGLETURN_TOTAL_FRAMES,
):
    device = pipeline._execution_device
    weight_dtype = weight_dtype or getattr(pipeline.transformer, "dtype", torch.float32)
    prompt_context = _normalize_singleturn_prompt_context(prompt_embeds, prompt_seq_len)
    negative_prompt_context = None
    if guidance_scale > 1.0:
        negative_prompt_context = pipeline._get_t5_prompt_embeds(
            prompt=[negative_prompt or ""] * len(prompt_context),
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=weight_dtype,
        )
    return _run_singleturn_generation(
        pipeline=pipeline,
        mask_frame_latent=mask_frame_latent,
        source_frame_latent=source_frame_latent,
        prompt_context=prompt_context,
        negative_prompt_context=negative_prompt_context,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        weight_dtype=weight_dtype,
        total_frames=total_frames,
    )


def refine_singleturn_sample_from_latents(
    pipeline,
    coarse_latents: torch.Tensor,
    prompt_embeds: torch.Tensor | List[torch.Tensor],
    prompt_seq_len: Optional[int | Sequence[int]] = None,
    negative_prompt: str = "",
    guidance_scale: float = 5.0,
    weight_dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 512,
    prefix_frames: int = SINGLETURN_REFINEMENT_FIXED_PREFIX_FRAMES,
):
    device = pipeline._execution_device
    weight_dtype = weight_dtype or getattr(pipeline.transformer, "dtype", torch.float32)
    if coarse_latents.ndim == 4:
        coarse_latents = coarse_latents.unsqueeze(0)
    if coarse_latents.ndim != 5 or coarse_latents.shape[2] != SINGLETURN_TOTAL_FRAMES:
        raise ValueError(
            f"coarse_latents must have shape (B, C, {SINGLETURN_TOTAL_FRAMES}, H, W), got {tuple(coarse_latents.shape)}"
        )

    coarse_latents = coarse_latents.to(device=device, dtype=weight_dtype)
    prompt_context = [embed.to(device=device, dtype=weight_dtype) for embed in _normalize_singleturn_prompt_context(prompt_embeds, prompt_seq_len)]
    negative_prompt_context = None
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        negative_prompt_context = pipeline._get_t5_prompt_embeds(
            prompt=[negative_prompt or ""] * len(prompt_context),
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=weight_dtype,
        )
        context = negative_prompt_context + prompt_context
    else:
        context = prompt_context

    seq_len = compute_wan_seq_len_from_latents(coarse_latents, pipeline.transformer.config.patch_size)
    refinement_timestep = torch.zeros((coarse_latents.shape[0],), device=device, dtype=weight_dtype)

    def autocast_context():
        if device.type == "cuda" and weight_dtype != torch.float32:
            return torch.autocast("cuda", dtype=weight_dtype)
        return contextlib.nullcontext()

    latent_model_input = torch.cat([coarse_latents] * 2) if do_classifier_free_guidance else coarse_latents
    refinement_timestep = refinement_timestep.expand(latent_model_input.shape[0])

    with autocast_context():
        refinement_pred = pipeline.transformer(
            x=latent_model_input,
            context=context,
            t=refinement_timestep,
            seq_len=seq_len,
        )

    if do_classifier_free_guidance:
        refinement_pred_uncond, refinement_pred_text = refinement_pred.chunk(2)
        refinement_pred = refinement_pred_uncond + guidance_scale * (refinement_pred_text - refinement_pred_uncond)

    refined_latents = apply_singleturn_refinement_prediction(
        coarse_latents,
        refinement_pred,
        prefix_frames=prefix_frames,
    )
    full_frames = decode_singleturn_latent_frames(pipeline.vae, refined_latents, decode_dtype=weight_dtype).cpu()
    tail_frames = full_frames[:, :, SINGLETURN_TAIL_START:].contiguous()
    return {
        "formatted_prompt": CORNE_SINGLETURN_PROMPT,
        "full_frames": full_frames,
        "tail_frames": tail_frames,
        "latents": refined_latents.detach().cpu(),
    }


def decode_singleturn_latent_frames(vae, latents: torch.Tensor, decode_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError(f"latents must have shape (B, C, T, H, W), got {tuple(latents.shape)}")

    decoded_frames = []
    try:
        decode_dtype = next(vae.parameters()).dtype
    except StopIteration:
        decode_dtype = decode_dtype or getattr(vae, "dtype", torch.float32)
    for frame_idx in range(latents.shape[2]):
        decoded = vae.decode(latents[:, :, frame_idx : frame_idx + 1].to(decode_dtype)).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded_frames.append(decoded)
    return torch.cat(decoded_frames, dim=2)


def save_singleturn_outputs(
    full_frames: torch.Tensor,
    tail_frames: torch.Tensor,
    output_dir: str,
    stem: str = "singleturn",
    fps: int = 4,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    full_frames = full_frames.detach().cpu().float()
    tail_frames = tail_frames.detach().cpu().float()

    full_gif = os.path.join(output_dir, f"{stem}_full.gif")
    tail_gif = os.path.join(output_dir, f"{stem}_tail.gif")
    full_last = os.path.join(output_dir, f"{stem}_frame8.png")
    tail_last = os.path.join(output_dir, f"{stem}_tail_frame8.png")

    save_videos_grid(full_frames, full_gif, fps=fps)
    save_videos_grid(tail_frames, tail_gif, fps=fps)

    full_last_frame = (full_frames[0, :, -1].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    tail_last_frame = (tail_frames[0, :, -1].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(full_last_frame).save(full_last)
    Image.fromarray(tail_last_frame).save(tail_last)

    return {
        "full_gif": full_gif,
        "tail_gif": tail_gif,
        "full_last_frame": full_last,
        "tail_last_frame": tail_last,
    }
