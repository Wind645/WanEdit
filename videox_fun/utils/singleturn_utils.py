import contextlib
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from videox_fun.pipeline.pipeline_wan import retrieve_timesteps
from videox_fun.utils.utils import save_videos_grid

DEFAULT_SINGLETURN_PROMPT_TEMPLATE = "Edit the source image according to this instruction: {prompt}"
SINGLETURN_TOTAL_FRAMES = 7
SINGLETURN_PREFIX_FRAMES = 4
SINGLETURN_TAIL_START = 4


def format_singleturn_prompt(prompt: str, prompt_template: str = DEFAULT_SINGLETURN_PROMPT_TEMPLATE) -> str:
    if "{prompt}" not in prompt_template:
        raise ValueError("prompt_template must contain the '{prompt}' placeholder.")
    return prompt_template.format(prompt="" if prompt is None else str(prompt))


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


def build_singleturn_image_transform(sample_size: int | Sequence[int]):
    sample_height, sample_width = normalize_singleturn_sample_size(sample_size)
    return transforms.Compose(
        [
            transforms.Resize(min(sample_height, sample_width)),
            transforms.CenterCrop((sample_height, sample_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


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


def _interp(start: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1.0 - alpha) * start + alpha * end


def _normalize_singleturn_source_latent(source_latent: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(source_latent):
        raise TypeError(f"source_latent must be a torch.Tensor, got {type(source_latent)}")
    if source_latent.ndim == 4:
        return source_latent.unsqueeze(0)
    if source_latent.ndim == 5:
        return source_latent
    raise ValueError(f"source_latent must have shape (B, C, 1, H, W) or (C, 1, H, W), got {tuple(source_latent.shape)}")


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


def build_singleturn_training_latents(
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    anchor_noise: torch.Tensor,
) -> torch.Tensor:
    _ensure_singleturn_latent_shape("source_latent", source_latent)
    _ensure_singleturn_latent_shape("target_latent", target_latent)
    _ensure_singleturn_latent_shape("anchor_noise", anchor_noise)
    _ensure_matching_shapes(source_latent, target_latent, "target_latent")
    _ensure_matching_shapes(source_latent, anchor_noise, "anchor_noise")

    frames = [
        source_latent,
        _interp(source_latent, anchor_noise, 1.0 / 3.0),
        _interp(source_latent, anchor_noise, 2.0 / 3.0),
        anchor_noise,
        _interp(anchor_noise, target_latent, 1.0 / 3.0),
        _interp(anchor_noise, target_latent, 2.0 / 3.0),
        target_latent,
    ]
    return torch.cat(frames, dim=2)


def build_singleturn_inference_latents(
    source_latent: torch.Tensor,
    anchor_noise: torch.Tensor,
    tail_noise: torch.Tensor,
) -> torch.Tensor:
    _ensure_singleturn_latent_shape("source_latent", source_latent)
    _ensure_singleturn_latent_shape("anchor_noise", anchor_noise)
    _ensure_matching_shapes(source_latent, anchor_noise, "anchor_noise")
    if tail_noise.ndim != 5:
        raise ValueError(f"tail_noise must have shape (B, C, T, H, W), got {tuple(tail_noise.shape)}")
    if tail_noise.shape[2] != 3:
        raise ValueError(f"tail_noise must contain exactly three latent frames, got {tail_noise.shape[2]}")
    if source_latent.shape[:2] + source_latent.shape[3:] != tail_noise.shape[:2] + tail_noise.shape[3:]:
        raise ValueError(
            "tail_noise must match source_latent batch/channel/spatial shape. "
            f"Got {tuple(source_latent.shape)} and {tuple(tail_noise.shape)}."
        )

    frames = [
        source_latent,
        _interp(source_latent, anchor_noise, 1.0 / 3.0),
        _interp(source_latent, anchor_noise, 2.0 / 3.0),
        anchor_noise,
        tail_noise[:, :, 0:1],
        tail_noise[:, :, 1:2],
        tail_noise[:, :, 2:3],
    ]
    return torch.cat(frames, dim=2)


def build_singleturn_loss_mask_like(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 5 or latents.shape[2] != SINGLETURN_TOTAL_FRAMES:
        raise ValueError(f"latents must have shape (B, C, 7, H, W), got {tuple(latents.shape)}")
    mask = torch.zeros_like(latents)
    mask[:, :, SINGLETURN_TAIL_START:] = 1
    return mask


def prepare_singleturn_noisy_latents(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if latents.shape != noise.shape:
        raise ValueError(f"latents and noise must share the same shape, got {tuple(latents.shape)} and {tuple(noise.shape)}")
    if latents.ndim != 5 or latents.shape[2] != SINGLETURN_TOTAL_FRAMES:
        raise ValueError(f"latents must have shape (B, C, 7, H, W), got {tuple(latents.shape)}")

    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    noisy_latents[:, :, :SINGLETURN_PREFIX_FRAMES] = latents[:, :, :SINGLETURN_PREFIX_FRAMES]
    target = noise - latents
    loss_mask = build_singleturn_loss_mask_like(latents)
    return noisy_latents, target, loss_mask


def zero_singleturn_prefix_prediction(model_pred: torch.Tensor) -> torch.Tensor:
    model_pred = model_pred.clone()
    model_pred[:, :, :SINGLETURN_PREFIX_FRAMES] = 0
    return model_pred


def restore_singleturn_prefix(updated_latents: torch.Tensor, frozen_prefix: torch.Tensor) -> torch.Tensor:
    updated_latents = updated_latents.clone()
    updated_latents[:, :, :SINGLETURN_PREFIX_FRAMES] = frozen_prefix
    return updated_latents


def compute_wan_seq_len_from_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> int:
    if latents.ndim != 5:
        raise ValueError(f"latents must have shape (B, C, T, H, W), got {tuple(latents.shape)}")
    _, _, num_frames, height, width = latents.shape
    return int(np.ceil((height * width) / (patch_size[1] * patch_size[2]) * num_frames))


def preprocess_singleturn_image(
    image_source,
    sample_size: int | Sequence[int],
    *,
    add_batch_dim: bool = True,
    add_frame_dim: bool = True,
) -> torch.Tensor:
    if isinstance(image_source, Image.Image):
        image = image_source.convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")
    tensor = build_singleturn_image_transform(sample_size)(image)
    if add_frame_dim:
        tensor = tensor.unsqueeze(0)
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)
    return tensor


def _run_singleturn_generation(
    pipeline,
    source_latent: torch.Tensor,
    prompt_context: List[torch.Tensor],
    negative_prompt_context: Optional[List[torch.Tensor]] = None,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    weight_dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 512,
):
    device = pipeline._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    weight_dtype = weight_dtype or getattr(pipeline.transformer, "dtype", torch.float32)
    source_latent = _normalize_singleturn_source_latent(source_latent).to(device=device, dtype=weight_dtype)
    prompt_context = [embed.to(device=device, dtype=weight_dtype) for embed in prompt_context]
    if not prompt_context:
        raise ValueError("prompt_context must not be empty.")
    if source_latent.shape[0] != len(prompt_context):
        raise ValueError(
            f"source_latent batch size {source_latent.shape[0]} does not match prompt_context size {len(prompt_context)}."
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

    anchor_noise = torch.randn(source_latent.size(), device=device, generator=generator, dtype=weight_dtype)
    tail_noise = torch.randn(
        (
            source_latent.shape[0],
            source_latent.shape[1],
            3,
            source_latent.shape[3],
            source_latent.shape[4],
        ),
        device=device,
        generator=generator,
        dtype=weight_dtype,
    )
    latents = build_singleturn_inference_latents(source_latent, anchor_noise, tail_noise)
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

        frozen_prefix = latents[:, :, :SINGLETURN_PREFIX_FRAMES].clone()
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
        "formatted_prompt": None,
        "full_frames": full_frames,
        "tail_frames": tail_frames,
        "latents": latents.detach().cpu(),
    }


def generate_singleturn_sample(
    pipeline,
    source_tensor: torch.Tensor,
    prompt: str,
    prompt_template: str = DEFAULT_SINGLETURN_PROMPT_TEMPLATE,
    negative_prompt: str = "",
    guidance_scale: float = 5.0,
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    weight_dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 512,
):
    device = pipeline._execution_device
    weight_dtype = weight_dtype or getattr(pipeline.transformer, "dtype", torch.float32)
    formatted_prompt = format_singleturn_prompt(prompt, prompt_template)

    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt=formatted_prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=guidance_scale > 1.0,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=weight_dtype,
    )
    generation = _run_singleturn_generation(
        pipeline=pipeline,
        source_latent=pipeline.vae.encode(source_tensor.permute(0, 2, 1, 3, 4))[0].mode(),
        prompt_context=prompt_embeds,
        negative_prompt_context=negative_prompt_embeds,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        weight_dtype=weight_dtype,
        max_sequence_length=max_sequence_length,
    )
    generation["formatted_prompt"] = formatted_prompt
    return generation


def generate_singleturn_sample_from_latents(
    pipeline,
    source_latent: torch.Tensor,
    prompt_embeds: torch.Tensor | List[torch.Tensor],
    prompt_seq_len: Optional[int | Sequence[int]] = None,
    negative_prompt: str = "",
    guidance_scale: float = 5.0,
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    weight_dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 512,
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
        source_latent=source_latent,
        prompt_context=prompt_context,
        negative_prompt_context=negative_prompt_context,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        weight_dtype=weight_dtype,
        max_sequence_length=max_sequence_length,
    )


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
    full_last = os.path.join(output_dir, f"{stem}_frame7.png")
    tail_last = os.path.join(output_dir, f"{stem}_tail_frame7.png")

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
