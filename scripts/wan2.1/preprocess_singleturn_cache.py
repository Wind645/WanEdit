#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoTokenizer
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

from videox_fun.data.singleturn_dataset import SingleTurnPreprocessIterableDataset
from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel
from videox_fun.utils.singleturn_utils import (
    CORNE_SINGLETURN_PROMPT,
    SINGLETURN_FIRST_FRAME_FIXED_PREFIX_FRAMES,
    SINGLETURN_TOTAL_FRAMES,
    normalize_singleturn_sample_size,
    preprocess_singleturn_mask,
    preprocess_singleturn_mask_frame,
    resize_singleturn_mask_to_latent_grid,
)


def resolve_model_path(model_root, subpath, default_subpath):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute SingleTurn 8-frame two-prefix object-removal cache.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Base Wan model path.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help=(
            "Object-removal root. Supports CORNE shot/, bg/, mask-check/, optional mask_sam/; "
            "or ObjectClear subset dirs with input/, gt/, object_effect_mask/, optional object_mask/."
        ),
    )
    parser.add_argument("--train_data_manifest", type=str, default=None, help="Deprecated and unsupported for CORNE object-removal mode.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write cached tensors and manifest.")
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Wan config path.")
    parser.add_argument("--video_sample_size", type=int, default=512, help="Deprecated square sample size shortcut.")
    parser.add_argument("--singleturn_sample_size", type=int, nargs=2, default=(480, 832), metavar=("HEIGHT", "WIDTH"), help="SingleTurn sample size used before VAE encode.")
    parser.add_argument("--tokenizer_max_length", type=int, default=512, help="Tokenizer max length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples to preprocess per batch.")
    parser.add_argument("--num_workers", type=int, default=0, help="Must be 0 for quota-preserving sequential traversal.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Cache tensor dtype.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    parser.add_argument("--max_samples_with_mask_sam", type=int, default=30000, help="Quota for samples whose mask-latent first frame comes from mask_sam.")
    parser.add_argument("--max_samples_without_mask_sam", type=int, default=30000, help="Quota for samples whose mask-latent first frame falls back to mask-check.")
    parser.add_argument("--skip_samples_with_mask_sam", type=int, default=0, help="Skip this many with_mask_sam samples before collecting the shard.")
    parser.add_argument("--skip_samples_without_mask_sam", type=int, default=0, help="Skip this many without_mask_sam samples before collecting the shard.")
    args = parser.parse_args()

    if args.train_data_manifest is not None:
        raise ValueError("SingleTurn object-removal preprocess no longer supports --train_data_manifest.")
    if args.num_workers != 0:
        raise ValueError("SingleTurn object-removal preprocess requires --num_workers 0 to preserve class quotas exactly.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if (
        args.max_samples_with_mask_sam < 0
        or args.max_samples_without_mask_sam < 0
        or args.skip_samples_with_mask_sam < 0
        or args.skip_samples_without_mask_sam < 0
    ):
        raise ValueError("Quota and skip arguments must be non-negative.")

    sample_size = normalize_singleturn_sample_size(
        args.singleturn_sample_size if args.singleturn_sample_size is not None else args.video_sample_size
    )
    if any(dim % 16 != 0 for dim in sample_size):
        raise ValueError(f"SingleTurn sample size must be divisible by 16, got {sample_size}.")
    args.singleturn_sample_size = sample_size
    return args


def get_weight_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16


def build_cache_variant_name(global_index: int, source_image: str, variant: str) -> str:
    base = f"{int(global_index):06d}_{Path(str(source_image)).stem}"
    if variant and variant != "normal":
        return f"{base}__{variant}.pt"
    return f"{base}.pt"


def build_generated_mask_path(output_dir: Path, global_index: int, source_image: str, variant: str) -> Path:
    return output_dir / "generated_masks" / variant / f"{int(global_index):06d}_{Path(str(source_image)).stem}.png"


def _load_grayscale_mask(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def _polygon_union_mask(mask_size: tuple[int, int], points: list[tuple[float, float]], fg: np.ndarray) -> Image.Image:
    coarse = Image.new("L", mask_size, 0)
    ImageDraw.Draw(coarse).polygon(points, outline=255, fill=255)
    return Image.fromarray(np.maximum(np.asarray(coarse, dtype=np.uint8), fg.astype(np.uint8) * 255), mode="L")


def _generate_coarse_mask(mask_image: Image.Image, seed: int, *, max_attempts: int = 16) -> Image.Image | None:
    mask = mask_image.convert("L")
    mask_arr = np.asarray(mask, dtype=np.uint8)
    fg = mask_arr > 127
    if not fg.any():
        return None

    rng = np.random.default_rng(int(seed))
    ys, xs = np.where(fg)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    source_area = max(1, int(fg.sum()))
    max_area_ratio = 2.5
    min_area_ratio = 1.15
    for _ in range(max(1, int(max_attempts))):
        expand_x = float(rng.uniform(1.05, 1.28))
        expand_y = float(rng.uniform(1.05, 1.28))
        rx = max(4.0, 0.5 * w * expand_x)
        ry = max(4.0, 0.5 * h * expand_y)
        num_vertices = int(rng.integers(7, 11))
        angle_offset = float(rng.uniform(0.0, 2.0 * np.pi))
        points = []
        for vertex_idx in range(num_vertices):
            theta = angle_offset + (2.0 * np.pi * vertex_idx / float(num_vertices)) + float(rng.uniform(-0.1, 0.1))
            radial_scale = float(rng.uniform(0.92, 1.08))
            px = cx + np.cos(theta) * rx * radial_scale
            py = cy + np.sin(theta) * ry * radial_scale
            points.append((px, py))

        coarse = _polygon_union_mask(mask.size, points, fg)
        coarse_area = int((np.asarray(coarse) > 127).sum())
        area_ratio = coarse_area / float(source_area)
        if min_area_ratio <= area_ratio <= max_area_ratio:
            return coarse
    return None


def build_manifest_entry(cache_path: Path, output_dir: Path, sample: dict) -> dict:
    entry = {
        "cache_path": str(cache_path.relative_to(output_dir)),
        "mode": "singleturn_object_removal_v2",
        "dataset_type": sample.get("dataset_type", "corne_object_removal"),
        "variant": sample.get("variant", "normal"),
        "source_image": sample["source_image"],
        "bg_image": sample["bg_image"],
        "mask_check_image": sample["mask_check_image"],
        "mask_frame_image": sample["mask_frame_image"],
        "used_mask_sam": bool(sample["used_mask_sam"]),
        "global_index": int(sample["global_index"]),
    }
    subset = sample.get("subset", "")
    if subset:
        entry["subset"] = subset
    mask_sam_image = sample.get("mask_sam_image", "")
    if mask_sam_image:
        entry["mask_sam_image"] = mask_sam_image
    generated_mask_image = sample.get("generated_mask_image", "")
    if generated_mask_image:
        entry["generated_mask_image"] = generated_mask_image
    return entry


def _stack_batch_tensors(batch_records: list[dict], key: str) -> torch.Tensor:
    return torch.stack([record[key] for record in batch_records], dim=0)


def _flush_batch(
    *,
    batch_records: list[dict],
    device: torch.device,
    weight_dtype: torch.dtype,
    vae,
    cache_dir: Path,
    output_dir: Path,
    shared_prompt_path: Path,
    singleturn_sample_size: tuple[int, int],
    manifest: list[dict],
    overwrite: bool,
) -> int:
    if not batch_records:
        return 0

    source_batch = _stack_batch_tensors(batch_records, "pixel_values_src_image").to(
        device=device,
        dtype=weight_dtype,
        non_blocking=True,
    )
    mask_frame_batch = _stack_batch_tensors(batch_records, "pixel_values_mask_frame").to(
        device=device,
        dtype=weight_dtype,
        non_blocking=True,
    )
    mask_check_frame_batch = _stack_batch_tensors(batch_records, "pixel_values_mask_check_frame").to(
        device=device,
        dtype=weight_dtype,
        non_blocking=True,
    )
    bg_batch = _stack_batch_tensors(batch_records, "pixel_values_tgt_image").to(
        device=device,
        dtype=weight_dtype,
        non_blocking=True,
    )
    mask_check_batch = _stack_batch_tensors(batch_records, "pixel_values_mask_check").to(
        device=device,
        dtype=weight_dtype,
        non_blocking=True,
    )

    with torch.no_grad():
        mask_frame_latents = vae.encode(mask_frame_batch.permute(0, 2, 1, 3, 4))[0].mode()
        mask_check_frame_latents = vae.encode(mask_check_frame_batch.permute(0, 2, 1, 3, 4))[0].mode()
        source_frame_latents = vae.encode(source_batch.permute(0, 2, 1, 3, 4))[0].mode()
        bg_latents = vae.encode(bg_batch.permute(0, 2, 1, 3, 4))[0].mode()
        latent_mask = resize_singleturn_mask_to_latent_grid(mask_check_batch, source_frame_latents)
        noise_latents = torch.randn_like(source_frame_latents)
        noisy_anchor_latents = torch.where(
            latent_mask.to(dtype=torch.bool).expand_as(source_frame_latents),
            noise_latents,
            source_frame_latents,
        )

    coarse_variant_enabled = any(record.get("dataset_type") == "objectclear_object_removal" for record in batch_records)
    coarse_mask_frame_batch = None
    coarse_mask_latents = None
    coarse_noisy_anchor_latents = None
    coarse_mask_paths: list[str | None] = []
    coarse_valid_offsets: list[int] = []
    if coarse_variant_enabled:
        coarse_mask_frame_tensors = []
        coarse_mask_binary_tensors = []
        for local_offset, record in enumerate(batch_records):
            global_index = int(record["global_index"])
            seed = global_index
            coarse_path = build_generated_mask_path(output_dir, global_index, record["source_image"], "coarse")
            coarse_path.parent.mkdir(parents=True, exist_ok=True)
            coarse_mask = _generate_coarse_mask(
                _load_grayscale_mask(record["mask_check_image"]),
                seed=seed,
            )
            if coarse_mask is None:
                coarse_mask_paths.append(None)
                continue
            coarse_mask.save(coarse_path)
            coarse_mask_paths.append(str(coarse_path))
            coarse_valid_offsets.append(local_offset)
            coarse_mask_frame_tensors.append(
                preprocess_singleturn_mask_frame(
                    coarse_mask,
                    singleturn_sample_size,
                    add_batch_dim=False,
                    add_frame_dim=True,
                )
            )
            coarse_mask_binary_tensors.append(
                preprocess_singleturn_mask(
                    coarse_mask,
                    singleturn_sample_size,
                    add_batch_dim=False,
                    add_frame_dim=True,
                )
            )
        if coarse_mask_frame_tensors:
            coarse_mask_frame_batch = torch.stack(coarse_mask_frame_tensors, dim=0).to(
                device=device,
                dtype=weight_dtype,
                non_blocking=True,
            )
            coarse_mask_binary_batch = torch.stack(coarse_mask_binary_tensors, dim=0).to(
                device=device,
                dtype=weight_dtype,
                non_blocking=True,
            )
            source_valid_latents = source_frame_latents[coarse_valid_offsets]
            noise_valid_latents = noise_latents[coarse_valid_offsets]
            with torch.no_grad():
                coarse_mask_latents = vae.encode(coarse_mask_frame_batch.permute(0, 2, 1, 3, 4))[0].mode()
                coarse_latent_mask = resize_singleturn_mask_to_latent_grid(coarse_mask_binary_batch, source_valid_latents)
                coarse_noisy_anchor_latents = torch.where(
                    coarse_latent_mask.to(dtype=torch.bool).expand_as(source_valid_latents),
                    noise_valid_latents,
                    source_valid_latents,
                )
            coarse_index_by_offset = {offset: idx for idx, offset in enumerate(coarse_valid_offsets)}
        else:
            coarse_index_by_offset = {}
    else:
        coarse_index_by_offset = {}

    written = 0
    for local_offset, record in enumerate(batch_records):
        variants = [("normal", record["mask_check_image"], record["mask_frame_image"], record.get("mask_sam_image", ""), bool(record["used_mask_sam"]), mask_check_frame_latents[local_offset], mask_frame_latents[local_offset])]
        coarse_mask_path = coarse_mask_paths[local_offset] if coarse_variant_enabled else None
        coarse_encoded_idx = coarse_index_by_offset.get(local_offset)
        if coarse_mask_path is not None and coarse_encoded_idx is not None:
            variants.append(
                (
                    "coarse",
                    coarse_mask_path,
                    coarse_mask_path,
                    coarse_mask_path,
                    True,
                    coarse_mask_latents[coarse_encoded_idx],
                    coarse_mask_latents[coarse_encoded_idx],
                    coarse_noisy_anchor_latents[coarse_encoded_idx],
                )
            )

        for variant_item in variants:
            if len(variant_item) == 7:
                (
                    variant_name,
                    mask_check_image,
                    mask_frame_image,
                    mask_sam_image,
                    used_mask_sam,
                    mask_check_latent,
                    mask_frame_latent,
                ) = variant_item
                noisy_anchor_latent = noisy_anchor_latents[local_offset]
            else:
                (
                    variant_name,
                    mask_check_image,
                    mask_frame_image,
                    mask_sam_image,
                    used_mask_sam,
                    mask_check_latent,
                    mask_frame_latent,
                    noisy_anchor_latent,
                ) = variant_item
            cache_path = cache_dir / build_cache_variant_name(int(record["global_index"]), record["source_image"], variant_name)
            if cache_path.exists() and not overwrite:
                raise FileExistsError(f"Cache file already exists: {cache_path}. Use --overwrite to replace it.")

            payload = {
                "mode": "singleturn_object_removal_v2",
                "dataset_type": record.get("dataset_type", "corne_object_removal"),
                "variant": variant_name,
                "mask_frame_latent": mask_frame_latent.detach().cpu().to(weight_dtype),
                "mask_check_latent": mask_check_latent.detach().cpu().to(weight_dtype),
                "source_frame_latent": source_frame_latents[local_offset].detach().cpu().to(weight_dtype),
                "noisy_anchor_latent": noisy_anchor_latent.detach().cpu().to(weight_dtype),
                "target_latent": bg_latents[local_offset].detach().cpu().to(weight_dtype),
                "source_image": record["source_image"],
                "bg_image": record["bg_image"],
                "mask_check_image": mask_check_image,
                "mask_frame_image": mask_frame_image,
                "used_mask_sam": bool(used_mask_sam),
                "singleturn_sample_size": list(singleturn_sample_size),
                "shared_prompt_cache": str(shared_prompt_path.relative_to(output_dir)),
                "text": CORNE_SINGLETURN_PROMPT,
                "formatted_text": CORNE_SINGLETURN_PROMPT,
            }
            subset = record.get("subset", "")
            if subset:
                payload["subset"] = subset
            if mask_sam_image:
                payload["mask_sam_image"] = mask_sam_image
                payload["mask_sam_latent"] = mask_frame_latent.detach().cpu().to(weight_dtype)
            if variant_name == "coarse":
                payload["generated_mask_image"] = mask_check_image

            torch.save(payload, cache_path)
            manifest.append(build_manifest_entry(cache_path, output_dir, {**record, "variant": variant_name, "mask_check_image": mask_check_image, "mask_frame_image": mask_frame_image, "mask_sam_image": mask_sam_image, "used_mask_sam": used_mask_sam, "generated_mask_image": mask_check_image}))
            written += 1

    return written


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    metadata_path = output_dir / "metadata.json"
    shared_prompt_path = output_dir / "shared_prompt_embeds.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = get_weight_dtype(args.dtype, device)
    config = OmegaConf.load(args.config_path)

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
            "tokenizer",
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
            "text_encoder",
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval().to(device)
    vae = AutoencoderKLWan.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
            "vae",
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).eval().to(device, dtype=weight_dtype)

    with torch.no_grad():
        prompt_ids = tokenizer(
            [CORNE_SINGLETURN_PROMPT],
            padding="max_length",
            max_length=args.tokenizer_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_attention_mask = prompt_ids.attention_mask
        shared_prompt_embeds = text_encoder(
            prompt_ids.input_ids.to(device),
            attention_mask=prompt_attention_mask.to(device),
        )[0][0].detach().cpu().to(weight_dtype)
        shared_prompt_seq_len = int(prompt_attention_mask.gt(0).sum(dim=1)[0].item())

    torch.save(
        {
            "prompt_embeds": shared_prompt_embeds,
            "prompt_seq_len": shared_prompt_seq_len,
            "text": CORNE_SINGLETURN_PROMPT,
            "formatted_text": CORNE_SINGLETURN_PROMPT,
            "mode": "singleturn_object_removal_prompt_cache",
            "tokenizer_max_length": args.tokenizer_max_length,
            "conditioning_format": "mask-latent + clean-source-latent",
        },
        shared_prompt_path,
    )

    preprocess_dataset = SingleTurnPreprocessIterableDataset(
        data_root=args.train_data_dir,
        sample_size=args.singleturn_sample_size,
        max_samples_with_mask_sam=args.max_samples_with_mask_sam,
        max_samples_without_mask_sam=args.max_samples_without_mask_sam,
        skip_samples_with_mask_sam=args.skip_samples_with_mask_sam,
        skip_samples_without_mask_sam=args.skip_samples_without_mask_sam,
    )
    dataset_type = getattr(preprocess_dataset, "dataset_type", "corne_object_removal")

    manifest: list[dict] = []
    batch_records: list[dict] = []
    generated_samples = 0
    progress_bar = tqdm(desc=f"Preprocessing {dataset_type} SingleTurn cache")

    for sample in preprocess_dataset:
        batch_records.append(sample)
        if len(batch_records) < args.batch_size:
            continue

        generated_samples += _flush_batch(
            batch_records=batch_records,
            device=device,
            weight_dtype=weight_dtype,
            vae=vae,
            cache_dir=cache_dir,
            output_dir=output_dir,
            shared_prompt_path=shared_prompt_path,
            singleturn_sample_size=args.singleturn_sample_size,
            manifest=manifest,
            overwrite=args.overwrite,
        )
        batch_records = []
        progress_bar.update(generated_samples - progress_bar.n)
        progress_bar.set_postfix(
            generated=generated_samples,
            with_mask_sam=preprocess_dataset.num_samples_with_mask_sam,
            without_mask_sam=preprocess_dataset.num_samples_without_mask_sam,
        )

    if batch_records:
        generated_samples += _flush_batch(
            batch_records=batch_records,
            device=device,
            weight_dtype=weight_dtype,
            vae=vae,
            cache_dir=cache_dir,
            output_dir=output_dir,
            shared_prompt_path=shared_prompt_path,
            singleturn_sample_size=args.singleturn_sample_size,
            manifest=manifest,
            overwrite=args.overwrite,
        )
        progress_bar.update(generated_samples - progress_bar.n)

    progress_bar.close()

    num_normal_samples = sum(1 for entry in manifest if entry.get("variant", "normal") == "normal")
    num_coarse_samples = sum(1 for entry in manifest if entry.get("variant") == "coarse")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    metadata = {
        "mode": "singleturn_object_removal_v2",
        "dataset_type": dataset_type,
        "conditioning_format": "8-frame 2-prefix mask-latent + clean-source-latent",
        "mask_check_source": "object_effect_mask" if dataset_type == "objectclear_object_removal" else "mask-check",
        "mask_sam_source": "object_mask" if dataset_type == "objectclear_object_removal" else "mask_sam",
        "variants": ["normal", "coarse"] if dataset_type == "objectclear_object_removal" else ["normal"],
        "coarse_mask_source": "object_effect_mask" if dataset_type == "objectclear_object_removal" else "",
        "coarse_mask_root": "generated_masks/coarse" if dataset_type == "objectclear_object_removal" else "",
        "prefix_frames": SINGLETURN_FIRST_FRAME_FIXED_PREFIX_FRAMES,
        "total_frames": SINGLETURN_TOTAL_FRAMES,
        "pixel_space_source_masking": False,
        "max_samples_with_mask_sam": args.max_samples_with_mask_sam,
        "max_samples_without_mask_sam": args.max_samples_without_mask_sam,
        "skip_samples_with_mask_sam": args.skip_samples_with_mask_sam,
        "skip_samples_without_mask_sam": args.skip_samples_without_mask_sam,
        "num_samples_with_mask_sam": preprocess_dataset.num_samples_with_mask_sam,
        "num_samples_without_mask_sam": preprocess_dataset.num_samples_without_mask_sam,
        "stopped_early_when_quotas_met": bool(preprocess_dataset.stopped_early_when_quotas_met),
        "num_samples_total": generated_samples,
        "num_original_samples": preprocess_dataset.num_samples_with_mask_sam + preprocess_dataset.num_samples_without_mask_sam,
        "num_normal_samples": num_normal_samples,
        "num_coarse_samples": num_coarse_samples,
        "singleturn_sample_size": list(args.singleturn_sample_size),
        "shared_prompt_text": CORNE_SINGLETURN_PROMPT,
        "shared_prompt_cache": str(shared_prompt_path.relative_to(output_dir)),
        "manifest_path": str(manifest_path.relative_to(output_dir)),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "metadata_path": str(metadata_path),
                "shared_prompt_cache": str(shared_prompt_path),
                "num_samples_total": generated_samples,
                "num_normal_samples": num_normal_samples,
                "num_coarse_samples": num_coarse_samples,
                "num_samples_with_mask_sam": preprocess_dataset.num_samples_with_mask_sam,
                "num_samples_without_mask_sam": preprocess_dataset.num_samples_without_mask_sam,
                "dataset_type": dataset_type,
                "variants": ["normal", "coarse"] if dataset_type == "objectclear_object_removal" else ["normal"],
                "skip_samples_with_mask_sam": args.skip_samples_with_mask_sam,
                "skip_samples_without_mask_sam": args.skip_samples_without_mask_sam,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
