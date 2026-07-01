#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoTokenizer

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
    build_singleturn_object_removal_latents,
    normalize_singleturn_sample_size,
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
    parser = argparse.ArgumentParser(description="Precompute CORNE SingleTurn object-removal cache.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Base Wan model path.")
    parser.add_argument("--train_data_dir", type=str, required=True, help="CORNE_extracted root containing shot/, bg/, mask-check/, and optional mask_sam/.")
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
    parser.add_argument("--max_samples_with_mask_sam", type=int, default=30000, help="Quota for samples whose conditioning mask comes from mask_sam.")
    parser.add_argument("--max_samples_without_mask_sam", type=int, default=30000, help="Quota for samples whose conditioning mask falls back to mask-check.")
    args = parser.parse_args()

    if args.train_data_manifest is not None:
        raise ValueError("CORNE object-removal preprocess no longer supports --train_data_manifest.")
    if args.num_workers != 0:
        raise ValueError("CORNE object-removal preprocess requires --num_workers 0 to preserve class quotas exactly.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.max_samples_with_mask_sam < 0 or args.max_samples_without_mask_sam < 0:
        raise ValueError("Quota arguments must be non-negative.")

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


def build_cache_name(global_index: int, source_image: str) -> str:
    return f"{int(global_index):06d}_{Path(str(source_image)).stem}.pt"


def build_manifest_entry(cache_path: Path, output_dir: Path, sample: dict) -> dict:
    entry = {
        "cache_path": str(cache_path.relative_to(output_dir)),
        "source_image": sample["source_image"],
        "bg_image": sample["bg_image"],
        "mask_check_image": sample["mask_check_image"],
        "used_mask_sam": bool(sample["used_mask_sam"]),
        "global_index": int(sample["global_index"]),
    }
    mask_sam_image = sample.get("mask_sam_image", "")
    if mask_sam_image:
        entry["mask_sam_image"] = mask_sam_image
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
        first_frame_latents = vae.encode(source_batch.permute(0, 2, 1, 3, 4))[0].mode()
        bg_latents = vae.encode(bg_batch.permute(0, 2, 1, 3, 4))[0].mode()
        latent_mask = resize_singleturn_mask_to_latent_grid(mask_check_batch, first_frame_latents)
        noise_latents = torch.randn_like(first_frame_latents)
        full_latents = build_singleturn_object_removal_latents(
            first_frame_latent=first_frame_latents,
            bg_latent=bg_latents,
            mask_check_latent=latent_mask,
            noise_latent=noise_latents,
        )

    for local_offset, record in enumerate(batch_records):
        cache_path = cache_dir / build_cache_name(int(record["global_index"]), record["source_image"])
        if cache_path.exists() and not overwrite:
            raise FileExistsError(f"Cache file already exists: {cache_path}. Use --overwrite to replace it.")

        payload = {
            "mode": "singleturn_object_removal",
            "dataset_type": "corne_object_removal",
            "full_latents": full_latents[local_offset].detach().cpu().to(weight_dtype),
            "first_frame_latent": first_frame_latents[local_offset].detach().cpu().to(weight_dtype),
            "source_image": record["source_image"],
            "bg_image": record["bg_image"],
            "mask_check_image": record["mask_check_image"],
            "used_mask_sam": bool(record["used_mask_sam"]),
            "singleturn_sample_size": list(singleturn_sample_size),
            "shared_prompt_cache": str(shared_prompt_path.relative_to(output_dir)),
            "text": CORNE_SINGLETURN_PROMPT,
            "formatted_text": CORNE_SINGLETURN_PROMPT,
        }
        mask_sam_image = record.get("mask_sam_image", "")
        if mask_sam_image:
            payload["mask_sam_image"] = mask_sam_image

        torch.save(payload, cache_path)
        manifest.append(build_manifest_entry(cache_path, output_dir, record))

    return len(batch_records)


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
        },
        shared_prompt_path,
    )

    preprocess_dataset = SingleTurnPreprocessIterableDataset(
        data_root=args.train_data_dir,
        sample_size=args.singleturn_sample_size,
        max_samples_with_mask_sam=args.max_samples_with_mask_sam,
        max_samples_without_mask_sam=args.max_samples_without_mask_sam,
    )

    manifest: list[dict] = []
    batch_records: list[dict] = []
    generated_samples = 0
    progress_bar = tqdm(desc="Preprocessing CORNE SingleTurn cache")

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

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    metadata = {
        "dataset_type": "corne_object_removal",
        "max_samples_with_mask_sam": args.max_samples_with_mask_sam,
        "max_samples_without_mask_sam": args.max_samples_without_mask_sam,
        "num_samples_with_mask_sam": preprocess_dataset.num_samples_with_mask_sam,
        "num_samples_without_mask_sam": preprocess_dataset.num_samples_without_mask_sam,
        "stopped_early_when_quotas_met": bool(preprocess_dataset.stopped_early_when_quotas_met),
        "num_samples_total": generated_samples,
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
                "num_samples_with_mask_sam": preprocess_dataset.num_samples_with_mask_sam,
                "num_samples_without_mask_sam": preprocess_dataset.num_samples_without_mask_sam,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
