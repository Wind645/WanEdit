#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
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
from videox_fun.utils.singleturn_utils import (DEFAULT_SINGLETURN_PROMPT_TEMPLATE,
                                               format_singleturn_prompt,
                                               normalize_singleturn_sample_size)


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
    parser = argparse.ArgumentParser(description="Precompute SingleTurn T5 and VAE tensors for cached training.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Base Wan model path.")
    parser.add_argument("--train_data_dir", type=str, required=True, help="SingleTurn data root.")
    parser.add_argument("--train_data_manifest", type=str, default=None, help="Optional SingleTurn manifest.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write cached tensors and manifest.")
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Wan config path.")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_SINGLETURN_PROMPT_TEMPLATE, help="Prompt template; must contain '{prompt}'.")
    parser.add_argument("--reconstruction_mode", action="store_true", help="Cache source latents only and one shared reconstruction prompt embedding.")
    parser.add_argument("--reconstruction_prompt", type=str, default="Reconstruct the source image.", help="Fixed prompt encoded once for reconstruction-mode training.")
    parser.add_argument("--video_sample_size", type=int, default=512, help="Resize/crop size used before VAE encode.")
    parser.add_argument("--singleturn_sample_size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"), help="Optional non-square SingleTurn sample size used before VAE encode.")
    parser.add_argument("--tokenizer_max_length", type=int, default=512, help="Tokenizer max length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples to preprocess per batch.")
    parser.add_argument("--num_workers", type=int, default=8, help="CPU worker count used to decode and preprocess images.")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Number of batches prefetched by each worker.")
    parser.add_argument("--parquet_batch_size", type=int, default=64, help="Internal parquet read batch size per worker.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Cache tensor dtype.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    args = parser.parse_args()
    sample_size = normalize_singleturn_sample_size(
        args.singleturn_sample_size if args.singleturn_sample_size is not None else args.video_sample_size
    )

    if (not args.reconstruction_mode) and "{prompt}" not in args.prompt_template:
        raise ValueError("--prompt_template must contain the '{prompt}' placeholder.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative.")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be positive.")
    if args.parquet_batch_size <= 0:
        raise ValueError("--parquet_batch_size must be positive.")
    if any(dim % 16 != 0 for dim in sample_size):
        raise ValueError(f"SingleTurn sample size must be divisible by 16, got {sample_size}.")
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


def build_manifest_entry(
    cache_path: Path,
    output_dir: Path,
    source_image=None,
    edited_image=None,
    prompt=None,
    formatted_prompt=None,
    row_index=None,
):
    entry = {"cache_path": str(cache_path.relative_to(output_dir))}
    if source_image is not None:
        entry["source_image"] = source_image
    if edited_image is not None:
        entry["edited_image"] = edited_image
    if prompt is not None:
        entry["prompt"] = prompt
    if formatted_prompt is not None:
        entry["formatted_prompt"] = formatted_prompt
    if row_index is not None:
        entry["row_index"] = row_index
    return entry


def main():
    args = parse_args()
    singleturn_sample_size = normalize_singleturn_sample_size(
        args.singleturn_sample_size if args.singleturn_sample_size is not None else args.video_sample_size
    )

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

    manifest = []
    preprocess_dataset = SingleTurnPreprocessIterableDataset(
        data_root=args.train_data_dir,
        manifest_path=args.train_data_manifest,
        sample_size=singleturn_sample_size,
        parquet_batch_size=args.parquet_batch_size,
        reconstruction_only=args.reconstruction_mode,
    )
    use_parquet_source = preprocess_dataset.use_parquet
    total_records = preprocess_dataset.total_length
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    preprocess_dataloader = DataLoader(preprocess_dataset, **dataloader_kwargs)

    generated_samples = 0
    progress_bar = tqdm(total=total_records, desc="Preprocessing SingleTurn cache")

    if args.reconstruction_mode:
        with torch.no_grad():
            prompt_ids = tokenizer(
                [args.reconstruction_prompt],
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
                "text": args.reconstruction_prompt,
                "formatted_text": args.reconstruction_prompt,
                "mode": "singleturn_reconstruction",
                "tokenizer_max_length": args.tokenizer_max_length,
            },
            shared_prompt_path,
        )

    for batch in preprocess_dataloader:
        batch_size = int(batch["pixel_values_src_image"].shape[0])
        raw_prompts = [""] * batch_size if args.reconstruction_mode else list(batch["text"])
        formatted_prompts = (
            [args.reconstruction_prompt] * batch_size
            if args.reconstruction_mode
            else [format_singleturn_prompt(raw_prompt, args.prompt_template) for raw_prompt in raw_prompts]
        )
        sample_records = []

        for offset in range(len(raw_prompts)):
            global_index = int(batch["global_index"][offset].item()) if torch.is_tensor(batch["global_index"]) else int(batch["global_index"][offset])
            source_image = batch["source_image"][offset]
            edited_image = "" if args.reconstruction_mode else batch["edited_image"][offset]
            row_index = int(batch["row_index"][offset].item()) if torch.is_tensor(batch["row_index"]) else int(batch["row_index"][offset])
            row_index = None if row_index < 0 else row_index
            raw_prompt = raw_prompts[offset]
            formatted_prompt = formatted_prompts[offset]

            cache_path = cache_dir / build_cache_name(global_index, source_image)
            if cache_path.exists() and not args.overwrite:
                raise FileExistsError(f"Cache file already exists: {cache_path}. Use --overwrite to replace it.")

            sample_records.append(
                {
                    "offset": offset,
                    "cache_path": cache_path,
                    "source_image": source_image,
                    "edited_image": edited_image,
                    "row_index": row_index,
                    "raw_prompt": raw_prompt,
                    "formatted_prompt": formatted_prompt,
                }
            )
            manifest.append(
                build_manifest_entry(
                    cache_path=cache_path,
                    output_dir=output_dir,
                    source_image=source_image,
                    edited_image=edited_image,
                    prompt=raw_prompt,
                    formatted_prompt=formatted_prompt,
                    row_index=row_index,
                )
            )

        source_batch = batch["pixel_values_src_image"].to(device=device, dtype=weight_dtype, non_blocking=True)
        sample_indices = torch.tensor([record["offset"] for record in sample_records], device=source_batch.device, dtype=torch.long)
        source_batch = source_batch.index_select(0, sample_indices)

        with torch.no_grad():
            source_posterior = vae.encode(source_batch.permute(0, 2, 1, 3, 4))[0]
            if not args.reconstruction_mode:
                target_batch = batch["pixel_values_tgt_image"].to(device=device, dtype=weight_dtype, non_blocking=True)
                target_batch = target_batch.index_select(0, sample_indices)
                selected_formatted_prompts = [record["formatted_prompt"] for record in sample_records]
                target_posterior = vae.encode(target_batch.permute(0, 2, 1, 3, 4))[0]

                prompt_ids = tokenizer(
                    selected_formatted_prompts,
                    padding="max_length",
                    max_length=args.tokenizer_max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                prompt_attention_mask = prompt_ids.attention_mask
                prompt_embeds = text_encoder(
                    prompt_ids.input_ids.to(device),
                    attention_mask=prompt_attention_mask.to(device),
                )[0].to(dtype=weight_dtype)
                prompt_seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()

        for local_offset, record in enumerate(sample_records):
            payload = {
                "source_latent_mean": source_posterior.mean[local_offset].detach().cpu().to(weight_dtype),
                "source_latent_logvar": source_posterior.logvar[local_offset].detach().cpu().to(weight_dtype),
                "text": record["raw_prompt"],
                "formatted_text": record["formatted_prompt"],
                "source_image": record["source_image"],
                "edited_image": record["edited_image"],
                "row_index": record["row_index"],
                "prompt_template": args.prompt_template,
                "singleturn_sample_size": list(singleturn_sample_size),
            }
            if args.reconstruction_mode:
                payload["mode"] = "singleturn_reconstruction"
                payload["shared_prompt_cache"] = str(shared_prompt_path.relative_to(output_dir))
            else:
                payload.update(
                    {
                        "target_latent_mean": target_posterior.mean[local_offset].detach().cpu().to(weight_dtype),
                        "target_latent_logvar": target_posterior.logvar[local_offset].detach().cpu().to(weight_dtype),
                        "prompt_embeds": prompt_embeds[local_offset].detach().cpu().to(weight_dtype),
                        "prompt_seq_len": int(prompt_seq_lens[local_offset].item()),
                    }
                )
            torch.save(payload, record["cache_path"])
        generated_samples += len(sample_records)

        progress_bar.update(len(raw_prompts))
        progress_bar.set_postfix(generated=generated_samples)

    progress_bar.close()

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len(manifest),
                "manifest_path": str(manifest_path),
                "train_data_dir": args.train_data_dir,
                "train_data_manifest": args.train_data_manifest,
                "source_layout": "parquet" if use_parquet_source else "files",
                "prompt_template": args.prompt_template,
                "reconstruction_mode": args.reconstruction_mode,
                "reconstruction_prompt": args.reconstruction_prompt if args.reconstruction_mode else None,
                "shared_prompt_cache": str(shared_prompt_path) if args.reconstruction_mode else None,
                "singleturn_sample_size": list(singleturn_sample_size),
                "tokenizer_max_length": args.tokenizer_max_length,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "generated_samples": generated_samples,
            },
            f,
            indent=2,
        )

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "num_samples": len(manifest),
                "generated_samples": generated_samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
