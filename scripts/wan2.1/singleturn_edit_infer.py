#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
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

from videox_fun.data.singleturn_dataset import CachedSingleTurnLatentDataset, load_singleturn_cache_payload
from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel, WanTransformer3DModel
from videox_fun.pipeline import WanPipeline
from videox_fun.utils.lora_utils import merge_lora
from videox_fun.utils.singleturn_utils import (
    DEFAULT_SINGLETURN_PROMPT_TEMPLATE,
    generate_singleturn_sample,
    generate_singleturn_sample_from_latents,
    normalize_singleturn_sample_size,
    preprocess_singleturn_image,
    save_singleturn_outputs,
)
from videox_fun.utils.utils import filter_kwargs


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


def _first_item(value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value[0]
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _resolve_cached_data_dir(args) -> Optional[str]:
    if args.cached_data_dir:
        return args.cached_data_dir
    if args.cached_data_meta:
        return str(Path(args.cached_data_meta).resolve().parent)
    if args.cached_sample_path:
        return str(Path(args.cached_sample_path).resolve().parent)
    return None


def _resolve_cache_path(cache_path: str, cached_data_dir: Optional[str]) -> str:
    if os.path.isabs(cache_path) or cached_data_dir is None:
        return cache_path
    return os.path.join(cached_data_dir, cache_path)


def _load_shared_prompt_cache(shared_prompt_cache: str):
    payload = load_singleturn_cache_payload(shared_prompt_cache)
    missing = [key for key in ("prompt_embeds", "prompt_seq_len") if key not in payload]
    if missing:
        raise ValueError(f"Shared prompt cache {shared_prompt_cache} is missing keys: {missing}")
    return {
        "prompt_embeds": payload["prompt_embeds"],
        "prompt_seq_len": int(payload["prompt_seq_len"]),
        "text": payload.get("text", ""),
        "formatted_text": payload.get("formatted_text", payload.get("text", "")),
    }


def _get_distributed_context() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, max(1, world_size)


def _resolve_runtime_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank, _ = _get_distributed_context()
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cuda")


def _save_singleturn_result(
    *,
    output_dir: str,
    stem: str,
    generation: dict,
    metadata: dict,
    fps: int,
):
    os.makedirs(output_dir, exist_ok=True)
    full_frames = generation["full_frames"]
    tail_frames = generation["tail_frames"]
    output_paths = save_singleturn_outputs(
        full_frames=full_frames,
        tail_frames=tail_frames,
        output_dir=output_dir,
        stem=stem,
        fps=fps,
    )

    metadata_path = os.path.join(output_dir, f"{stem}_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({**metadata, "outputs": output_paths}, f, indent=2)

    return output_paths, metadata_path


def _run_singleturn_image_mode(pipeline, args, weight_dtype, generator):
    source_tensor = preprocess_singleturn_image(args.image_path, args.sample_size).to(
        device=pipeline._execution_device,
        dtype=weight_dtype,
    )

    with torch.no_grad():
        generation = generate_singleturn_sample(
            pipeline=pipeline,
            source_tensor=source_tensor,
            prompt=args.prompt,
            prompt_template=args.prompt_template,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            weight_dtype=weight_dtype,
        )

    output_name = args.output_name or Path(args.image_path).stem
    output_paths, metadata_path = _save_singleturn_result(
        output_dir=args.output_dir,
        stem=output_name,
        generation=generation,
        metadata={
            "mode": "image",
            "image_path": args.image_path,
            "prompt": args.prompt,
            "formatted_prompt": generation["formatted_prompt"],
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "sample_size": list(args.sample_size),
        },
        fps=args.fps,
    )

    return {"outputs": output_paths, "metadata": metadata_path}


def _run_singleturn_cached_sample(
    *,
    pipeline,
    args,
    weight_dtype,
    generator,
    sample,
    output_dir: str,
    stem: str,
):
    source_latent = sample["source_latent_mean"]
    prompt_embeds = sample["prompt_embeds"]
    prompt_seq_len = sample["prompt_seq_len"]
    if torch.is_tensor(prompt_seq_len) and prompt_seq_len.ndim == 1 and prompt_seq_len.numel() == 1:
        prompt_seq_len = int(prompt_seq_len[0].item())
    elif torch.is_tensor(prompt_seq_len) and prompt_seq_len.ndim == 0:
        prompt_seq_len = int(prompt_seq_len.item())

    with torch.no_grad():
        generation = generate_singleturn_sample_from_latents(
            pipeline=pipeline,
            source_latent=source_latent,
            prompt_embeds=prompt_embeds,
            prompt_seq_len=prompt_seq_len,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            weight_dtype=weight_dtype,
        )

    raw_prompt = sample.get("text", "")
    formatted_prompt = sample.get("formatted_text", raw_prompt)
    output_paths, metadata_path = _save_singleturn_result(
        output_dir=output_dir,
        stem=stem,
        generation=generation,
        metadata={
            "mode": "cache",
            "cache_path": sample.get("cache_path", ""),
            "source_image": sample.get("source_image", ""),
            "edited_image": sample.get("edited_image", ""),
            "prompt": raw_prompt,
            "formatted_prompt": formatted_prompt,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
        },
        fps=args.fps,
    )

    return {
        "output_paths": output_paths,
        "metadata": metadata_path,
        "sample_output_dir": output_dir,
        "stem": stem,
        "cache_path": sample.get("cache_path", ""),
        "source_image": sample.get("source_image", ""),
        "prompt": raw_prompt,
        "formatted_prompt": formatted_prompt,
    }


def _run_singleturn_cached_mode(pipeline, args, weight_dtype, generator):
    cached_data_dir = _resolve_cached_data_dir(args)
    local_rank, world_size = _get_distributed_context()
    results = []
    shared_prompt_cache = None
    if args.shared_prompt_cache is not None:
        shared_prompt_cache = _load_shared_prompt_cache(args.shared_prompt_cache)

    if args.cached_sample_path is not None:
        cache_path = _resolve_cache_path(args.cached_sample_path, cached_data_dir)
        payload = load_singleturn_cache_payload(cache_path)
        sample = {
            "source_latent_mean": payload["source_latent_mean"],
            "prompt_embeds": shared_prompt_cache["prompt_embeds"] if shared_prompt_cache is not None else payload["prompt_embeds"],
            "prompt_seq_len": shared_prompt_cache["prompt_seq_len"] if shared_prompt_cache is not None else int(payload["prompt_seq_len"]),
            "text": shared_prompt_cache["text"] if shared_prompt_cache is not None else payload.get("text", ""),
            "formatted_text": (
                shared_prompt_cache["formatted_text"]
                if shared_prompt_cache is not None
                else payload.get("formatted_text", payload.get("text", ""))
            ),
            "cache_path": cache_path,
            "source_image": payload.get("source_image", ""),
            "edited_image": payload.get("edited_image", ""),
            "idx": 0,
        }
        output_name = args.output_name or Path(cache_path).stem
        sample_output_dir = args.output_dir if args.output_name else os.path.join(args.output_dir, output_name)
        results.append(
            _run_singleturn_cached_sample(
                pipeline=pipeline,
                args=args,
                weight_dtype=weight_dtype,
                generator=generator,
                sample=sample,
                output_dir=sample_output_dir,
                stem="singleturn",
            )
        )
    else:
        dataset = CachedSingleTurnLatentDataset(args.cached_data_meta, cached_data_dir)
        start_index = max(0, int(args.cached_start_index))
        if start_index >= len(dataset):
            raise ValueError(
                f"--cached_start_index {start_index} is out of range for cached dataset of length {len(dataset)}."
            )

        end_index = len(dataset)
        if args.cached_num_samples is not None:
            if args.cached_num_samples <= 0:
                raise ValueError("--cached_num_samples must be positive when provided.")
            end_index = min(len(dataset), start_index + int(args.cached_num_samples))

        indices = list(range(start_index, end_index))
        if not indices:
            raise ValueError("No cached samples selected for inference.")

        if world_size > 1:
            indices = indices[local_rank::world_size]
            if not indices:
                return {"outputs": [], "summary": None}

        dataset = Subset(dataset, indices)
        dataloader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": args.cached_num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": args.cached_num_workers > 0,
        }
        if args.cached_num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = args.cached_prefetch_factor
        dataloader = DataLoader(dataset, **dataloader_kwargs)

        for batch in tqdm(dataloader, desc="Running cached SingleTurn inference"):
            cache_path = _first_item(batch["cache_path"])
            batch_index = int(_first_item(batch["idx"]))
            sample_name = args.output_name or Path(str(cache_path)).stem
            sample_dir_name = f"{batch_index:06d}_{sample_name}"
            sample_output_root = args.output_dir if world_size == 1 else os.path.join(args.output_dir, f"rank{local_rank}")
            sample_output_dir = os.path.join(sample_output_root, sample_dir_name)

            sample = {
                "source_latent_mean": batch["source_latent_mean"],
                "prompt_embeds": shared_prompt_cache["prompt_embeds"] if shared_prompt_cache is not None else batch["prompt_embeds"],
                "prompt_seq_len": (
                    shared_prompt_cache["prompt_seq_len"] if shared_prompt_cache is not None else batch["prompt_seq_len"]
                ),
                "text": shared_prompt_cache["text"] if shared_prompt_cache is not None else _first_item(batch["text"]),
                "formatted_text": (
                    shared_prompt_cache["formatted_text"]
                    if shared_prompt_cache is not None
                    else _first_item(batch["formatted_text"])
                ),
                "cache_path": cache_path,
                "source_image": _first_item(batch["source_image"]),
                "edited_image": _first_item(batch["edited_image"]),
                "idx": batch_index,
            }

            results.append(
                _run_singleturn_cached_sample(
                    pipeline=pipeline,
                    args=args,
                    weight_dtype=weight_dtype,
                    generator=generator,
                    sample=sample,
                    output_dir=sample_output_dir,
                    stem="singleturn",
                )
            )

    summary_root = args.output_dir if world_size == 1 else os.path.join(args.output_dir, f"rank{local_rank}")
    summary_path = os.path.join(summary_root, "cached_infer_manifest.json")
    os.makedirs(summary_root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return {"outputs": results, "summary": summary_path}


def parse_args():
    parser = argparse.ArgumentParser(description="SingleTurn Wan LoRA image editing inference")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Base Wan model path.")
    parser.add_argument("--image_path", type=str, default=None, help="Source image path for direct image-mode inference.")
    parser.add_argument("--prompt", type=str, default=None, help="Edit instruction for direct image-mode inference.")
    parser.add_argument("--cached_sample_path", type=str, default=None, help="Path to one cached SingleTurn sample (.pt).")
    parser.add_argument("--cached_data_meta", type=str, default=None, help="Manifest for cached SingleTurn samples.")
    parser.add_argument("--cached_data_dir", type=str, default=None, help="Root directory used to resolve relative cache paths.")
    parser.add_argument("--shared_prompt_cache", type=str, default=None, help="Optional shared prompt embedding cache that overrides per-sample cached prompts.")
    parser.add_argument("--cached_start_index", type=int, default=0, help="Start index when iterating over cached manifests.")
    parser.add_argument("--cached_num_samples", type=int, default=None, help="Optional limit when iterating over cached manifests.")
    parser.add_argument("--cached_num_workers", type=int, default=2, help="CPU workers used to load cached samples.")
    parser.add_argument("--cached_prefetch_factor", type=int, default=2, help="Prefetch factor for cached sample loading.")
    parser.add_argument("--output_dir", type=str, default="outputs/singleturn", help="Directory for outputs.")
    parser.add_argument("--output_name", type=str, default=None, help="Optional filename stem for outputs.")
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Wan config path.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA checkpoint.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA merge multiplier.")
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=DEFAULT_SINGLETURN_PROMPT_TEMPLATE,
        help="Prompt template; must contain '{prompt}'.",
    )
    parser.add_argument("--negative_prompt", type=str, default="", help="Optional negative prompt.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument(
        "--sample_size",
        type=int,
        nargs="+",
        default=[512],
        help="Resize/crop size used before VAE encode. Pass one value for square or two values for HEIGHT WIDTH.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--fps", type=int, default=4, help="GIF playback FPS.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Inference weight dtype.")
    args = parser.parse_args()

    if "{prompt}" not in args.prompt_template:
        raise ValueError("--prompt_template must contain the '{prompt}' placeholder.")

    cache_mode = args.cached_sample_path is not None or args.cached_data_meta is not None
    image_mode = args.image_path is not None or args.prompt is not None
    if cache_mode and image_mode:
        raise ValueError("Choose either direct image mode or cached mode, not both.")
    if not cache_mode:
        if args.image_path is None or args.prompt is None:
            raise ValueError("Image mode requires both --image_path and --prompt.")
    else:
        if args.cached_sample_path is not None and args.cached_data_meta is not None:
            raise ValueError("Choose either --cached_sample_path or --cached_data_meta, not both.")
        if args.cached_sample_path is None and args.cached_data_meta is None:
            raise ValueError("Cached mode requires --cached_sample_path or --cached_data_meta.")
        if args.cached_num_workers < 0:
            raise ValueError("--cached_num_workers must be non-negative.")
        if args.cached_prefetch_factor <= 0:
            raise ValueError("--cached_prefetch_factor must be positive.")

    args.sample_size = normalize_singleturn_sample_size(args.sample_size)
    if any(dim % 16 != 0 for dim in args.sample_size):
        raise ValueError(f"--sample_size must be divisible by 16, got {args.sample_size}.")
    return args


def get_weight_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16


def main():
    args = parse_args()

    device = _resolve_runtime_device()
    weight_dtype = get_weight_dtype(args.dtype, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

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
    ).eval()
    vae = AutoencoderKLWan.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
            "vae",
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).eval().to(dtype=weight_dtype)
    transformer = WanTransformer3DModel.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
            "transformer",
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipeline = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    ).to(device)

    if args.lora_path:
        pipeline = merge_lora(
            pipeline,
            args.lora_path,
            args.lora_alpha,
            device=device,
            dtype=weight_dtype,
            transformer_only=True,
        )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.cached_sample_path is not None or args.cached_data_meta is not None:
        result = _run_singleturn_cached_mode(
            pipeline=pipeline,
            args=args,
            weight_dtype=weight_dtype,
            generator=generator,
        )
    else:
        result = _run_singleturn_image_mode(
            pipeline=pipeline,
            args=args,
            weight_dtype=weight_dtype,
            generator=generator,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
