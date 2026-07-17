#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
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

from videox_fun.data.singleturn_dataset import load_singleturn_cache_payload
from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel, WanTransformer3DModel
from videox_fun.pipeline import WanPipeline
from videox_fun.utils.lora_utils import merge_lora
from videox_fun.utils.singleturn_utils import (
    SINGLETURN_OBJECT_REMOVAL_DENSIFIED_TOTAL_FRAMES,
    SINGLETURN_TAIL_TRAJECTORY_PREFIX_FRAMES,
    build_singleturn_tail_trajectory_prefix_latents,
    encode_singleturn_video_to_latents,
    generate_singleturn_tail_trajectory_sample_from_latents,
    normalize_singleturn_sample_size,
    save_singleturn_outputs,
)
from videox_fun.utils.utils import filter_kwargs

OUTPUT_STEM = "singleturn_tail_trajectory_refine"
EXPECTED_CACHE_MODE = "singleturn_object_removal_v3_tail_interp11"


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
    parser = argparse.ArgumentParser(description="SingleTurn 11-frame tail trajectory refinement from one coarse sample.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/Wan2.1-T2V-1.3B",
        help="Base Wan model path.",
    )
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Wan config path.")
    parser.add_argument("--coarse_sample_dir", type=str, required=True, help="Directory containing one completed coarse SingleTurn sample.")
    parser.add_argument("--lora_path", type=str, required=True, help="11-frame coarse LoRA checkpoint used for tail trajectory refinement.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA merge multiplier.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for refined outputs.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Optional negative prompt.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Inference weight dtype.")
    parser.add_argument("--fps", type=int, default=4, help="Saved video FPS.")
    parser.add_argument(
        "--video_format",
        type=str,
        default="mp4",
        choices=["gif", "mp4"],
        help="Saved preview video format.",
    )
    return parser.parse_args()


def _resolve_runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_weight_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16


def _resolve_saved_path(base_dir: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()

    checked = [
        base_dir / candidate,
        base_dir.parent / candidate,
        base_dir / path.name,
        base_dir.parent / path.name,
    ]
    for resolved in checked:
        if resolved.exists():
            return resolved.resolve()
    return path


def _resolve_shared_prompt_cache_path(cache_path: Path, shared_prompt_cache: str) -> Path:
    path = Path(shared_prompt_cache)
    if path.is_absolute():
        return path
    checked = [
        cache_path.parent.parent / shared_prompt_cache,
        cache_path.parent / shared_prompt_cache,
        Path(shared_prompt_cache),
    ]
    for resolved in checked:
        if resolved.exists():
            return resolved.resolve()
    return path


def _load_prompt_cache(cache_path: Path, payload: dict) -> dict:
    prompt_embeds = payload.get("prompt_embeds")
    prompt_seq_len = payload.get("prompt_seq_len")
    prompt_text = payload.get("text", "")
    formatted_text = payload.get("formatted_text", prompt_text)

    shared_prompt_cache = payload.get("shared_prompt_cache")
    if shared_prompt_cache:
        shared_payload = load_singleturn_cache_payload(str(_resolve_shared_prompt_cache_path(cache_path, shared_prompt_cache)))
        prompt_embeds = shared_payload.get("prompt_embeds", prompt_embeds)
        prompt_seq_len = shared_payload.get("prompt_seq_len", prompt_seq_len)
        prompt_text = shared_payload.get("text", prompt_text)
        formatted_text = shared_payload.get("formatted_text", prompt_text)

    if prompt_embeds is None or prompt_seq_len is None:
        raise ValueError(
            f"Cached SingleTurn sample {cache_path} must contain prompt_embeds/prompt_seq_len or shared_prompt_cache."
        )
    return {
        "prompt_embeds": prompt_embeds,
        "prompt_seq_len": int(prompt_seq_len),
        "text": prompt_text,
        "formatted_text": formatted_text,
    }


def _resolve_coarse_video_path(sample_dir: Path, meta: dict) -> Path:
    outputs = meta.get("outputs") or meta.get("coarse_outputs") or {}
    candidates = []
    full_mp4 = outputs.get("full_mp4")
    if full_mp4:
        candidates.append(_resolve_saved_path(sample_dir, full_mp4))
    candidates.append(sample_dir / "singleturn_full.mp4")
    full_gif = outputs.get("full_gif")
    if full_gif:
        candidates.append(_resolve_saved_path(sample_dir, full_gif))
    candidates.append(sample_dir / "singleturn_full.gif")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find coarse full video under {sample_dir}. Checked: {[str(candidate) for candidate in candidates]}"
    )


def _build_pipeline(args, config, device: torch.device, weight_dtype: torch.dtype):
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
    pipeline = merge_lora(
        pipeline,
        args.lora_path,
        args.lora_alpha,
        device=device,
        dtype=weight_dtype,
        transformer_only=True,
    )
    return pipeline


def main():
    args = parse_args()

    coarse_sample_dir = Path(args.coarse_sample_dir).resolve()
    if not coarse_sample_dir.is_dir():
        raise FileNotFoundError(f"Coarse sample directory does not exist: {coarse_sample_dir}")

    meta_path = coarse_sample_dir / "singleturn_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing coarse sample metadata: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    total_frames = int(meta.get("singleturn_total_frames", -1))
    if total_frames != SINGLETURN_OBJECT_REMOVAL_DENSIFIED_TOTAL_FRAMES:
        raise ValueError(
            "Tail trajectory refinement only supports 11-frame coarse samples. "
            f"Got singleturn_total_frames={total_frames} in {meta_path}."
        )

    cache_path_value = meta.get("cache_path")
    if not cache_path_value:
        raise ValueError(f"Missing cache_path in {meta_path}.")
    cache_path = _resolve_saved_path(coarse_sample_dir, cache_path_value)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Resolved coarse cache path does not exist: {cache_path}")

    coarse_payload = load_singleturn_cache_payload(str(cache_path))
    if coarse_payload.get("mode") != EXPECTED_CACHE_MODE:
        raise ValueError(
            f"Tail trajectory refinement requires cache mode {EXPECTED_CACHE_MODE!r}, "
            f"got {coarse_payload.get('mode')!r} from {cache_path}."
        )

    missing = [key for key in ("mask_frame_latent", "source_frame_latent", "singleturn_sample_size") if key not in coarse_payload]
    if missing:
        raise ValueError(f"Cached SingleTurn sample {cache_path} is missing keys: {missing}")

    coarse_video_path = _resolve_coarse_video_path(coarse_sample_dir, meta)
    sample_size = normalize_singleturn_sample_size(coarse_payload["singleturn_sample_size"])
    prompt_cache = _load_prompt_cache(cache_path, coarse_payload)

    device = _resolve_runtime_device()
    weight_dtype = get_weight_dtype(args.dtype, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    config = OmegaConf.load(args.config_path)
    pipeline = _build_pipeline(args, config, device, weight_dtype)

    encoded_latents_11 = encode_singleturn_video_to_latents(
        pipeline.vae,
        coarse_video_path,
        sample_size,
        device=device,
        weight_dtype=weight_dtype,
        expected_frames=SINGLETURN_OBJECT_REMOVAL_DENSIFIED_TOTAL_FRAMES,
    )
    prefix_latents_8 = build_singleturn_tail_trajectory_prefix_latents(encoded_latents_11)

    with torch.no_grad():
        generation = generate_singleturn_tail_trajectory_sample_from_latents(
            pipeline=pipeline,
            mask_frame_latent=coarse_payload["mask_frame_latent"],
            source_frame_latent=coarse_payload["source_frame_latent"],
            prompt_embeds=prompt_cache["prompt_embeds"],
            prompt_seq_len=prompt_cache["prompt_seq_len"],
            prefix_latents_8=prefix_latents_8,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            weight_dtype=weight_dtype,
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = save_singleturn_outputs(
        full_frames=generation["full_frames"],
        tail_frames=generation["tail_frames"],
        output_dir=str(output_dir),
        stem=OUTPUT_STEM,
        fps=args.fps,
        video_format=args.video_format,
    )

    final_latents_path = output_dir / f"{OUTPUT_STEM}_final_latents.pt"
    prefix_latents_path = output_dir / f"{OUTPUT_STEM}_prefix_latents_8.pt"
    init_latents_path = output_dir / f"{OUTPUT_STEM}_init_latents_11.pt"
    encoded_latents_path = output_dir / f"{OUTPUT_STEM}_encoded_latents_11.pt"
    torch.save(generation["latents"], final_latents_path)
    torch.save(generation["prefix_latents"], prefix_latents_path)
    torch.save(generation["init_latents"], init_latents_path)
    torch.save(encoded_latents_11.detach().cpu(), encoded_latents_path)

    metadata = {
        "mode": "singleturn_tail_trajectory_refine_v1",
        "coarse_sample_dir": str(coarse_sample_dir),
        "coarse_meta_path": str(meta_path.resolve()),
        "coarse_cache_path": str(cache_path.resolve()),
        "coarse_video_path": str(coarse_video_path),
        "coarse_cache_mode": coarse_payload.get("mode", ""),
        "prompt": prompt_cache["text"],
        "formatted_prompt": prompt_cache["formatted_text"],
        "source_text": coarse_payload.get("text", ""),
        "source_formatted_text": coarse_payload.get("formatted_text", coarse_payload.get("text", "")),
        "singleturn_total_frames": SINGLETURN_OBJECT_REMOVAL_DENSIFIED_TOTAL_FRAMES,
        "frozen_prefix_frames": SINGLETURN_TAIL_TRAJECTORY_PREFIX_FRAMES,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "sample_size": list(sample_size),
        "fps": args.fps,
        "dtype": args.dtype,
        "lora_path": args.lora_path,
        "lora_alpha": args.lora_alpha,
        "prefix_construction_formula": {
            "P1_to_P5": "copy F1..F5",
            "P6": "0.75 * F5 + 0.25 * F9",
            "P7": "0.6 * F5 + 0.4 * F10",
            "P8": "0.5 * F5 + 0.5 * F11",
        },
        "artifacts": {
            "encoded_latents_11": str(encoded_latents_path),
            "prefix_latents_8": str(prefix_latents_path),
            "init_latents_11": str(init_latents_path),
            "final_latents": str(final_latents_path),
        },
        "outputs": output_paths,
    }
    metadata_path = output_dir / f"{OUTPUT_STEM}_meta.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "outputs": output_paths,
                "artifacts": metadata["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
