#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.models import AutoencoderKLWan
from videox_fun.utils.singleturn_utils import decode_singleturn_latent_frames
from videox_fun.utils.utils import save_videos_grid


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
    parser = argparse.ArgumentParser(description="Decode one SingleTurn refinement cache sample into preview media.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def get_weight_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16


def copy_if_exists(src: str, dst: Path):
    src_path = Path(src)
    if not src_path.exists():
        return None
    if src_path.suffix.lower() == ".gif":
        dst.write_bytes(src_path.read_bytes())
    else:
        Image.open(src_path).save(dst)
    return str(dst)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.cache_path, map_location="cpu")
    device = torch.device(args.device)
    weight_dtype = get_weight_dtype(args.dtype)
    config = OmegaConf.load(args.config_path)

    vae = AutoencoderKLWan.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
            "vae",
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).eval().to(device, dtype=weight_dtype)

    decoded_outputs = {}
    for key in ("input_latents", "target_latents"):
        latents = payload[key]
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        latents = latents.to(device=device, dtype=weight_dtype)
        with torch.no_grad():
            frames = decode_singleturn_latent_frames(vae, latents, decode_dtype=weight_dtype).cpu().float()

        gif_path = output_dir / f"{key}.gif"
        last_frame_path = output_dir / f"{key}_last.png"
        save_videos_grid(frames, str(gif_path), fps=args.fps)
        last_frame = (frames[0, :, -1].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
        Image.fromarray(last_frame).save(last_frame_path)
        decoded_outputs[key] = {
            "gif": str(gif_path),
            "last_frame": str(last_frame_path),
            "shape": list(payload[key].shape),
            "dtype": str(payload[key].dtype),
        }

    reference_outputs = {}
    for src_key, out_name in (
        ("source_image", "source.png"),
        ("bg_image", "bg.png"),
        ("mask_check_image", "mask_check.png"),
        ("mask_frame_image", "mask_frame.png"),
        ("coarse_full_gif", "coarse_full.gif"),
    ):
        src = payload.get(src_key)
        if isinstance(src, str):
            copied = copy_if_exists(src, output_dir / out_name)
            if copied is not None:
                reference_outputs[src_key] = copied

    summary = {
        "cache_path": args.cache_path,
        "mode": payload.get("mode"),
        "coarse_output_dir": payload.get("coarse_output_dir"),
        "source_cache_path": payload.get("source_cache_path"),
        "singleturn_sample_size": payload.get("singleturn_sample_size"),
        "decoded_outputs": decoded_outputs,
        "reference_outputs": reference_outputs,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
