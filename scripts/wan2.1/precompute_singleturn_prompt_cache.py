#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

import torch
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

from videox_fun.models import WanT5EncoderModel


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
    parser = argparse.ArgumentParser(description="Precompute one shared Wan T5 prompt embedding cache.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Base Wan model path.")
    parser.add_argument("--output_path", type=str, required=True, help="Output .pt path for shared prompt embedding.")
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Wan config path.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt to encode. Empty string gives null prompt embedding.")
    parser.add_argument("--tokenizer_max_length", type=int, default=512, help="Tokenizer max length.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Cache tensor dtype.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache.")
    return parser.parse_args()


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
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        print(json.dumps({"output_path": str(output_path), "status": "exists"}, indent=2))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    with torch.no_grad():
        prompt_ids = tokenizer(
            [args.prompt],
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
        )[0][0].detach().cpu().to(weight_dtype)
        prompt_seq_len = int(prompt_attention_mask.gt(0).sum(dim=1)[0].item())

    torch.save(
        {
            "prompt_embeds": prompt_embeds,
            "prompt_seq_len": prompt_seq_len,
            "text": args.prompt,
            "formatted_text": args.prompt,
            "mode": "singleturn_shared_prompt",
            "tokenizer_max_length": args.tokenizer_max_length,
        },
        output_path,
    )
    print(json.dumps({"output_path": str(output_path), "prompt_seq_len": prompt_seq_len}, indent=2))


if __name__ == "__main__":
    main()
