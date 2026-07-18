#!/usr/bin/env python

import argparse
import json
import os
import shutil
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

from videox_fun.data.singleturn_dataset import SingleTurnPreprocessIterableDataset, load_singleturn_cache_payload
from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel
from videox_fun.utils.singleturn_utils import (
    CORNE_SINGLETURN_PROMPT,
    SINGLETURN_OBJECT_REMOVAL_CACHE_MODE,
    build_singleturn_noisy_anchor_latents,
    normalize_singleturn_sample_size,
    preprocess_singleturn_mask_frame,
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
    parser = argparse.ArgumentParser(description="Build CORNE SingleTurn five-keyframe object-removal cache.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/data/zhikai/VideoCoF/models/Wan2.1-T2V-1.3B",
        help="Base Wan model path.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="CORNE_extracted root containing shot/, bg/, mask-check/, and mask_sam/. Used only for raw-image preprocessing mode.",
    )
    parser.add_argument("--train_data_manifest", type=str, default=None, help="Deprecated and unsupported for CORNE object-removal mode.")
    parser.add_argument(
        "--source_cached_data_meta",
        type=str,
        default="/home/data/nas_hdd/CORNE_extracted/cache/singleturn_object_removal_wan2.1_1.3b_v3_twoprefix/manifest.json",
        help="Manifest for an existing legacy SingleTurn cache to convert into the new SAM-strict keyframe format.",
    )
    parser.add_argument(
        "--source_cached_data_dir",
        type=str,
        default=None,
        help="Root directory used to resolve relative cache paths from --source_cached_data_meta.",
    )
    parser.add_argument(
        "--legacy_duplicate_mask_frame_for_both_masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When converting legacy cache samples, reuse the existing legacy mask frame/latent for both "
            "mask_sam and mask_check instead of encoding mask_check from an image path."
        ),
    )
    parser.add_argument(
        "--legacy_only_without_mask_sam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When converting legacy cache samples, only keep entries whose legacy payload used_mask_sam is False.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/data/nas_hdd/CORNE_extracted/cache/singleturn_object_removal_wan2.1_1.3b_sam_strict_keyframe_cache_v1__legacy_without_mask_sam_30000",
        help="Directory to write cached tensors and manifest.",
    )
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
        raise ValueError("CORNE object-removal preprocess no longer supports --train_data_manifest.")
    if args.num_workers != 0:
        raise ValueError("CORNE object-removal preprocess requires --num_workers 0 to preserve class quotas exactly.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if (
        args.max_samples_with_mask_sam < 0
        or args.max_samples_without_mask_sam < 0
        or args.skip_samples_with_mask_sam < 0
        or args.skip_samples_without_mask_sam < 0
    ):
        raise ValueError("Quota and skip arguments must be non-negative.")
    if args.train_data_dir is None and args.source_cached_data_meta is None:
        raise ValueError("Provide either --train_data_dir for raw preprocessing or --source_cached_data_meta for cache conversion.")
    if args.train_data_dir is not None and args.source_cached_data_meta is not None:
        raise ValueError("Choose either raw-image preprocessing (--train_data_dir) or cache conversion (--source_cached_data_meta), not both.")
    if args.source_cached_data_dir is not None and args.source_cached_data_meta is None:
        raise ValueError("--source_cached_data_dir requires --source_cached_data_meta.")

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
        "mode": SINGLETURN_OBJECT_REMOVAL_CACHE_MODE,
        "source_image": sample["source_image"],
        "bg_image": sample["bg_image"],
        "mask_check_image": sample["mask_check_image"],
        "mask_frame_image": sample["mask_frame_image"],
        "used_mask_sam": bool(sample["used_mask_sam"]),
        "global_index": int(sample["global_index"]),
    }
    mask_sam_image = sample.get("mask_sam_image", "")
    if mask_sam_image:
        entry["mask_sam_image"] = mask_sam_image
    return entry


LEGACY_NOISY_ANCHOR_FRAME_INDEX_BY_TOTAL_FRAMES = {
    8: 4,
    11: 4,
    21: 7,
}


def _resolve_source_cache_root(source_cached_data_meta: str, source_cached_data_dir: str | None) -> Path:
    if source_cached_data_dir is not None:
        return Path(source_cached_data_dir).resolve()
    return Path(source_cached_data_meta).resolve().parent


def _resolve_source_cache_path(cache_path: str, source_root: Path) -> Path:
    resolved = Path(cache_path)
    if resolved.is_absolute():
        return resolved
    return (source_root / resolved).resolve()


def _parse_global_index(value: object, fallback_name: str) -> int:
    if value is not None:
        return int(value)
    prefix = str(fallback_name).split("_", 1)[0]
    return int(prefix)


def _resolve_shared_prompt_cache_source(source_root: Path, source_metadata: dict | None, payload: dict) -> Path | None:
    candidate = None
    if source_metadata is not None:
        candidate = source_metadata.get("shared_prompt_cache")
    if not candidate:
        candidate = payload.get("shared_prompt_cache")
    if not candidate:
        return None
    candidate_path = Path(str(candidate))
    if candidate_path.is_absolute():
        return candidate_path
    return (source_root / candidate_path).resolve()


def _extract_legacy_keyframe_latents(
    payload: dict,
    *,
    duplicate_mask_frame_for_both_masks: bool = False,
) -> dict[str, torch.Tensor]:
    full_latents = payload.get("full_latents")
    if not torch.is_tensor(full_latents) or full_latents.ndim != 4:
        raise ValueError(
            "Legacy cache payload must contain full_latents with shape (C, T, H, W), "
            f"got {type(full_latents)} / {getattr(full_latents, 'shape', None)}"
        )
    total_frames = int(full_latents.shape[1])
    noisy_anchor_frame_index = LEGACY_NOISY_ANCHOR_FRAME_INDEX_BY_TOTAL_FRAMES.get(total_frames)
    if noisy_anchor_frame_index is None:
        raise ValueError(
            "Unsupported legacy full_latents length for cache conversion. "
            f"Expected one of {sorted(LEGACY_NOISY_ANCHOR_FRAME_INDEX_BY_TOTAL_FRAMES)}, got {total_frames}."
        )

    mask_sam_latent = payload.get("mask_frame_latent", full_latents[:, 0:1])
    source_frame_latent = payload.get("source_frame_latent", full_latents[:, 1:2])

    if not torch.is_tensor(mask_sam_latent) or mask_sam_latent.ndim != 4:
        raise ValueError(
            "Legacy cache payload must contain mask_frame_latent with shape (C, 1, H, W), "
            f"got {type(mask_sam_latent)} / {getattr(mask_sam_latent, 'shape', None)}"
        )
    if not torch.is_tensor(source_frame_latent) or source_frame_latent.ndim != 4:
        raise ValueError(
            "Legacy cache payload must contain source_frame_latent with shape (C, 1, H, W), "
            f"got {type(source_frame_latent)} / {getattr(source_frame_latent, 'shape', None)}"
        )

    result = {
        "mask_sam_latent": mask_sam_latent.detach().cpu(),
        "source_frame_latent": source_frame_latent.detach().cpu(),
        "noisy_anchor_latent": full_latents[:, noisy_anchor_frame_index : noisy_anchor_frame_index + 1].detach().cpu(),
        "target_latent": full_latents[:, -1:].detach().cpu(),
    }
    if duplicate_mask_frame_for_both_masks:
        result["mask_check_latent"] = mask_sam_latent.detach().cpu()
    return result


def _build_conversion_record(
    *,
    entry: dict,
    payload: dict,
    source_root: Path,
    expected_sample_size: tuple[int, int],
    duplicate_mask_frame_for_both_masks: bool,
) -> dict:
    sample_size = normalize_singleturn_sample_size(payload.get("singleturn_sample_size", expected_sample_size))
    if sample_size != expected_sample_size:
        raise ValueError(
            "Legacy cache sample size does not match requested singleturn_sample_size. "
            f"Got legacy={sample_size}, requested={expected_sample_size}."
        )

    mask_frame_image = payload.get("mask_frame_image", entry.get("mask_frame_image", ""))
    mask_check_image = payload.get("mask_check_image", entry.get("mask_check_image", ""))
    mask_sam_image = payload.get("mask_sam_image", entry.get("mask_sam_image", ""))
    if duplicate_mask_frame_for_both_masks:
        if not mask_frame_image:
            raise ValueError("Legacy cache sample is missing mask_frame_image.")
        mask_sam_image = mask_frame_image
        mask_check_image = mask_frame_image
    else:
        if not mask_check_image:
            raise ValueError("Legacy cache sample is missing mask_check_image.")
        if not mask_sam_image:
            raise ValueError("Legacy cache sample is missing mask_sam_image.")

    source_image = payload.get("source_image", entry.get("source_image", ""))
    bg_image = payload.get("bg_image", entry.get("bg_image", ""))
    global_index = _parse_global_index(entry.get("global_index"), Path(str(entry.get("cache_path", ""))).name)
    latents = _extract_legacy_keyframe_latents(
        payload,
        duplicate_mask_frame_for_both_masks=duplicate_mask_frame_for_both_masks,
    )

    record = {
        "mask_sam_latent": latents["mask_sam_latent"],
        "source_frame_latent": latents["source_frame_latent"],
        "noisy_anchor_latent": latents["noisy_anchor_latent"],
        "target_latent": latents["target_latent"],
        "source_image": source_image,
        "bg_image": bg_image,
        "mask_check_image": mask_check_image,
        "mask_sam_image": mask_sam_image,
        "mask_frame_image": mask_frame_image or mask_sam_image,
        "used_mask_sam": bool(payload.get("used_mask_sam", entry.get("used_mask_sam", False))),
        "global_index": global_index,
    }
    if duplicate_mask_frame_for_both_masks:
        record["mask_check_latent"] = latents["mask_check_latent"]
    else:
        record["pixel_values_mask_check_frame"] = preprocess_singleturn_mask_frame(
            mask_check_image,
            sample_size,
            add_batch_dim=False,
            add_frame_dim=True,
        )
    return record


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
    mask_sam_batch = _stack_batch_tensors(batch_records, "pixel_values_mask_frame").to(
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
        mask_sam_latents = vae.encode(mask_sam_batch.permute(0, 2, 1, 3, 4))[0].mode()
        mask_check_latents = vae.encode(mask_check_frame_batch.permute(0, 2, 1, 3, 4))[0].mode()
        source_frame_latents = vae.encode(source_batch.permute(0, 2, 1, 3, 4))[0].mode()
        noisy_anchor_latents = build_singleturn_noisy_anchor_latents(
            source_frame_latents,
            mask_check_batch,
        )
        target_latents = vae.encode(bg_batch.permute(0, 2, 1, 3, 4))[0].mode()

    for local_offset, record in enumerate(batch_records):
        cache_path = cache_dir / build_cache_name(int(record["global_index"]), record["source_image"])
        if cache_path.exists() and not overwrite:
            raise FileExistsError(f"Cache file already exists: {cache_path}. Use --overwrite to replace it.")

        payload = {
            "mode": SINGLETURN_OBJECT_REMOVAL_CACHE_MODE,
            "dataset_type": "corne_object_removal",
            "mask_sam_latent": mask_sam_latents[local_offset].detach().cpu().to(weight_dtype),
            "mask_check_latent": mask_check_latents[local_offset].detach().cpu().to(weight_dtype),
            "source_frame_latent": source_frame_latents[local_offset].detach().cpu().to(weight_dtype),
            "noisy_anchor_latent": noisy_anchor_latents[local_offset].detach().cpu().to(weight_dtype),
            "target_latent": target_latents[local_offset].detach().cpu().to(weight_dtype),
            "source_image": record["source_image"],
            "bg_image": record["bg_image"],
            "mask_check_image": record["mask_check_image"],
            "mask_sam_image": record["mask_sam_image"],
            "used_mask_sam": bool(record["used_mask_sam"]),
            "singleturn_sample_size": list(singleturn_sample_size),
            "shared_prompt_cache": str(shared_prompt_path.relative_to(output_dir)),
            "text": CORNE_SINGLETURN_PROMPT,
            "formatted_text": CORNE_SINGLETURN_PROMPT,
        }

        torch.save(payload, cache_path)
        manifest.append(build_manifest_entry(cache_path, output_dir, record))

    return len(batch_records)


def _flush_conversion_batch(
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

    if "mask_check_latent" in batch_records[0]:
        mask_check_latents = _stack_batch_tensors(batch_records, "mask_check_latent").to(
            device=device,
            dtype=weight_dtype,
            non_blocking=True,
        )
    else:
        mask_check_frame_batch = _stack_batch_tensors(batch_records, "pixel_values_mask_check_frame").to(
            device=device,
            dtype=weight_dtype,
            non_blocking=True,
        )
        with torch.no_grad():
            mask_check_latents = vae.encode(mask_check_frame_batch.permute(0, 2, 1, 3, 4))[0].mode()

    for local_offset, record in enumerate(batch_records):
        cache_path = cache_dir / build_cache_name(int(record["global_index"]), record["source_image"])
        if cache_path.exists() and not overwrite:
            raise FileExistsError(f"Cache file already exists: {cache_path}. Use --overwrite to replace it.")

        payload = {
            "mode": SINGLETURN_OBJECT_REMOVAL_CACHE_MODE,
            "dataset_type": "corne_object_removal",
            "mask_sam_latent": record["mask_sam_latent"].to(dtype=weight_dtype),
            "mask_check_latent": mask_check_latents[local_offset].detach().cpu().to(weight_dtype),
            "source_frame_latent": record["source_frame_latent"].to(dtype=weight_dtype),
            "noisy_anchor_latent": record["noisy_anchor_latent"].to(dtype=weight_dtype),
            "target_latent": record["target_latent"].to(dtype=weight_dtype),
            "source_image": record["source_image"],
            "bg_image": record["bg_image"],
            "mask_check_image": record["mask_check_image"],
            "mask_sam_image": record["mask_sam_image"],
            "used_mask_sam": bool(record["used_mask_sam"]),
            "singleturn_sample_size": list(singleturn_sample_size),
            "shared_prompt_cache": str(shared_prompt_path.relative_to(output_dir)),
            "text": CORNE_SINGLETURN_PROMPT,
            "formatted_text": CORNE_SINGLETURN_PROMPT,
        }

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

    conversion_mode = args.source_cached_data_meta is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = get_weight_dtype(args.dtype, device)
    config = OmegaConf.load(args.config_path)

    source_root = None
    source_metadata = None
    source_manifest = None
    shared_prompt_text = CORNE_SINGLETURN_PROMPT
    shared_prompt_cache_source = None
    need_prompt_reencode = not conversion_mode

    if conversion_mode:
        source_manifest_path = Path(args.source_cached_data_meta).resolve()
        with open(source_manifest_path, "r", encoding="utf-8") as f:
            source_manifest = json.load(f)
        if not isinstance(source_manifest, list) or not source_manifest:
            raise ValueError("Legacy cache manifest must be a non-empty list.")

        source_root = _resolve_source_cache_root(args.source_cached_data_meta, args.source_cached_data_dir)
        source_metadata_path = source_root / "metadata.json"
        if source_metadata_path.is_file():
            with open(source_metadata_path, "r", encoding="utf-8") as f:
                source_metadata = json.load(f)

        first_cache_path = _resolve_source_cache_path(source_manifest[0]["cache_path"], source_root)
        first_payload = load_singleturn_cache_payload(str(first_cache_path))
        shared_prompt_cache_source = _resolve_shared_prompt_cache_source(source_root, source_metadata, first_payload)
        if shared_prompt_cache_source is not None and shared_prompt_cache_source.is_file():
            shutil.copy2(shared_prompt_cache_source, shared_prompt_path)
            copied_prompt_payload = load_singleturn_cache_payload(str(shared_prompt_path))
            shared_prompt_text = copied_prompt_payload.get("text", CORNE_SINGLETURN_PROMPT)
            need_prompt_reencode = False

    if need_prompt_reencode:
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
                "conditioning_format": "shared_singleturn_prompt_cache",
            },
            shared_prompt_path,
        )

    vae = AutoencoderKLWan.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
            "vae",
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).eval().to(device, dtype=weight_dtype)

    manifest: list[dict] = []
    batch_records: list[dict] = []
    generated_samples = 0

    if conversion_mode:
        dropped_missing_mask_sam = 0
        dropped_missing_mask_check = 0
        skipped_unsupported_source_cache = 0
        skipped_bad_sample_size = 0
        selected_with_mask_sam = 0
        remaining_skip_with_mask_sam = int(args.skip_samples_with_mask_sam)
        stopped_early_when_quotas_met = False
        progress_bar = tqdm(source_manifest, desc="Converting legacy SingleTurn cache")

        for entry in progress_bar:
            cache_path = _resolve_source_cache_path(entry["cache_path"], source_root)
            payload = load_singleturn_cache_payload(str(cache_path))
            legacy_used_mask_sam = bool(payload.get("used_mask_sam", entry.get("used_mask_sam", False)))

            if args.legacy_only_without_mask_sam and legacy_used_mask_sam:
                continue

            mask_frame_image = payload.get("mask_frame_image", entry.get("mask_frame_image", ""))
            mask_sam_image = payload.get("mask_sam_image", entry.get("mask_sam_image", ""))
            mask_check_image = payload.get("mask_check_image", entry.get("mask_check_image", ""))
            if args.legacy_duplicate_mask_frame_for_both_masks:
                if not mask_frame_image:
                    dropped_missing_mask_sam += 1
                    progress_bar.set_postfix(
                        generated=generated_samples,
                        dropped_missing_mask_sam=dropped_missing_mask_sam,
                        dropped_missing_mask_check=dropped_missing_mask_check,
                    )
                    continue
            else:
                if not mask_sam_image:
                    dropped_missing_mask_sam += 1
                    progress_bar.set_postfix(
                        generated=generated_samples,
                        dropped_missing_mask_sam=dropped_missing_mask_sam,
                        dropped_missing_mask_check=dropped_missing_mask_check,
                    )
                    continue
                if not mask_check_image:
                    dropped_missing_mask_check += 1
                    progress_bar.set_postfix(
                        generated=generated_samples,
                        dropped_missing_mask_sam=dropped_missing_mask_sam,
                        dropped_missing_mask_check=dropped_missing_mask_check,
                    )
                    continue

            if remaining_skip_with_mask_sam > 0:
                remaining_skip_with_mask_sam -= 1
                continue
            if selected_with_mask_sam >= args.max_samples_with_mask_sam:
                stopped_early_when_quotas_met = True
                break

            try:
                record = _build_conversion_record(
                    entry=entry,
                    payload=payload,
                    source_root=source_root,
                    expected_sample_size=args.singleturn_sample_size,
                    duplicate_mask_frame_for_both_masks=args.legacy_duplicate_mask_frame_for_both_masks,
                )
            except ValueError as exc:
                if "sample size does not match" in str(exc):
                    skipped_bad_sample_size += 1
                else:
                    skipped_unsupported_source_cache += 1
                progress_bar.set_postfix(
                    generated=generated_samples,
                    skipped_unsupported_source_cache=skipped_unsupported_source_cache,
                    skipped_bad_sample_size=skipped_bad_sample_size,
                )
                continue

            selected_with_mask_sam += 1
            batch_records.append(record)
            if len(batch_records) < args.batch_size:
                continue

            generated_samples += _flush_conversion_batch(
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
            progress_bar.set_postfix(
                generated=generated_samples,
                dropped_missing_mask_sam=dropped_missing_mask_sam,
                dropped_missing_mask_check=dropped_missing_mask_check,
                skipped_unsupported_source_cache=skipped_unsupported_source_cache,
            )

        if batch_records:
            generated_samples += _flush_conversion_batch(
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
        progress_bar.close()
        num_samples_with_mask_sam = generated_samples
        num_samples_without_mask_sam = 0
    else:
        preprocess_dataset = SingleTurnPreprocessIterableDataset(
            data_root=args.train_data_dir,
            sample_size=args.singleturn_sample_size,
            max_samples_with_mask_sam=args.max_samples_with_mask_sam,
            max_samples_without_mask_sam=args.max_samples_without_mask_sam,
            skip_samples_with_mask_sam=args.skip_samples_with_mask_sam,
            skip_samples_without_mask_sam=args.skip_samples_without_mask_sam,
        )

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
                dropped_missing_mask_sam=preprocess_dataset.dropped_missing_mask_sam,
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
        dropped_missing_mask_sam = preprocess_dataset.dropped_missing_mask_sam
        dropped_missing_mask_check = preprocess_dataset.dropped_missing_mask_check
        skipped_unsupported_source_cache = 0
        skipped_bad_sample_size = 0
        stopped_early_when_quotas_met = bool(preprocess_dataset.stopped_early_when_quotas_met)
        num_samples_with_mask_sam = preprocess_dataset.num_samples_with_mask_sam
        num_samples_without_mask_sam = preprocess_dataset.num_samples_without_mask_sam

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    metadata = {
        "mode": SINGLETURN_OBJECT_REMOVAL_CACHE_MODE,
        "dataset_type": "corne_object_removal",
        "conditioning_format": "five_keyframe_shared_prompt_cache",
        "pixel_space_source_masking": not conversion_mode,
        "cache_build_mode": "legacy_cache_conversion" if conversion_mode else "raw_image_precompute",
        "legacy_mask_strategy": (
            "duplicate_mask_frame_for_both_masks"
            if conversion_mode and args.legacy_duplicate_mask_frame_for_both_masks
            else "encode_mask_check_from_image"
        ),
        "legacy_only_without_mask_sam": bool(args.legacy_only_without_mask_sam),
        "max_samples_with_mask_sam": args.max_samples_with_mask_sam,
        "max_samples_without_mask_sam": args.max_samples_without_mask_sam,
        "skip_samples_with_mask_sam": args.skip_samples_with_mask_sam,
        "skip_samples_without_mask_sam": args.skip_samples_without_mask_sam,
        "num_samples_with_mask_sam": num_samples_with_mask_sam,
        "num_samples_without_mask_sam": num_samples_without_mask_sam,
        "dropped_missing_mask_sam": dropped_missing_mask_sam,
        "dropped_missing_mask_check": dropped_missing_mask_check,
        "skipped_unsupported_source_cache": skipped_unsupported_source_cache,
        "skipped_bad_sample_size": skipped_bad_sample_size,
        "stopped_early_when_quotas_met": stopped_early_when_quotas_met,
        "kept_sample_count": generated_samples,
        "num_samples_total": generated_samples,
        "singleturn_sample_size": list(args.singleturn_sample_size),
        "shared_prompt_text": shared_prompt_text,
        "shared_prompt_cache": str(shared_prompt_path.relative_to(output_dir)),
        "manifest_path": str(manifest_path.relative_to(output_dir)),
    }
    if conversion_mode:
        metadata["source_cached_data_meta"] = str(Path(args.source_cached_data_meta).resolve())
        metadata["source_cached_data_dir"] = str(source_root)
        metadata["source_shared_prompt_cache"] = (
            str(shared_prompt_cache_source) if shared_prompt_cache_source is not None else ""
        )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "metadata_path": str(metadata_path),
                "shared_prompt_cache": str(shared_prompt_path),
                "num_samples_total": generated_samples,
                "num_samples_with_mask_sam": num_samples_with_mask_sam,
                "dropped_missing_mask_sam": dropped_missing_mask_sam,
                "dropped_missing_mask_check": dropped_missing_mask_check,
                "skipped_unsupported_source_cache": skipped_unsupported_source_cache,
                "skipped_bad_sample_size": skipped_bad_sample_size,
                "skip_samples_with_mask_sam": args.skip_samples_with_mask_sam,
                "skip_samples_without_mask_sam": args.skip_samples_without_mask_sam,
                "cache_build_mode": metadata["cache_build_mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
