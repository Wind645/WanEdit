import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from videox_fun.utils.singleturn_utils import (
    CORNE_SINGLETURN_PROMPT,
    SINGLETURN_OBJECT_REMOVAL_CACHE_MODE,
    is_supported_singleturn_object_removal_mode,
    normalize_singleturn_sample_size,
    preprocess_singleturn_image,
    preprocess_singleturn_mask_frame,
    preprocess_singleturn_mask,
    SINGLETURN_TOTAL_FRAMES,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def resolve_singleturn_path(data_root: str, candidate: str) -> str:
    if os.path.isabs(candidate):
        return candidate
    return os.path.join(data_root, candidate)


def load_singleturn_image(image_data: Any, data_root: Optional[str] = None) -> Image.Image:
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB")

    if isinstance(image_data, dict):
        payload = image_data.get("bytes")
        if isinstance(payload, bytearray):
            payload = bytes(payload)
        if isinstance(payload, bytes):
            return Image.open(io.BytesIO(payload)).convert("RGB")

        path = image_data.get("path")
        if isinstance(path, str) and path:
            return load_singleturn_image(path, data_root)

        raise ValueError("Image record dict must contain either raw bytes or a path.")

    if isinstance(image_data, bytearray):
        image_data = bytes(image_data)
    if isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    if isinstance(image_data, str):
        path = resolve_singleturn_path(data_root, image_data) if data_root is not None else image_data
        return Image.open(path).convert("RGB")

    raise ValueError(f"Unsupported SingleTurn image payload type: {type(image_data)}")


def load_singleturn_prompt(record: Dict[str, str], data_root: str) -> str:
    del record
    del data_root
    return CORNE_SINGLETURN_PROMPT


def has_singleturn_parquet_data(data_root: str) -> bool:
    del data_root
    return False


def iter_singleturn_parquet_records(data_root: str, batch_size: int = 1) -> Iterator[Dict[str, Any]]:
    del batch_size
    raise ValueError(
        "Parquet-based SingleTurn preprocessing is no longer supported. "
        "Use a CORNE object-removal root with shot/, bg/, mask-check/, and optional mask_sam/."
    )


def _load_cache_payload(cache_path: str):
    try:
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(cache_path, map_location="cpu")


def load_singleturn_cache_payload(cache_path: str):
    return _load_cache_payload(cache_path)


def _iter_image_files(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Expected directory does not exist: {directory}")
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _resolve_companion_file(directory: Path, stem: str) -> Optional[Path]:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _corne_dirs(data_root: str) -> tuple[Path, Path, Path, Path]:
    root = Path(data_root)
    shot_dir = root / "shot"
    bg_dir = root / "bg"
    mask_check_dir = root / "mask-check"
    mask_sam_dir = root / "mask_sam"
    if not shot_dir.is_dir() or not bg_dir.is_dir() or not mask_check_dir.is_dir():
        raise ValueError(
            "SingleTurn CORNE object-removal data root must contain shot/, bg/, and mask-check/ directories. "
            f"Got data_root={data_root}"
        )
    return shot_dir, bg_dir, mask_check_dir, mask_sam_dir


def _build_corne_record(
    shot_path: Path,
    bg_dir: Path,
    mask_check_dir: Path,
    mask_sam_dir: Path,
) -> Optional[Dict[str, Any]]:
    stem = shot_path.stem
    bg_path = _resolve_companion_file(bg_dir, stem)
    mask_check_path = _resolve_companion_file(mask_check_dir, stem)
    if bg_path is None or mask_check_path is None:
        return None

    mask_sam_path = _resolve_companion_file(mask_sam_dir, stem) if mask_sam_dir.is_dir() else None
    if mask_sam_path is None:
        return None
    used_mask_sam = True
    return {
        "source_image": str(shot_path),
        "bg_image": str(bg_path),
        "mask_check_image": str(mask_check_path),
        "mask_sam_image": str(mask_sam_path),
        "used_mask_sam": used_mask_sam,
        "mask_frame_image": str(mask_sam_path),
        "type": "image",
        "file_path": str(shot_path),
    }


def inspect_singleturn_corne_records(data_root: str) -> Dict[str, Any]:
    shot_dir, bg_dir, mask_check_dir, mask_sam_dir = _corne_dirs(data_root)
    records: List[Dict[str, Any]] = []
    dropped_missing_mask_sam = 0
    dropped_missing_mask_check = 0

    for shot_path in _iter_image_files(shot_dir):
        stem = shot_path.stem
        bg_path = _resolve_companion_file(bg_dir, stem)
        if bg_path is None:
            continue
        mask_check_path = _resolve_companion_file(mask_check_dir, stem)
        if mask_check_path is None:
            dropped_missing_mask_check += 1
            continue
        mask_sam_path = _resolve_companion_file(mask_sam_dir, stem) if mask_sam_dir.is_dir() else None
        if mask_sam_path is None:
            dropped_missing_mask_sam += 1
            continue
        records.append(
            {
                "source_image": str(shot_path),
                "bg_image": str(bg_path),
                "mask_check_image": str(mask_check_path),
                "mask_sam_image": str(mask_sam_path),
                "used_mask_sam": True,
                "mask_frame_image": str(mask_sam_path),
                "type": "image",
                "file_path": str(shot_path),
            }
        )

    return {
        "records": records,
        "dropped_missing_mask_sam": dropped_missing_mask_sam,
        "dropped_missing_mask_check": dropped_missing_mask_check,
    }


def discover_singleturn_records(
    data_root: str,
    manifest_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if manifest_path is not None:
        raise ValueError(
            "SingleTurn CORNE object-removal mode no longer uses manifest-based source/edited/prompt records. "
            "Point --train_data_dir at the CORNE root instead."
        )

    inspection = inspect_singleturn_corne_records(data_root)
    records = inspection["records"]

    if not records:
        raise ValueError(
            "No valid CORNE object-removal samples were found. Each sample basename must exist in shot/, bg/, mask-check/, and mask_sam/."
        )
    return records


class SingleTurnEditDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        manifest_path: Optional[str] = None,
        video_sample_size=512,
        text_drop_ratio: float = 0.0,
    ):
        del text_drop_ratio
        self.data_root = data_root
        self.dataset = discover_singleturn_records(data_root=data_root, manifest_path=manifest_path)
        self.length = len(self.dataset)
        self.video_sample_size = normalize_singleturn_sample_size(video_sample_size)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        record = self.dataset[idx % self.length]
        return {
            "pixel_values_src_image": preprocess_singleturn_image(
                record["source_image"],
                self.video_sample_size,
                add_batch_dim=False,
                add_frame_dim=True,
            ),
            "pixel_values_mask_frame": preprocess_singleturn_mask_frame(
                record["mask_frame_image"],
                self.video_sample_size,
                add_batch_dim=False,
                add_frame_dim=True,
            ),
            "pixel_values_mask_check_frame": preprocess_singleturn_mask_frame(
                record["mask_check_image"],
                self.video_sample_size,
                add_batch_dim=False,
                add_frame_dim=True,
            ),
            "pixel_values_tgt_image": preprocess_singleturn_image(
                record["bg_image"],
                self.video_sample_size,
                add_batch_dim=False,
                add_frame_dim=True,
            ),
            "pixel_values_mask_check": preprocess_singleturn_mask(
                record["mask_check_image"],
                self.video_sample_size,
                add_batch_dim=False,
                add_frame_dim=True,
            ),
            "text": CORNE_SINGLETURN_PROMPT,
            "source_image": record["source_image"],
            "bg_image": record["bg_image"],
            "mask_check_image": record["mask_check_image"],
            "mask_frame_image": record["mask_frame_image"],
            "mask_sam_image": record["mask_sam_image"] or "",
            "used_mask_sam": bool(record["used_mask_sam"]),
            "data_type": "image",
            "idx": idx,
        }


class SingleTurnPreprocessIterableDataset(IterableDataset):
    def __init__(
        self,
        data_root: str,
        manifest_path: Optional[str] = None,
        sample_size=512,
        parquet_batch_size: int = 32,
        reconstruction_only: bool = False,
        max_samples_with_mask_sam: int = 30000,
        max_samples_without_mask_sam: int = 30000,
        skip_samples_with_mask_sam: int = 0,
        skip_samples_without_mask_sam: int = 0,
    ):
        del manifest_path
        del parquet_batch_size
        del reconstruction_only
        self.data_root = data_root
        self.sample_size = normalize_singleturn_sample_size(sample_size)
        self.max_samples_with_mask_sam = int(max_samples_with_mask_sam)
        self.max_samples_without_mask_sam = int(max_samples_without_mask_sam)
        self.skip_samples_with_mask_sam = int(skip_samples_with_mask_sam)
        self.skip_samples_without_mask_sam = int(skip_samples_without_mask_sam)
        self.use_parquet = False
        self.records = None
        self.total_length = None
        self.num_samples_with_mask_sam = 0
        self.num_samples_without_mask_sam = 0
        self.dropped_missing_mask_sam = 0
        self.dropped_missing_mask_check = 0
        self.stopped_early_when_quotas_met = False
        inspection = inspect_singleturn_corne_records(data_root)
        self.records = inspection["records"]
        self.dropped_missing_mask_sam = int(inspection["dropped_missing_mask_sam"])
        self.dropped_missing_mask_check = int(inspection["dropped_missing_mask_check"])

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError(
                "SingleTurn CORNE preprocess requires sequential iteration to preserve per-class quotas. "
                "Run preprocess with --num_workers 0."
            )
        if self.skip_samples_with_mask_sam < 0 or self.skip_samples_without_mask_sam < 0:
            raise ValueError("SingleTurn skip quotas must be non-negative.")

        self.num_samples_with_mask_sam = 0
        self.num_samples_without_mask_sam = 0
        self.stopped_early_when_quotas_met = False
        remaining_skip_with_mask_sam = self.skip_samples_with_mask_sam
        remaining_skip_without_mask_sam = self.skip_samples_without_mask_sam

        for global_index, record in enumerate(self.records):
            if (
                self.num_samples_with_mask_sam >= self.max_samples_with_mask_sam
                and self.num_samples_without_mask_sam >= self.max_samples_without_mask_sam
            ):
                self.stopped_early_when_quotas_met = True
                break

            used_mask_sam = True
            if used_mask_sam:
                if remaining_skip_with_mask_sam > 0:
                    remaining_skip_with_mask_sam -= 1
                    continue
                if self.num_samples_with_mask_sam >= self.max_samples_with_mask_sam:
                    continue
                self.num_samples_with_mask_sam += 1
            else:
                if remaining_skip_without_mask_sam > 0:
                    remaining_skip_without_mask_sam -= 1
                    continue
                if self.num_samples_without_mask_sam >= self.max_samples_without_mask_sam:
                    continue
                self.num_samples_without_mask_sam += 1

            yield {
                "pixel_values_src_image": preprocess_singleturn_image(
                    record["source_image"],
                    self.sample_size,
                    add_batch_dim=False,
                    add_frame_dim=True,
                ),
                "pixel_values_mask_frame": preprocess_singleturn_mask_frame(
                    record["mask_frame_image"],
                    self.sample_size,
                    add_batch_dim=False,
                    add_frame_dim=True,
                ),
                "pixel_values_mask_check_frame": preprocess_singleturn_mask_frame(
                    record["mask_check_image"],
                    self.sample_size,
                    add_batch_dim=False,
                    add_frame_dim=True,
                ),
                "pixel_values_tgt_image": preprocess_singleturn_image(
                    record["bg_image"],
                    self.sample_size,
                    add_batch_dim=False,
                    add_frame_dim=True,
                ),
                "pixel_values_mask_check": preprocess_singleturn_mask(
                    record["mask_check_image"],
                    self.sample_size,
                    add_batch_dim=False,
                    add_frame_dim=True,
                ),
                "text": CORNE_SINGLETURN_PROMPT,
                "source_image": record["source_image"],
                "bg_image": record["bg_image"],
                "mask_check_image": record["mask_check_image"],
                "mask_frame_image": record["mask_frame_image"],
                "mask_sam_image": record["mask_sam_image"] or "",
                "used_mask_sam": used_mask_sam,
                "global_index": global_index,
                "row_index": -1,
            }


class CachedSingleTurnLatentDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        data_root: Optional[str] = None,
        expected_mode: str = "singleturn_object_removal_cached",
    ):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("SingleTurn cached manifest must be a non-empty list.")
        self.data_root = data_root
        self.manifest = manifest
        self.expected_mode = expected_mode
        self._shared_prompt_cache_payloads: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def _resolve_cache_path(self, cache_path: str) -> str:
        if self.data_root is not None and not os.path.isabs(cache_path):
            return os.path.join(self.data_root, cache_path)
        return cache_path

    def _load_shared_prompt_cache(self, prompt_cache_path: str) -> Dict[str, Any]:
        resolved_path = self._resolve_cache_path(prompt_cache_path)
        cached = self._shared_prompt_cache_payloads.get(resolved_path)
        if cached is not None:
            return cached

        payload = load_singleturn_cache_payload(resolved_path)
        missing = [key for key in ("prompt_embeds", "prompt_seq_len") if key not in payload]
        if missing:
            raise ValueError(f"SingleTurn shared prompt cache {resolved_path} is missing keys: {missing}")
        self._shared_prompt_cache_payloads[resolved_path] = payload
        return payload

    def __getitem__(self, index: int):
        entry = self.manifest[index]
        cache_path = self._resolve_cache_path(entry["cache_path"])
        payload = load_singleturn_cache_payload(cache_path)

        if "source_latent_mean" in payload or "target_latent_mean" in payload:
            raise ValueError(
                f"Cached SingleTurn sample {cache_path} uses the deprecated instructpix2pix posterior payload. "
                "Re-run preprocess_singleturn_cache.py to generate CORNE object-removal caches."
            )

        payload_mode = payload.get("mode")
        if self.expected_mode == "singleturn_object_removal_cached":
            mode_matches = payload_mode == SINGLETURN_OBJECT_REMOVAL_CACHE_MODE
        else:
            mode_matches = payload_mode == self.expected_mode
        if not mode_matches:
            raise ValueError(
                f"Cached SingleTurn sample {cache_path} has unsupported mode={payload_mode!r}. "
                f"Expected mode={self.expected_mode!r}."
            )

        prompt_embeds = payload.get("prompt_embeds")
        prompt_seq_len = payload.get("prompt_seq_len")
        prompt_text = payload.get("text", CORNE_SINGLETURN_PROMPT)
        formatted_text = payload.get("formatted_text", prompt_text)
        shared_prompt_cache = payload.get("shared_prompt_cache")
        if shared_prompt_cache is not None:
            shared_payload = self._load_shared_prompt_cache(shared_prompt_cache)
            prompt_embeds = shared_payload["prompt_embeds"]
            prompt_seq_len = int(shared_payload["prompt_seq_len"])
            prompt_text = shared_payload.get("text", prompt_text)
            formatted_text = shared_payload.get("formatted_text", prompt_text)

        if prompt_embeds is None or prompt_seq_len is None:
            raise ValueError(
                f"Cached SingleTurn sample {cache_path} must contain prompt_embeds/prompt_seq_len or shared_prompt_cache."
            )

        sample = {
            "prompt_embeds": prompt_embeds,
            "prompt_seq_len": int(prompt_seq_len),
            "text": prompt_text,
            "formatted_text": formatted_text,
            "cache_path": cache_path,
            "source_image": entry.get("source_image", payload.get("source_image", "")),
            "bg_image": entry.get("bg_image", payload.get("bg_image", "")),
            "mask_check_image": entry.get("mask_check_image", payload.get("mask_check_image", "")),
            "mask_frame_image": entry.get("mask_frame_image", payload.get("mask_frame_image", "")),
            "mask_sam_image": entry.get("mask_sam_image", payload.get("mask_sam_image", "")),
            "used_mask_sam": bool(entry.get("used_mask_sam", payload.get("used_mask_sam", False))),
            "row_index": int(payload.get("row_index", entry.get("row_index", -1)) or -1),
            "global_index": int(entry.get("global_index", index)),
            "data_type": "image",
            "idx": index,
        }
        if self.expected_mode == "singleturn_object_removal_cached" or self.expected_mode == SINGLETURN_OBJECT_REMOVAL_CACHE_MODE:
            missing = [
                key
                for key in (
                    "mask_sam_latent",
                    "mask_check_latent",
                    "source_frame_latent",
                    "noisy_anchor_latent",
                    "target_latent",
                )
                if key not in payload
            ]
            if missing:
                raise ValueError(f"Cached SingleTurn sample {cache_path} is missing keys: {missing}")
            sample.update(
                {
                    "mask_sam_latent": payload["mask_sam_latent"],
                    "mask_check_latent": payload["mask_check_latent"],
                    "source_frame_latent": payload["source_frame_latent"],
                    "noisy_anchor_latent": payload["noisy_anchor_latent"],
                    "target_latent": payload["target_latent"],
                    "singleturn_sample_size": payload.get("singleturn_sample_size", entry.get("singleturn_sample_size", [])),
                    "cache_mode": payload_mode,
                }
            )
            return sample

        if self.expected_mode == "singleturn_object_removal_refine_v1":
            missing = [key for key in ("input_latents", "target_latents", "refinement_loss_weight_map") if key not in payload]
            if missing:
                raise ValueError(f"Cached SingleTurn refine sample {cache_path} is missing keys: {missing}")
            if payload["input_latents"].shape[-3] != SINGLETURN_TOTAL_FRAMES:
                raise ValueError(
                    f"Cached SingleTurn refine sample {cache_path} has {payload['input_latents'].shape[-3]} frames; "
                    f"expected {SINGLETURN_TOTAL_FRAMES}."
                )
            if payload["target_latents"].shape != payload["input_latents"].shape:
                raise ValueError(
                    "Cached SingleTurn refine sample must store matching input_latents and target_latents shapes, "
                    f"got {tuple(payload['input_latents'].shape)} and {tuple(payload['target_latents'].shape)}."
                )
            sample.update(
                {
                    "input_latents": payload["input_latents"],
                    "target_latents": payload["target_latents"],
                    "refinement_loss_weight_map": payload["refinement_loss_weight_map"],
                    "coarse_output_dir": entry.get("coarse_output_dir", payload.get("coarse_output_dir", "")),
                    "coarse_meta_path": entry.get("coarse_meta_path", payload.get("coarse_meta_path", "")),
                }
            )
            return sample

        raise ValueError(f"Unsupported SingleTurn cached expected_mode={self.expected_mode!r}")


class CachedSingleTurnReconstructionDataset(Dataset):
    def __init__(self, manifest_path: str, data_root: Optional[str] = None):
        del data_root
        raise ValueError(
            "SingleTurn reconstruction cached training is no longer supported for CORNE object-removal mode. "
            f"Got manifest_path={manifest_path}"
        )
