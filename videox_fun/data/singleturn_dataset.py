import io
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TEXT_EXTENSIONS = {".txt", ".text", ".md"}
SOURCE_HINTS = ("source", "src", "original", "orig", "input")
EDITED_HINTS = ("edited", "edit", "target", "tgt", "output", "result")
PROMPT_HINTS = ("prompt", "instruction", "text")
PARQUET_SOURCE_IMAGE_KEYS = ("original_image", "source_image", "input_image")
PARQUET_EDITED_IMAGE_KEYS = ("edited_image", "target_image", "output_image")
PARQUET_PROMPT_KEYS = ("edit_prompt", "prompt", "instruction", "text")


def _normalize_singleturn_sample_size(sample_size) -> tuple[int, int]:
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


def _build_singleturn_image_transform(sample_size):
    sample_height, sample_width = _normalize_singleturn_sample_size(sample_size)
    return transforms.Compose(
        [
            transforms.Resize(min(sample_height, sample_width)),
            transforms.CenterCrop((sample_height, sample_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _load_manifest_records(manifest_path: str) -> List[Dict[str, str]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        if manifest_path.endswith(".jsonl"):
            raw_records = [json.loads(line) for line in f if line.strip()]
        else:
            raw_records = json.load(f)

    if isinstance(raw_records, dict):
        raw_records = list(raw_records.values())
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("SingleTurn manifest must be a non-empty list or dict of records.")

    records = []
    for idx, record in enumerate(raw_records):
        missing = [key for key in ("source_image", "edited_image", "prompt") if key not in record]
        if missing:
            raise ValueError(
                f"SingleTurn manifest record {idx} is missing required keys {missing}. "
                "Each record must provide source_image, edited_image, and prompt."
            )
        records.append(
            {
                "source_image": record["source_image"],
                "edited_image": record["edited_image"],
                "prompt": "" if record["prompt"] is None else str(record["prompt"]),
                "type": "image",
                "file_path": record["source_image"],
            }
        )
    return records


def _match_by_basename(directory: Path, expected_suffixes: Iterable[str]) -> Dict[str, Path]:
    matched: Dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in expected_suffixes:
            continue
        matched[path.stem] = path
    return matched


def _discover_basename_layout(data_root: Path) -> Optional[List[Dict[str, str]]]:
    source_dir = data_root / "image"
    edited_dir = data_root / "edited_image"
    prompt_dir = data_root / "prompt"
    if not source_dir.is_dir() or not edited_dir.is_dir() or not prompt_dir.is_dir():
        return None

    source_files = _match_by_basename(source_dir, IMAGE_EXTENSIONS)
    edited_files = _match_by_basename(edited_dir, IMAGE_EXTENSIONS)
    prompt_files = _match_by_basename(prompt_dir, TEXT_EXTENSIONS)

    all_keys = set(source_files) | set(edited_files) | set(prompt_files)
    if not all_keys:
        raise ValueError(f"SingleTurn basename layout under {data_root} is empty.")

    missing = [
        key
        for key in sorted(all_keys)
        if key not in source_files or key not in edited_files or key not in prompt_files
    ]
    if missing:
        raise ValueError(
            "SingleTurn basename-matched layout is incomplete. Every sample basename must exist in "
            "`image/`, `edited_image/`, and `prompt/`. Missing or mismatched basenames include: "
            + ", ".join(missing[:10])
        )

    records = []
    for key in sorted(all_keys):
        records.append(
            {
                "source_image": str(source_files[key]),
                "edited_image": str(edited_files[key]),
                "prompt_file": str(prompt_files[key]),
                "type": "image",
                "file_path": str(source_files[key]),
            }
        )
    return records


def _classify_image_role(path: Path) -> Optional[str]:
    stem = path.stem.lower()
    if any(token in stem for token in EDITED_HINTS):
        return "edited"
    if any(token in stem for token in SOURCE_HINTS):
        return "source"
    return None


def _select_prompt_file(text_files: List[Path]) -> Optional[Path]:
    if len(text_files) == 1:
        return text_files[0]
    hinted = [path for path in text_files if any(token in path.stem.lower() for token in PROMPT_HINTS)]
    if len(hinted) == 1:
        return hinted[0]
    return None


def _resolve_folder_pair(images: List[Path]) -> Optional[Dict[str, Path]]:
    if len(images) != 2:
        return None

    roles = [_classify_image_role(path) for path in images]
    if roles.count("source") == 1 and roles.count("edited") == 1:
        return {
            "source_image": images[roles.index("source")],
            "edited_image": images[roles.index("edited")],
        }

    if roles.count("edited") == 1 and roles.count(None) == 1:
        edited_index = roles.index("edited")
        source_index = 1 - edited_index
        return {"source_image": images[source_index], "edited_image": images[edited_index]}

    if roles.count("source") == 1 and roles.count(None) == 1:
        source_index = roles.index("source")
        edited_index = 1 - source_index
        return {"source_image": images[source_index], "edited_image": images[edited_index]}

    return None


def _discover_sample_folders(data_root: Path) -> Optional[List[Dict[str, str]]]:
    records = []
    for root, _, files in os.walk(data_root):
        if not files:
            continue
        root_path = Path(root)
        image_files = sorted(path for path in root_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        text_files = sorted(path for path in root_path.iterdir() if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS)
        if not image_files and not text_files:
            continue

        if len(image_files) != 2 or len(text_files) == 0:
            continue

        prompt_file = _select_prompt_file(text_files)
        image_pair = _resolve_folder_pair(image_files)
        if prompt_file is None or image_pair is None:
            raise ValueError(
                f"Failed to infer SingleTurn sample layout in folder {root_path}. "
                "Expected one prompt text file and two images that can be identified as source/original and edited/target."
            )

        records.append(
            {
                "source_image": str(image_pair["source_image"]),
                "edited_image": str(image_pair["edited_image"]),
                "prompt_file": str(prompt_file),
                "type": "image",
                "file_path": str(image_pair["source_image"]),
            }
        )

    if records:
        return sorted(records, key=lambda record: record["file_path"])
    return None


def _find_singleturn_parquet_files(data_root: str) -> List[Path]:
    root_path = Path(data_root)
    if root_path.is_file() and root_path.suffix.lower() == ".parquet":
        return [root_path]

    search_roots = []
    if root_path.is_dir():
        search_roots.append(root_path / "data")
        search_roots.append(root_path)

    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        parquet_files = sorted(search_root.glob("*.parquet"))
        if parquet_files:
            return parquet_files

    return []


def has_singleturn_parquet_data(data_root: str) -> bool:
    return len(_find_singleturn_parquet_files(data_root)) > 0


def _get_singleturn_parquet_file_infos(data_root: str) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Please install pyarrow to preprocess a parquet-based SingleTurn dataset.") from exc

    parquet_files = _find_singleturn_parquet_files(data_root)
    if not parquet_files:
        return []

    file_infos = []
    start_index = 0
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        num_rows = int(parquet_file.metadata.num_rows)
        file_infos.append(
            {
                "path": parquet_path,
                "num_rows": num_rows,
                "start_index": start_index,
            }
        )
        start_index += num_rows
    return file_infos


def count_singleturn_parquet_records(data_root: str) -> int:
    file_infos = _get_singleturn_parquet_file_infos(data_root)
    if not file_infos:
        raise FileNotFoundError(f"No parquet shards found under {data_root}")
    return int(sum(file_info["num_rows"] for file_info in file_infos))


def _get_first_present_key(column_names: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    name_set = set(column_names)
    for candidate in candidates:
        if candidate in name_set:
            return candidate
    return None


def _get_image_source_name(image_data: Any, default_name: str) -> str:
    if isinstance(image_data, dict):
        path = image_data.get("path")
        if path:
            return str(path)
    if isinstance(image_data, str):
        return image_data
    return default_name


def iter_singleturn_parquet_records(data_root: str, batch_size: int = 1) -> Iterator[Dict[str, Any]]:
    file_infos = _get_singleturn_parquet_file_infos(data_root)
    return _iter_singleturn_parquet_records_from_file_infos(file_infos, batch_size=batch_size)


def _iter_singleturn_parquet_records_from_file_infos(
    file_infos: List[Dict[str, Any]],
    batch_size: int = 1,
) -> Iterator[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Please install pyarrow to preprocess a parquet-based SingleTurn dataset.") from exc

    if not file_infos:
        return

    first_file = pq.ParquetFile(file_infos[0]["path"])
    source_key = _get_first_present_key(first_file.schema_arrow.names, PARQUET_SOURCE_IMAGE_KEYS)
    edited_key = _get_first_present_key(first_file.schema_arrow.names, PARQUET_EDITED_IMAGE_KEYS)
    prompt_key = _get_first_present_key(first_file.schema_arrow.names, PARQUET_PROMPT_KEYS)
    missing = [
        name
        for name, value in (
            ("source image column", source_key),
            ("edited image column", edited_key),
            ("prompt column", prompt_key),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"Parquet SingleTurn dataset is missing required columns: {', '.join(missing)}. "
            f"Available columns: {first_file.schema_arrow.names}"
        )

    for file_info in file_infos:
        file_start = int(file_info["start_index"])
        parquet_path = file_info["path"]
        parquet_file = pq.ParquetFile(parquet_path)
        local_index = 0
        for batch in parquet_file.iter_batches(
            batch_size=max(1, int(batch_size)),
            columns=[source_key, edited_key, prompt_key],
        ):
            rows = batch.to_pylist()
            for row in rows:
                row_index = file_start + local_index
                source_image = row.get(source_key)
                edited_image = row.get(edited_key)
                prompt = row.get(prompt_key)
                source_name = _get_image_source_name(source_image, f"parquet_row_{row_index:06d}_source")
                edited_name = _get_image_source_name(edited_image, f"parquet_row_{row_index:06d}_edited")
                yield {
                    "source_image_data": source_image,
                    "edited_image_data": edited_image,
                    "source_image": source_name,
                    "edited_image": edited_name,
                    "prompt": "" if prompt is None else str(prompt),
                    "row_index": row_index,
                    "global_index": row_index,
                    "type": "image",
                    "file_path": f"{parquet_path.name}#{row_index}",
                }
                local_index += 1


def discover_singleturn_records(
    data_root: str,
    manifest_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    root_path = Path(data_root)
    if not root_path.exists():
        raise FileNotFoundError(f"SingleTurn data root does not exist: {data_root}")

    if manifest_path is not None:
        return _load_manifest_records(manifest_path)

    basename_records = _discover_basename_layout(root_path)
    if basename_records is not None:
        return basename_records

    folder_records = _discover_sample_folders(root_path)
    if folder_records is not None:
        return folder_records

    if has_singleturn_parquet_data(data_root):
        raise ValueError(
            "Detected a parquet-based SingleTurn dataset layout. Use "
            "`scripts/wan2.1/preprocess_singleturn_cache.py` for this dataset, or export it into a manifest / "
            "folder-based layout for uncached training."
        )

    raise ValueError(
        "Failed to discover SingleTurn training data. Supported layouts are either "
        "(1) basename-matched `image/`, `edited_image/`, and `prompt/` directories or "
        "(2) per-sample folders containing one source image, one edited image, and one prompt text file."
    )


class SingleTurnEditDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        manifest_path: Optional[str] = None,
        video_sample_size=512,
        text_drop_ratio: float = 0.1,
    ):
        self.data_root = data_root
        self.dataset = discover_singleturn_records(data_root=data_root, manifest_path=manifest_path)
        self.length = len(self.dataset)
        self.text_drop_ratio = text_drop_ratio
        self.video_sample_size = _normalize_singleturn_sample_size(video_sample_size)
        self.image_transforms = _build_singleturn_image_transform(self.video_sample_size)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        while True:
            record = self.dataset[idx % self.length]
            try:
                prompt = load_singleturn_prompt(record, self.data_root)
                source_image = load_singleturn_image(
                    record.get("source_image_data", record.get("source_image")),
                    self.data_root,
                )
                edited_image = load_singleturn_image(
                    record.get("edited_image_data", record.get("edited_image")),
                    self.data_root,
                )

                source_tensor = self.image_transforms(source_image).unsqueeze(0)
                edited_tensor = self.image_transforms(edited_image).unsqueeze(0)

                if random.random() < self.text_drop_ratio:
                    prompt = ""

                return {
                    "pixel_values_src_image": source_tensor,
                    "pixel_values_tgt_image": edited_tensor,
                    "text": prompt,
                    "data_type": "image",
                    "idx": idx,
                }
            except Exception as exc:
                print(f"Error loading SingleTurn sample: {exc} | record={record}")
                idx = random.randint(0, self.length - 1)


class SingleTurnPreprocessIterableDataset(IterableDataset):
    def __init__(
        self,
        data_root: str,
        manifest_path: Optional[str] = None,
        sample_size=512,
        parquet_batch_size: int = 32,
        reconstruction_only: bool = False,
    ):
        self.data_root = data_root
        self.manifest_path = manifest_path
        self.sample_size = _normalize_singleturn_sample_size(sample_size)
        self.parquet_batch_size = max(1, int(parquet_batch_size))
        self.reconstruction_only = bool(reconstruction_only)
        self.use_parquet = manifest_path is None and has_singleturn_parquet_data(data_root)
        self.records = None if self.use_parquet else discover_singleturn_records(data_root=data_root, manifest_path=manifest_path)
        self.total_length = count_singleturn_parquet_records(data_root) if self.use_parquet else len(self.records)
        self.image_transform = _build_singleturn_image_transform(self.sample_size)

    def __len__(self) -> int:
        return self.total_length

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        if self.use_parquet:
            file_infos = _get_singleturn_parquet_file_infos(self.data_root)
            worker_file_infos = file_infos[worker_id::num_workers]
            iterator = _iter_singleturn_parquet_records_from_file_infos(
                worker_file_infos,
                batch_size=self.parquet_batch_size,
            )
        else:
            iterator = (
                {
                    **self.records[global_index],
                    "global_index": global_index,
                }
                for global_index in range(worker_id, self.total_length, num_workers)
            )

        for record in iterator:
            source_ref = record.get("source_image_data", record.get("source_image"))
            source_image = load_singleturn_image(source_ref, self.data_root)
            if self.reconstruction_only:
                raw_prompt = ""
                edited_image = None
            else:
                raw_prompt = load_singleturn_prompt(record, self.data_root)
                edited_ref = record.get("edited_image_data", record.get("edited_image"))
                edited_image = load_singleturn_image(edited_ref, self.data_root)

            sample = {
                "pixel_values_src_image": self.image_transform(source_image).unsqueeze(0),
                "text": raw_prompt,
                "source_image": record.get("source_image", ""),
                "edited_image": record.get("edited_image", ""),
                "row_index": int(record.get("row_index", -1) if record.get("row_index", -1) is not None else -1),
                "global_index": int(record.get("global_index", record.get("row_index", -1))),
            }
            if edited_image is not None:
                sample["pixel_values_tgt_image"] = self.image_transform(edited_image).unsqueeze(0)
            yield sample


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
    for prompt_key in ("prompt", "edit_prompt", "instruction", "text"):
        if prompt_key in record:
            return "" if record[prompt_key] is None else str(record[prompt_key])
    prompt_file = resolve_singleturn_path(data_root, record["prompt_file"])
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_cache_payload(cache_path: str):
    try:
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(cache_path, map_location="cpu")


def load_singleturn_cache_payload(cache_path: str):
    return _load_cache_payload(cache_path)


class CachedSingleTurnLatentDataset(Dataset):
    def __init__(self, manifest_path: str, data_root: Optional[str] = None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("SingleTurn cached manifest must be a non-empty list.")
        self.data_root = data_root
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        entry = self.manifest[index]
        cache_path = entry["cache_path"]
        if self.data_root is not None and not os.path.isabs(cache_path):
            cache_path = os.path.join(self.data_root, cache_path)
        payload = load_singleturn_cache_payload(cache_path)
        missing = [
            key
            for key in (
                "source_latent_mean",
                "source_latent_logvar",
                "target_latent_mean",
                "target_latent_logvar",
                "prompt_embeds",
                "prompt_seq_len",
            )
            if key not in payload
        ]
        if missing:
            raise ValueError(f"Cached SingleTurn sample {cache_path} is missing keys: {missing}")
        source_image = entry.get("source_image", payload.get("source_image", ""))
        edited_image = entry.get("edited_image", payload.get("edited_image", ""))
        prompt = payload.get("text", entry.get("prompt", ""))
        formatted_text = payload.get("formatted_text", entry.get("formatted_prompt", prompt))
        row_index = payload.get("row_index", entry.get("row_index", -1))
        if row_index is None:
            row_index = -1
        return {
            "source_latent_mean": payload["source_latent_mean"],
            "source_latent_logvar": payload["source_latent_logvar"],
            "target_latent_mean": payload["target_latent_mean"],
            "target_latent_logvar": payload["target_latent_logvar"],
            "prompt_embeds": payload["prompt_embeds"],
            "prompt_seq_len": int(payload["prompt_seq_len"]),
            "text": prompt,
            "formatted_text": formatted_text,
            "cache_path": cache_path,
            "source_image": source_image,
            "edited_image": edited_image,
            "row_index": int(row_index),
            "global_index": int(entry.get("global_index", index)),
            "data_type": "image",
            "idx": index,
        }


class CachedSingleTurnReconstructionDataset(Dataset):
    def __init__(self, manifest_path: str, data_root: Optional[str] = None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if isinstance(manifest, dict):
            manifest = manifest.get("samples", manifest.get("manifest", manifest))
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("SingleTurn reconstruction cached manifest must be a non-empty list.")
        self.data_root = data_root
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        entry = self.manifest[index]
        cache_path = entry["cache_path"]
        if self.data_root is not None and not os.path.isabs(cache_path):
            cache_path = os.path.join(self.data_root, cache_path)
        payload = load_singleturn_cache_payload(cache_path)
        missing = [
            key
            for key in (
                "source_latent_mean",
                "source_latent_logvar",
            )
            if key not in payload
        ]
        if missing:
            raise ValueError(f"Cached SingleTurn reconstruction sample {cache_path} is missing keys: {missing}")
        source_image = entry.get("source_image", payload.get("source_image", ""))
        row_index = payload.get("row_index", entry.get("row_index", -1))
        if row_index is None:
            row_index = -1
        return {
            "source_latent_mean": payload["source_latent_mean"],
            "source_latent_logvar": payload["source_latent_logvar"],
            "cache_path": cache_path,
            "source_image": source_image,
            "row_index": int(row_index),
            "global_index": int(entry.get("global_index", index)),
            "data_type": "image",
            "idx": index,
        }
