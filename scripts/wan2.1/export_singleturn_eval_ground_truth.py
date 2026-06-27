#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError("Please install pyarrow to export SingleTurn ground-truth images.") from exc

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

from videox_fun.data.singleturn_dataset import (  # noqa: E402
    PARQUET_EDITED_IMAGE_KEYS,
    PARQUET_PROMPT_KEYS,
    PARQUET_SOURCE_IMAGE_KEYS,
    _get_first_present_key,
    _get_singleturn_parquet_file_infos,
    load_singleturn_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export GT source/edited images for cached SingleTurn eval outputs.")
    parser.add_argument("--eval_dir", type=str, required=True, help="Eval output root containing per-sample folders.")
    parser.add_argument("--data_root", type=str, required=True, help="SingleTurn dataset root with parquet shards.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exported GT files.")
    return parser.parse_args()


def _resolve_global_index(cache_path: str) -> int:
    stem = Path(cache_path).stem
    prefix = stem.split("_", 1)[0]
    return int(prefix)


def _find_parquet_row(file_infos, global_index: int):
    selected = None
    for file_info in file_infos:
        start = int(file_info["start_index"])
        end = start + int(file_info["num_rows"])
        if start <= global_index < end:
            selected = file_info
            break
    if selected is None:
        raise ValueError(f"Failed to resolve global index {global_index} into parquet shards.")
    local_index = global_index - int(selected["start_index"])
    return selected["path"], local_index


def _load_single_row(parquet_path: Path, local_index: int):
    parquet_file = pq.ParquetFile(parquet_path)
    schema_names = parquet_file.schema_arrow.names
    source_key = _get_first_present_key(schema_names, PARQUET_SOURCE_IMAGE_KEYS)
    edited_key = _get_first_present_key(schema_names, PARQUET_EDITED_IMAGE_KEYS)
    prompt_key = _get_first_present_key(schema_names, PARQUET_PROMPT_KEYS)
    missing = [name for name, key in (("source image", source_key), ("edited image", edited_key)) if key is None]
    if missing:
        raise ValueError(f"Parquet shard {parquet_path} is missing columns: {missing}")

    row = pq.read_table(parquet_path, columns=[key for key in (source_key, edited_key, prompt_key) if key is not None])
    row = row.slice(local_index, 1).to_pylist()[0]
    return {
        "source_image_data": row[source_key],
        "edited_image_data": row[edited_key],
        "prompt": row.get(prompt_key, "") if prompt_key is not None else "",
    }


def _save_image(image, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    file_infos = _get_singleturn_parquet_file_infos(args.data_root)
    if not file_infos:
        raise FileNotFoundError(f"No parquet shards found under {args.data_root}")

    meta_paths = sorted(eval_dir.glob("*/singleturn_meta.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No singleturn_meta.json files found under {eval_dir}")

    exported = []
    for meta_path in meta_paths:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        sample_dir = meta_path.parent
        gt_source_path = sample_dir / "gt_source.png"
        gt_edited_path = sample_dir / "gt_edited.png"
        gt_meta_path = sample_dir / "gt_meta.json"

        if (not args.overwrite) and gt_source_path.exists() and gt_edited_path.exists() and gt_meta_path.exists():
            exported.append(str(sample_dir))
            continue

        global_index = _resolve_global_index(meta["cache_path"])
        parquet_path, local_index = _find_parquet_row(file_infos, global_index)
        row = _load_single_row(parquet_path, local_index)

        source_image = load_singleturn_image(row["source_image_data"], args.data_root)
        edited_image = load_singleturn_image(row["edited_image_data"], args.data_root)
        _save_image(source_image, gt_source_path)
        _save_image(edited_image, gt_edited_path)

        with open(gt_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "parquet_path": str(parquet_path),
                    "global_index": global_index,
                    "local_index": local_index,
                    "prompt": meta.get("prompt", row.get("prompt", "")),
                    "formatted_prompt": meta.get("formatted_prompt", ""),
                    "source_image_name": meta.get("source_image", ""),
                    "edited_image_name": meta.get("edited_image", ""),
                },
                f,
                indent=2,
            )
        exported.append(str(sample_dir))

    print(json.dumps({"exported_count": len(exported), "sample_dirs": exported}, indent=2))


if __name__ == "__main__":
    main()
