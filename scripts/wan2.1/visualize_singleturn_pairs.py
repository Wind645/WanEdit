#!/usr/bin/env python

import argparse
import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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
    discover_singleturn_records,
    has_singleturn_parquet_data,
    iter_singleturn_parquet_records,
    load_singleturn_image,
    load_singleturn_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SingleTurn source/edited pairs as JPG contact sheets.")
    parser.add_argument("--data_root", type=str, required=True, help="SingleTurn dataset root.")
    parser.add_argument("--manifest_path", type=str, default=None, help="Optional manifest path for non-parquet layouts.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for generated JPG files.")
    parser.add_argument("--num_samples", type=int, default=20, help="How many pairs to visualize.")
    parser.add_argument("--thumb_height", type=int, default=192, help="Thumbnail height for each image.")
    parser.add_argument("--pair_columns", type=int, default=2, help="How many source-edited pairs per row in pairs.jpg.")
    parser.add_argument("--grid_columns", type=int, default=5, help="How many images per row in source/edited grids.")
    return parser.parse_args()


def _load_records(data_root: str, manifest_path: str | None, num_samples: int):
    records = []
    if manifest_path is None and has_singleturn_parquet_data(data_root):
        for idx, record in enumerate(iter_singleturn_parquet_records(data_root, batch_size=8)):
            records.append(record)
            if idx + 1 >= num_samples:
                break
    else:
        discovered = discover_singleturn_records(data_root=data_root, manifest_path=manifest_path)
        records = discovered[:num_samples]
    return records


def _thumbnail(image: Image.Image, thumb_height: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    thumb_width = max(1, int(round(width * thumb_height / max(1, height))))
    return image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, font) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    for end in range(len(text), 0, -1):
        candidate = text[:end].rstrip() + suffix
        if draw.textlength(candidate, font=font) <= max_width:
            return candidate
    return suffix


def _make_pairs_sheet(records, data_root: str, thumb_height: int, pair_columns: int) -> Image.Image:
    font = ImageFont.load_default()
    source_images = []
    edited_images = []
    for record in records:
        source_ref = record.get("source_image_data", record.get("source_image"))
        edited_ref = record.get("edited_image_data", record.get("edited_image"))
        source_images.append(_thumbnail(load_singleturn_image(source_ref, data_root), thumb_height))
        edited_images.append(_thumbnail(load_singleturn_image(edited_ref, data_root), thumb_height))

    pair_widths = [src.width + edt.width + 18 for src, edt in zip(source_images, edited_images)]
    cell_width = max(pair_widths) + 16
    text_height = 28
    cell_height = thumb_height + text_height + 16
    rows = math.ceil(len(records) / pair_columns)
    canvas = Image.new("RGB", (cell_width * pair_columns, cell_height * rows), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (record, src, edt) in enumerate(zip(records, source_images, edited_images)):
        row = idx // pair_columns
        col = idx % pair_columns
        x0 = col * cell_width
        y0 = row * cell_height
        canvas.paste(src, (x0 + 8, y0 + 8))
        canvas.paste(edt, (x0 + 8 + src.width + 10, y0 + 8))
        prompt = record.get("prompt")
        if prompt is None and "prompt_file" in record:
            prompt = load_singleturn_prompt(record, data_root)
        label = f"{idx:02d}: {_fit_text(draw, prompt or '', cell_width - 16, font)}"
        draw.text((x0 + 8, y0 + thumb_height + 10), label, fill="black", font=font)
    return canvas


def _make_single_grid(records, data_root: str, thumb_height: int, grid_columns: int, key: str) -> Image.Image:
    font = ImageFont.load_default()
    images = []
    labels = []
    for idx, record in enumerate(records):
        ref = record.get(f"{key}_image_data", record.get(f"{key}_image"))
        images.append(_thumbnail(load_singleturn_image(ref, data_root), thumb_height))
        labels.append(f"{idx:02d}")

    cell_width = max(image.width for image in images) + 12
    text_height = 22
    cell_height = thumb_height + text_height + 12
    rows = math.ceil(len(images) / grid_columns)
    canvas = Image.new("RGB", (cell_width * grid_columns, cell_height * rows), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (image, label) in enumerate(zip(images, labels)):
        row = idx // grid_columns
        col = idx % grid_columns
        x0 = col * cell_width
        y0 = row * cell_height
        canvas.paste(image, (x0 + 6, y0 + 6))
        draw.text((x0 + 6, y0 + thumb_height + 6), label, fill="black", font=font)
    return canvas


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(args.data_root, args.manifest_path, args.num_samples)
    if not records:
        raise ValueError("No samples found to visualize.")

    pairs_sheet = _make_pairs_sheet(records, args.data_root, args.thumb_height, max(1, args.pair_columns))
    source_sheet = _make_single_grid(records, args.data_root, args.thumb_height, max(1, args.grid_columns), "source")
    edited_sheet = _make_single_grid(records, args.data_root, args.thumb_height, max(1, args.grid_columns), "edited")

    pairs_path = output_dir / "pairs.jpg"
    source_path = output_dir / "source_grid.jpg"
    edited_path = output_dir / "edited_grid.jpg"
    pairs_sheet.save(pairs_path, format="JPEG", quality=95)
    source_sheet.save(source_path, format="JPEG", quality=95)
    edited_sheet.save(edited_path, format="JPEG", quality=95)

    print(
        f"Saved {len(records)} samples to:\n"
        f"  {pairs_path}\n"
        f"  {source_path}\n"
        f"  {edited_path}"
    )


if __name__ == "__main__":
    main()
