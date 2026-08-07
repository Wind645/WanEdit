#!/usr/bin/env bash
set -euo pipefail

OBJECTCLEAR_ROOT="${OBJECTCLEAR_ROOT:-/mnt/cpfs/jiachengliu/dataset/ObjectClear/extracted/train/captured}"
CORNE_ROOT="${CORNE_ROOT:-/mnt/cpfs/jiachengliu/dataset/CORNE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/cpfs/jiachengliu/dataset/ObjectClear_CORNE_60k_scribble_v1}"
TARGET_TOTAL="${TARGET_TOTAL:-60000}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-16}"
OVERWRITE="${OVERWRITE:-0}"

ARGS=(
  --objectclear_root "$OBJECTCLEAR_ROOT"
  --corne_root "$CORNE_ROOT"
  --output_root "$OUTPUT_ROOT"
  --target_total "$TARGET_TOTAL"
  --seed "$SEED"
  --num_workers "$NUM_WORKERS"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python scripts/wan2.1/build_objectclear_corne_60k_scribble_dataset.py "${ARGS[@]}"
