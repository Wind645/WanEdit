#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
export COARSE_OUTPUT_DIR=${COARSE_OUTPUT_DIR:-outputs/singleturn_object_removal_v3_twoprefix}
export OUTPUT_DIR=${OUTPUT_DIR:-/home/data/nas_hdd/CORNE_extracted/cache/singleturn_object_removal_refine_wan2.1_1.3b_v1}

DTYPE=${DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-0}

cmd=(
  python
  scripts/wan2.1/preprocess_singleturn_refine_cache.py
  --pretrained_model_name_or_path "$MODEL_NAME"
  --coarse_output_dir "$COARSE_OUTPUT_DIR"
  --output_dir "$OUTPUT_DIR"
  --dtype "$DTYPE"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
