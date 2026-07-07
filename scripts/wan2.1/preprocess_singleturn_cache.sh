#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
export SINGLETURN_DATA_DIR=${SINGLETURN_DATA_DIR:-/home/data/nas_hdd/CORNE_extracted}
export OUTPUT_DIR=${OUTPUT_DIR:-${SINGLETURN_DATA_DIR}/cache/singleturn_object_removal_wan2.1_1.3b_v3_twoprefix}

SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
TOKENIZER_MAX_LENGTH=${TOKENIZER_MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-0}
DTYPE=${DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-0}
MAX_SAMPLES_WITH_MASK_SAM=${MAX_SAMPLES_WITH_MASK_SAM:-30000}
MAX_SAMPLES_WITHOUT_MASK_SAM=${MAX_SAMPLES_WITHOUT_MASK_SAM:-30000}

cmd=(
  python
  scripts/wan2.1/preprocess_singleturn_cache.py
  --pretrained_model_name_or_path "$MODEL_NAME"
  --train_data_dir "$SINGLETURN_DATA_DIR"
  --output_dir "$OUTPUT_DIR"
  --singleturn_sample_size "$SAMPLE_HEIGHT" "$SAMPLE_WIDTH"
  --tokenizer_max_length "$TOKENIZER_MAX_LENGTH"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --dtype "$DTYPE"
  --max_samples_with_mask_sam "$MAX_SAMPLES_WITH_MASK_SAM"
  --max_samples_without_mask_sam "$MAX_SAMPLES_WITHOUT_MASK_SAM"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
