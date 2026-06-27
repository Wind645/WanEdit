#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
if [[ -z "${SINGLETURN_DATA_DIR:-}" ]]; then
  if [[ -d /home/data/nas_hdd/instructpix2pix ]]; then
    export SINGLETURN_DATA_DIR=/home/data/nas_hdd/instructpix2pix
  else
    export SINGLETURN_DATA_DIR=/home/data/nas_hdd/Singleturn
  fi
fi
export SINGLETURN_MANIFEST=${SINGLETURN_MANIFEST:-}
export OUTPUT_DIR=${OUTPUT_DIR:-${SINGLETURN_DATA_DIR}/cache/singleturn_reconstruction_wan2.1_1.3b}
export RECONSTRUCTION_PROMPT=${RECONSTRUCTION_PROMPT:-Reconstruct the source image.}

SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
TOKENIZER_MAX_LENGTH=${TOKENIZER_MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
PARQUET_BATCH_SIZE=${PARQUET_BATCH_SIZE:-128}
DTYPE=${DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-0}

cmd=(
  python
  scripts/wan2.1/preprocess_singleturn_cache.py
  --pretrained_model_name_or_path "$MODEL_NAME"
  --train_data_dir "$SINGLETURN_DATA_DIR"
  --output_dir "$OUTPUT_DIR"
  --reconstruction_mode
  --reconstruction_prompt "$RECONSTRUCTION_PROMPT"
  --singleturn_sample_size "$SAMPLE_HEIGHT" "$SAMPLE_WIDTH"
  --tokenizer_max_length "$TOKENIZER_MAX_LENGTH"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --parquet_batch_size "$PARQUET_BATCH_SIZE"
  --dtype "$DTYPE"
)

if [[ -n "${SINGLETURN_MANIFEST}" ]]; then
  cmd+=(--train_data_manifest "$SINGLETURN_MANIFEST")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
