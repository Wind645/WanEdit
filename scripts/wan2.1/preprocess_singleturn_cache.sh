#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

export PYTHON_BIN=${PYTHON_BIN:-/home/data/zhikai/miniconda3/envs/videocof/bin/python}
export MODEL_NAME=${MODEL_NAME:-/home/data/zhikai/VideoCoF/models/Wan2.1-T2V-1.3B}
export SINGLETURN_DATA_DIR=${SINGLETURN_DATA_DIR:-}
export SOURCE_CACHED_DATA_DIR=${SOURCE_CACHED_DATA_DIR:-/home/data/nas_hdd/CORNE_extracted/cache/singleturn_object_removal_wan2.1_1.3b_v3_twoprefix}
export SOURCE_CACHED_DATA_META=${SOURCE_CACHED_DATA_META:-${SOURCE_CACHED_DATA_DIR}/manifest.json}
export OUTPUT_DIR=${OUTPUT_DIR:-/home/data/nas_hdd/CORNE_extracted/cache/singleturn_object_removal_wan2.1_1.3b_sam_strict_keyframe_cache_v1}

SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
TOKENIZER_MAX_LENGTH=${TOKENIZER_MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-0}
DTYPE=${DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-0}
MAX_SAMPLES_WITH_MASK_SAM=${MAX_SAMPLES_WITH_MASK_SAM:-30000}
MAX_SAMPLES_WITHOUT_MASK_SAM=${MAX_SAMPLES_WITHOUT_MASK_SAM:-30000}
SKIP_SAMPLES_WITH_MASK_SAM=${SKIP_SAMPLES_WITH_MASK_SAM:-0}
SKIP_SAMPLES_WITHOUT_MASK_SAM=${SKIP_SAMPLES_WITHOUT_MASK_SAM:-0}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter does not exist: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "Model directory does not exist: ${MODEL_NAME}" >&2
  exit 1
fi

cmd=(
  "$PYTHON_BIN"
  "$REPO_ROOT/scripts/wan2.1/preprocess_singleturn_cache.py"
  --pretrained_model_name_or_path "$MODEL_NAME"
  --output_dir "$OUTPUT_DIR"
  --config_path "$REPO_ROOT/config/wan2.1/wan_civitai.yaml"
  --singleturn_sample_size "$SAMPLE_HEIGHT" "$SAMPLE_WIDTH"
  --tokenizer_max_length "$TOKENIZER_MAX_LENGTH"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --dtype "$DTYPE"
  --max_samples_with_mask_sam "$MAX_SAMPLES_WITH_MASK_SAM"
  --max_samples_without_mask_sam "$MAX_SAMPLES_WITHOUT_MASK_SAM"
  --skip_samples_with_mask_sam "$SKIP_SAMPLES_WITH_MASK_SAM"
  --skip_samples_without_mask_sam "$SKIP_SAMPLES_WITHOUT_MASK_SAM"
)

if [[ -n "${SOURCE_CACHED_DATA_META}" ]]; then
  if [[ ! -f "${SOURCE_CACHED_DATA_META}" ]]; then
    echo "Legacy cache manifest does not exist: ${SOURCE_CACHED_DATA_META}" >&2
    exit 1
  fi
  cmd+=(
    --source_cached_data_meta "$SOURCE_CACHED_DATA_META"
    --source_cached_data_dir "$SOURCE_CACHED_DATA_DIR"
  )
else
  if [[ -z "${SINGLETURN_DATA_DIR}" ]]; then
    echo "Set SOURCE_CACHED_DATA_META for cache conversion, or set SINGLETURN_DATA_DIR for raw-image preprocessing." >&2
    exit 1
  fi
  for required_dir in shot bg mask-check mask_sam; do
    if [[ ! -d "${SINGLETURN_DATA_DIR}/${required_dir}" ]]; then
      echo "Missing required CORNE directory: ${SINGLETURN_DATA_DIR}/${required_dir}" >&2
      exit 1
    fi
  done
  cmd+=(--train_data_dir "$SINGLETURN_DATA_DIR")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "tqdm progress bar: enabled in preprocess_singleturn_cache.py"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
