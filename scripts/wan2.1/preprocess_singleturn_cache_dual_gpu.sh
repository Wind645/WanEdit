#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
export SINGLETURN_DATA_DIR=${SINGLETURN_DATA_DIR:-/home/data/nas_hdd/CORNE_extracted}
export CACHE_ROOT=${CACHE_ROOT:-${SINGLETURN_DATA_DIR}/cache}
export OUTPUT_NAME=${OUTPUT_NAME:-singleturn_object_removal_wan2.1_1.3b_v2}

GPU_A=${GPU_A:-6}
GPU_B=${GPU_B:-7}
SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
TOKENIZER_MAX_LENGTH=${TOKENIZER_MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
DTYPE=${DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-0}
MAX_SAMPLES_WITH_MASK_SAM=${MAX_SAMPLES_WITH_MASK_SAM:-30000}
MAX_SAMPLES_WITHOUT_MASK_SAM=${MAX_SAMPLES_WITHOUT_MASK_SAM:-30000}

WITH_DIR="${CACHE_ROOT}/${OUTPUT_NAME}__with_mask_sam_tmp"
WITHOUT_DIR="${CACHE_ROOT}/${OUTPUT_NAME}__without_mask_sam_tmp"
FINAL_DIR="${CACHE_ROOT}/${OUTPUT_NAME}"

common_args=(
  scripts/wan2.1/preprocess_singleturn_cache.py
  --pretrained_model_name_or_path "$MODEL_NAME"
  --train_data_dir "$SINGLETURN_DATA_DIR"
  --singleturn_sample_size "$SAMPLE_HEIGHT" "$SAMPLE_WIDTH"
  --tokenizer_max_length "$TOKENIZER_MAX_LENGTH"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --dtype "$DTYPE"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  common_args+=(--overwrite)
fi

echo "Launching with_mask_sam shard on GPU ${GPU_A} -> ${WITH_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_A}" python "${common_args[@]}" \
  --output_dir "${WITH_DIR}" \
  --max_samples_with_mask_sam "${MAX_SAMPLES_WITH_MASK_SAM}" \
  --max_samples_without_mask_sam 0 &
pid_a=$!

echo "Launching without_mask_sam shard on GPU ${GPU_B} -> ${WITHOUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_B}" python "${common_args[@]}" \
  --output_dir "${WITHOUT_DIR}" \
  --max_samples_with_mask_sam 0 \
  --max_samples_without_mask_sam "${MAX_SAMPLES_WITHOUT_MASK_SAM}" &
pid_b=$!

wait "${pid_a}"
wait "${pid_b}"

merge_args=(
  python
  scripts/wan2.1/merge_singleturn_cache_dirs.py
  --input_dirs "${WITH_DIR}" "${WITHOUT_DIR}"
  --output_dir "${FINAL_DIR}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  merge_args+=(--overwrite)
fi

echo "Merging shards into ${FINAL_DIR}"
"${merge_args[@]}"

echo "Done. Final cache root: ${FINAL_DIR}"
