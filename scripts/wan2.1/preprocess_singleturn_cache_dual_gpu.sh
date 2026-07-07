#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
export SINGLETURN_DATA_DIR=${SINGLETURN_DATA_DIR:-/home/data/nas_hdd/CORNE_extracted}
export CACHE_ROOT=${CACHE_ROOT:-${SINGLETURN_DATA_DIR}/cache}
export OUTPUT_NAME=${OUTPUT_NAME:-singleturn_object_removal_wan2.1_1.3b_v3_twoprefix}

GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
GPU_C=${GPU_C:-2}
GPU_D=${GPU_D:-3}
SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
TOKENIZER_MAX_LENGTH=${TOKENIZER_MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
DTYPE=${DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-0}
MAX_SAMPLES_WITH_MASK_SAM=${MAX_SAMPLES_WITH_MASK_SAM:-30000}
MAX_SAMPLES_WITHOUT_MASK_SAM=${MAX_SAMPLES_WITHOUT_MASK_SAM:-30000}

if (( MAX_SAMPLES_WITH_MASK_SAM % 2 != 0 || MAX_SAMPLES_WITHOUT_MASK_SAM % 2 != 0 )); then
  echo "MAX_SAMPLES_WITH_MASK_SAM and MAX_SAMPLES_WITHOUT_MASK_SAM must both be even for 4-way sharding." >&2
  exit 1
fi

WITH_SHARD=$(( MAX_SAMPLES_WITH_MASK_SAM / 2 ))
WITHOUT_SHARD=$(( MAX_SAMPLES_WITHOUT_MASK_SAM / 2 ))

WITH_DIR_A="${CACHE_ROOT}/${OUTPUT_NAME}__with_mask_sam_shard0_tmp"
WITH_DIR_B="${CACHE_ROOT}/${OUTPUT_NAME}__with_mask_sam_shard1_tmp"
WITHOUT_DIR_A="${CACHE_ROOT}/${OUTPUT_NAME}__without_mask_sam_shard0_tmp"
WITHOUT_DIR_B="${CACHE_ROOT}/${OUTPUT_NAME}__without_mask_sam_shard1_tmp"
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

echo "Launching with_mask_sam shard0 on GPU ${GPU_A} -> ${WITH_DIR_A}"
CUDA_VISIBLE_DEVICES="${GPU_A}" python "${common_args[@]}" \
  --output_dir "${WITH_DIR_A}" \
  --max_samples_with_mask_sam "${WITH_SHARD}" \
  --max_samples_without_mask_sam 0 \
  --skip_samples_with_mask_sam 0 \
  --skip_samples_without_mask_sam 0 &
pid_a=$!

echo "Launching with_mask_sam shard1 on GPU ${GPU_B} -> ${WITH_DIR_B}"
CUDA_VISIBLE_DEVICES="${GPU_B}" python "${common_args[@]}" \
  --output_dir "${WITH_DIR_B}" \
  --max_samples_with_mask_sam "${WITH_SHARD}" \
  --max_samples_without_mask_sam 0 \
  --skip_samples_with_mask_sam "${WITH_SHARD}" \
  --skip_samples_without_mask_sam 0 &
pid_b=$!

echo "Launching without_mask_sam shard0 on GPU ${GPU_C} -> ${WITHOUT_DIR_A}"
CUDA_VISIBLE_DEVICES="${GPU_C}" python "${common_args[@]}" \
  --output_dir "${WITHOUT_DIR_A}" \
  --max_samples_with_mask_sam 0 \
  --max_samples_without_mask_sam "${WITHOUT_SHARD}" \
  --skip_samples_with_mask_sam 0 \
  --skip_samples_without_mask_sam 0 &
pid_c=$!

echo "Launching without_mask_sam shard1 on GPU ${GPU_D} -> ${WITHOUT_DIR_B}"
CUDA_VISIBLE_DEVICES="${GPU_D}" python "${common_args[@]}" \
  --output_dir "${WITHOUT_DIR_B}" \
  --max_samples_with_mask_sam 0 \
  --max_samples_without_mask_sam "${WITHOUT_SHARD}" \
  --skip_samples_with_mask_sam 0 \
  --skip_samples_without_mask_sam "${WITHOUT_SHARD}" &
pid_d=$!

wait "${pid_a}"
wait "${pid_b}"
wait "${pid_c}"
wait "${pid_d}"

merge_args=(
  python
  scripts/wan2.1/merge_singleturn_cache_dirs.py
  --input_dirs "${WITH_DIR_A}" "${WITH_DIR_B}" "${WITHOUT_DIR_A}" "${WITHOUT_DIR_B}"
  --output_dir "${FINAL_DIR}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  merge_args+=(--overwrite)
fi

echo "Merging shards into ${FINAL_DIR}"
"${merge_args[@]}"

echo "Done. Final cache root: ${FINAL_DIR}"
