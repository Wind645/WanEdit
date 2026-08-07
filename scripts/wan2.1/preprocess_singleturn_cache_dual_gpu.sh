#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-/mnt/cpfs/jiachengliu/pretrained_models/Wan-AI/Wan2.1-T2V-1.3B}
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

count_objectclear_inputs() {
  local root="$1"
  find "$root" -mindepth 3 -maxdepth 3 -path '*/input/*' -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l
}

count_corne_with_mask_sam() {
  local root="$1"
  if [[ -d "${root}/mask_sam" ]]; then
    find "${root}/mask_sam" -maxdepth 1 -type f \
      \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l
  else
    echo 0
  fi
}

count_available_with_mask_sam() {
  local root="$1"
  if find "$root" -mindepth 2 -maxdepth 2 -type d -path '*/object_mask' -print -quit | grep -q .; then
    count_objectclear_inputs "$root"
  else
    count_corne_with_mask_sam "$root"
  fi
}

launch_preprocess() {
  local gpu="$1"
  local out_dir="$2"
  local max_with="$3"
  local max_without="$4"
  local skip_with="$5"
  local skip_without="$6"
  echo "Launching shard on GPU ${gpu} -> ${out_dir} max_with=${max_with} max_without=${max_without} skip_with=${skip_with} skip_without=${skip_without}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${common_args[@]}" \
    --output_dir "${out_dir}" \
    --max_samples_with_mask_sam "${max_with}" \
    --max_samples_without_mask_sam "${max_without}" \
    --skip_samples_with_mask_sam "${skip_with}" \
    --skip_samples_without_mask_sam "${skip_without}" &
  pids+=("$!")
  input_dirs+=("${out_dir}")
}

if (( MAX_SAMPLES_WITHOUT_MASK_SAM != 0 && (MAX_SAMPLES_WITH_MASK_SAM % 2 != 0 || MAX_SAMPLES_WITHOUT_MASK_SAM % 2 != 0) )); then
  echo "MAX_SAMPLES_WITH_MASK_SAM and MAX_SAMPLES_WITHOUT_MASK_SAM must both be even for 4-way sharding." >&2
  exit 1
fi

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

pids=()
input_dirs=()

if (( MAX_SAMPLES_WITHOUT_MASK_SAM == 0 )); then
  AVAILABLE_WITH=$(count_available_with_mask_sam "$SINGLETURN_DATA_DIR")
  if (( AVAILABLE_WITH <= 0 )); then
    echo "No with_mask_sam samples found under ${SINGLETURN_DATA_DIR}" >&2
    exit 1
  fi
  if (( MAX_SAMPLES_WITH_MASK_SAM > AVAILABLE_WITH )); then
    EFFECTIVE_WITH_TOTAL=${AVAILABLE_WITH}
  else
    EFFECTIVE_WITH_TOTAL=${MAX_SAMPLES_WITH_MASK_SAM}
  fi
  GPUS=("${GPU_A}" "${GPU_B}" "${GPU_C}" "${GPU_D}")
  NUM_SHARDS=${#GPUS[@]}
  BASE_SHARD=$(( EFFECTIVE_WITH_TOTAL / NUM_SHARDS ))
  REMAINDER=$(( EFFECTIVE_WITH_TOTAL % NUM_SHARDS ))
  SKIP_WITH=0
  echo "Detected with_mask_sam samples=${AVAILABLE_WITH}; using ${EFFECTIVE_WITH_TOTAL} samples over ${NUM_SHARDS} GPUs."
  for shard_idx in "${!GPUS[@]}"; do
    SHARD_SIZE=${BASE_SHARD}
    if (( shard_idx < REMAINDER )); then
      SHARD_SIZE=$(( SHARD_SIZE + 1 ))
    fi
    if (( SHARD_SIZE == 0 )); then
      continue
    fi
    SHARD_DIR="${CACHE_ROOT}/${OUTPUT_NAME}__with_mask_sam_shard${shard_idx}_tmp"
    launch_preprocess "${GPUS[$shard_idx]}" "${SHARD_DIR}" "${SHARD_SIZE}" 0 "${SKIP_WITH}" 0
    SKIP_WITH=$(( SKIP_WITH + SHARD_SIZE ))
  done
else
  if (( MAX_SAMPLES_WITH_MASK_SAM % 2 != 0 || MAX_SAMPLES_WITHOUT_MASK_SAM % 2 != 0 )); then
    echo "MAX_SAMPLES_WITH_MASK_SAM and MAX_SAMPLES_WITHOUT_MASK_SAM must both be even for 4-way sharding." >&2
    exit 1
  fi
  WITH_SHARD=$(( MAX_SAMPLES_WITH_MASK_SAM / 2 ))
  WITHOUT_SHARD=$(( MAX_SAMPLES_WITHOUT_MASK_SAM / 2 ))
  launch_preprocess "${GPU_A}" "${CACHE_ROOT}/${OUTPUT_NAME}__with_mask_sam_shard0_tmp" "${WITH_SHARD}" 0 0 0
  launch_preprocess "${GPU_B}" "${CACHE_ROOT}/${OUTPUT_NAME}__with_mask_sam_shard1_tmp" "${WITH_SHARD}" 0 "${WITH_SHARD}" 0
  launch_preprocess "${GPU_C}" "${CACHE_ROOT}/${OUTPUT_NAME}__without_mask_sam_shard0_tmp" 0 "${WITHOUT_SHARD}" 0 0
  launch_preprocess "${GPU_D}" "${CACHE_ROOT}/${OUTPUT_NAME}__without_mask_sam_shard1_tmp" 0 "${WITHOUT_SHARD}" 0 "${WITHOUT_SHARD}"
fi

for pid in "${pids[@]}"; do
  wait "${pid}"
done

merge_args=(
  python
  scripts/wan2.1/merge_singleturn_cache_dirs.py
  --input_dirs "${input_dirs[@]}"
  --output_dir "${FINAL_DIR}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  merge_args+=(--overwrite)
fi

echo "Merging shards into ${FINAL_DIR}"
"${merge_args[@]}"

echo "Done. Final cache root: ${FINAL_DIR}"
