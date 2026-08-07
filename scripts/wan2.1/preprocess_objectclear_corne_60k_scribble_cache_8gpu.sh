#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-/mnt/cpfs/jiachengliu/pretrained_models/Wan-AI/Wan2.1-T2V-1.3B}
export SINGLETURN_DATA_DIR=${SINGLETURN_DATA_DIR:-/mnt/cpfs/jiachengliu/dataset/ObjectClear_CORNE_60k_scribble_v1}
export CACHE_ROOT=${CACHE_ROOT:-${SINGLETURN_DATA_DIR}/cache}
export OUTPUT_NAME=${OUTPUT_NAME:-singleturn_objectclear_corne_60k_scribble_wan2.1_1.3b_keyframe_cache_v1}

GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
GPU_C=${GPU_C:-2}
GPU_D=${GPU_D:-3}
GPU_E=${GPU_E:-4}
GPU_F=${GPU_F:-5}
GPU_G=${GPU_G:-6}
GPU_H=${GPU_H:-7}
SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
TOKENIZER_MAX_LENGTH=${TOKENIZER_MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
DTYPE=${DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-0}
MAX_SAMPLES_WITH_MASK_SAM=${MAX_SAMPLES_WITH_MASK_SAM:-60000}

count_flat_with_mask_sam() {
  local root="$1"
  if [[ -d "${root}/mask_inp" ]]; then
    find -L "${root}/mask_inp" -maxdepth 1 -type f \
      \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l
  else
    echo 0
  fi
}

launch_preprocess() {
  local gpu="$1"
  local out_dir="$2"
  local max_with="$3"
  local skip_with="$4"
  echo "Launching shard on GPU ${gpu} -> ${out_dir} max_with=${max_with} skip_with=${skip_with}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${common_args[@]}" \
    --output_dir "${out_dir}" \
    --max_samples_with_mask_sam "${max_with}" \
    --max_samples_without_mask_sam 0 \
    --skip_samples_with_mask_sam "${skip_with}" \
    --skip_samples_without_mask_sam 0 &
  pids+=("$!")
  input_dirs+=("${out_dir}")
}

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

AVAILABLE_WITH=$(count_flat_with_mask_sam "$SINGLETURN_DATA_DIR")
if (( AVAILABLE_WITH <= 0 )); then
  echo "No mask_inp samples found under ${SINGLETURN_DATA_DIR}" >&2
  exit 1
fi

if (( MAX_SAMPLES_WITH_MASK_SAM > AVAILABLE_WITH )); then
  EFFECTIVE_WITH_TOTAL=${AVAILABLE_WITH}
else
  EFFECTIVE_WITH_TOTAL=${MAX_SAMPLES_WITH_MASK_SAM}
fi

GPUS=("${GPU_A}" "${GPU_B}" "${GPU_C}" "${GPU_D}" "${GPU_E}" "${GPU_F}" "${GPU_G}" "${GPU_H}")
NUM_SHARDS=${#GPUS[@]}
BASE_SHARD=$(( EFFECTIVE_WITH_TOTAL / NUM_SHARDS ))
REMAINDER=$(( EFFECTIVE_WITH_TOTAL % NUM_SHARDS ))
SKIP_WITH=0
pids=()
input_dirs=()

echo "Detected mask_inp samples=${AVAILABLE_WITH}; using ${EFFECTIVE_WITH_TOTAL} samples over ${NUM_SHARDS} GPUs."
for shard_idx in "${!GPUS[@]}"; do
  SHARD_SIZE=${BASE_SHARD}
  if (( shard_idx < REMAINDER )); then
    SHARD_SIZE=$(( SHARD_SIZE + 1 ))
  fi
  if (( SHARD_SIZE == 0 )); then
    continue
  fi
  SHARD_DIR="${CACHE_ROOT}/${OUTPUT_NAME}__with_mask_sam_shard${shard_idx}_tmp"
  launch_preprocess "${GPUS[$shard_idx]}" "${SHARD_DIR}" "${SHARD_SIZE}" "${SKIP_WITH}"
  SKIP_WITH=$(( SKIP_WITH + SHARD_SIZE ))
done

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
