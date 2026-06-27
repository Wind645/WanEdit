#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
export IMAGE_PATH=${IMAGE_PATH:-}
export PROMPT=${PROMPT:-}
export CACHED_SAMPLE_PATH=${CACHED_SAMPLE_PATH:-}
export CACHED_DATA_META=${CACHED_DATA_META:-}
export CACHED_DATA_DIR=${CACHED_DATA_DIR:-}
export SHARED_PROMPT_CACHE=${SHARED_PROMPT_CACHE:-}
export CACHED_START_INDEX=${CACHED_START_INDEX:-0}
export CACHED_NUM_SAMPLES=${CACHED_NUM_SAMPLES:-}
export CACHED_NUM_WORKERS=${CACHED_NUM_WORKERS:-2}
export CACHED_PREFETCH_FACTOR=${CACHED_PREFETCH_FACTOR:-2}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/singleturn}
export LORA_PATH=${LORA_PATH:-}
export PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-Edit the source image according to this instruction: {prompt}}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}
SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-${SAMPLE_SIZE:-480}}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-${SAMPLE_SIZE:-832}}
SEED=${SEED:-0}
FPS=${FPS:-4}
DTYPE=${DTYPE:-bf16}
ACCELERATE_MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION:-bf16}

if [[ "${DTYPE}" == "fp16" ]]; then
  ACCELERATE_MIXED_PRECISION=fp16
elif [[ "${DTYPE}" == "fp32" ]]; then
  ACCELERATE_MIXED_PRECISION=no
fi

if [[ -z "${IMAGE_PATH}" ]]; then
  if [[ -z "${CACHED_SAMPLE_PATH}" && -z "${CACHED_DATA_META}" ]]; then
    echo "Set either IMAGE_PATH/PROMPT or CACHED_SAMPLE_PATH/CACHED_DATA_META." >&2
    exit 1
  fi
fi

cmd_base=(
  scripts/wan2.1/singleturn_edit_infer.py
  --pretrained_model_name_or_path "$MODEL_NAME"
  --output_dir "$OUTPUT_DIR"
  --prompt_template "$PROMPT_TEMPLATE"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --sample_size "$SAMPLE_HEIGHT" "$SAMPLE_WIDTH"
  --seed "$SEED"
  --fps "$FPS"
  --dtype "$DTYPE"
)

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  cmd=(
    accelerate
    launch
    --num_processes "$NPROC_PER_NODE"
    --num_machines 1
    --dynamo_backend no
    --mixed_precision "$ACCELERATE_MIXED_PRECISION"
    "${cmd_base[@]}"
  )
else
  cmd=(
    python
    "${cmd_base[@]}"
  )
fi

if [[ -n "${CACHED_SAMPLE_PATH}" || -n "${CACHED_DATA_META}" ]]; then
  if [[ -n "${IMAGE_PATH}" || -n "${PROMPT}" ]]; then
    echo "Cached mode does not accept IMAGE_PATH/PROMPT." >&2
    exit 1
  fi
  if [[ -n "${CACHED_SAMPLE_PATH}" ]]; then
    cmd+=(--cached_sample_path "$CACHED_SAMPLE_PATH")
  fi
  if [[ -n "${CACHED_DATA_META}" ]]; then
    cmd+=(--cached_data_meta "$CACHED_DATA_META")
  fi
  if [[ -n "${CACHED_DATA_DIR}" ]]; then
    cmd+=(--cached_data_dir "$CACHED_DATA_DIR")
  fi
  if [[ -n "${SHARED_PROMPT_CACHE}" ]]; then
    cmd+=(--shared_prompt_cache "$SHARED_PROMPT_CACHE")
  fi
  cmd+=(--cached_start_index "$CACHED_START_INDEX")
  if [[ -n "${CACHED_NUM_SAMPLES}" ]]; then
    cmd+=(--cached_num_samples "$CACHED_NUM_SAMPLES")
  fi
  cmd+=(--cached_num_workers "$CACHED_NUM_WORKERS")
  cmd+=(--cached_prefetch_factor "$CACHED_PREFETCH_FACTOR")
else
  if [[ -z "${PROMPT}" ]]; then
    echo "PROMPT must be set." >&2
    exit 1
  fi
  cmd+=(--image_path "$IMAGE_PATH" --prompt "$PROMPT")
fi

if [[ -n "${LORA_PATH}" ]]; then
  cmd+=(--lora_path "$LORA_PATH")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
