#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
export IMAGE_PATH=${IMAGE_PATH:-}
export MASK_PATH=${MASK_PATH:-}
export RAW_SOURCE_DIR=${RAW_SOURCE_DIR:-}
export RAW_MASK_DIR=${RAW_MASK_DIR:-}
export RAW_GT_DIR=${RAW_GT_DIR:-}
export PROMPT=${PROMPT:-}
export CACHED_SAMPLE_PATH=${CACHED_SAMPLE_PATH:-}
export CACHED_DATA_META=${CACHED_DATA_META:-/home/data/nas_hdd/CORNE_extracted/cache/singleturn_object_removal_wan2.1_1.3b_v3_twoprefix/manifest.json}
export CACHED_DATA_DIR=${CACHED_DATA_DIR:-/home/data/nas_hdd/CORNE_extracted/cache/singleturn_object_removal_wan2.1_1.3b_v3_twoprefix}
export SHARED_PROMPT_CACHE=${SHARED_PROMPT_CACHE:-}
export CACHED_START_INDEX=${CACHED_START_INDEX:-0}
export CACHED_NUM_SAMPLES=${CACHED_NUM_SAMPLES:-}
export CACHED_NUM_WORKERS=${CACHED_NUM_WORKERS:-2}
export CACHED_PREFETCH_FACTOR=${CACHED_PREFETCH_FACTOR:-2}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/singleturn_object_removal_v3_twoprefix}
export VIDEO_FORMAT=${VIDEO_FORMAT:-gif}
export LORA_PATH=${LORA_PATH:-}
export ENABLE_REFINEMENT=${ENABLE_REFINEMENT:-0}
export REFINEMENT_LORA_PATH=${REFINEMENT_LORA_PATH:-}
export REFINEMENT_LORA_ALPHA=${REFINEMENT_LORA_ALPHA:-1.0}
export REFINEMENT_GUIDANCE_SCALE=${REFINEMENT_GUIDANCE_SCALE:-1.0}

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

cmd_base=(
  scripts/wan2.1/singleturn_edit_infer.py
  --pretrained_model_name_or_path "$MODEL_NAME"
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --sample_size "$SAMPLE_HEIGHT" "$SAMPLE_WIDTH"
  --seed "$SEED"
  --fps "$FPS"
  --dtype "$DTYPE"
  --video_format "$VIDEO_FORMAT"
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

if [[ -n "${RAW_SOURCE_DIR}" || -n "${RAW_MASK_DIR}" || -n "${RAW_GT_DIR}" ]]; then
  if [[ -z "${RAW_SOURCE_DIR}" || -z "${RAW_MASK_DIR}" || -z "${RAW_GT_DIR}" ]]; then
    echo "RAW_SOURCE_DIR, RAW_MASK_DIR, and RAW_GT_DIR must all be set in raw-folder mode." >&2
    exit 1
  fi
  if [[ -n "${IMAGE_PATH}" || -n "${MASK_PATH}" ]]; then
    echo "Raw-folder mode does not accept IMAGE_PATH/MASK_PATH." >&2
    exit 1
  fi
  if [[ -n "${PROMPT}" ]]; then
    echo "PROMPT is ignored in CORNE object-removal raw-folder mode." >&2
  fi
  cmd+=(
    --raw_source_dir "$RAW_SOURCE_DIR"
    --raw_mask_dir "$RAW_MASK_DIR"
    --raw_gt_dir "$RAW_GT_DIR"
  )
elif [[ -n "${IMAGE_PATH}" || -n "${MASK_PATH}" ]]; then
  if [[ -z "${IMAGE_PATH}" || -z "${MASK_PATH}" ]]; then
    echo "IMAGE_PATH and MASK_PATH must both be set in image mode." >&2
    exit 1
  fi
  if [[ -n "${PROMPT}" ]]; then
    echo "PROMPT is ignored in CORNE object-removal image mode." >&2
  fi
  cmd+=(--image_path "$IMAGE_PATH" --mask_path "$MASK_PATH")
elif [[ -n "${CACHED_SAMPLE_PATH}" || -n "${CACHED_DATA_META}" ]]; then
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
  echo "Set either RAW_SOURCE_DIR/RAW_MASK_DIR/RAW_GT_DIR, IMAGE_PATH/MASK_PATH, or CACHED_SAMPLE_PATH/CACHED_DATA_META." >&2
  exit 1
fi

if [[ -n "${LORA_PATH}" ]]; then
  cmd+=(--lora_path "$LORA_PATH")
fi

if [[ "${ENABLE_REFINEMENT}" == "1" ]]; then
  if [[ -z "${REFINEMENT_LORA_PATH}" ]]; then
    echo "REFINEMENT_LORA_PATH must be set when ENABLE_REFINEMENT=1." >&2
    exit 1
  fi
  cmd+=(
    --enable_refinement
    --refinement_lora_path "$REFINEMENT_LORA_PATH"
    --refinement_lora_alpha "$REFINEMENT_LORA_ALPHA"
    --refinement_guidance_scale "$REFINEMENT_GUIDANCE_SCALE"
  )
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}" "$@"
