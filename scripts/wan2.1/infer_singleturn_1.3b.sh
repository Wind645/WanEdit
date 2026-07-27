#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-/mnt/cpfs/jiachengliu/pretrained_models/Wan-AI/Wan2.1-T2V-1.3B}
export IMAGE_PATH=${IMAGE_PATH:-}
export MASK_PATH=${MASK_PATH:-}
export PROMPT=${PROMPT:-}
export CACHED_SAMPLE_PATH=${CACHED_SAMPLE_PATH:-}
export RAW_DATA_DIR=${RAW_DATA_DIR:-}
export RAW_SELECTED_TRIPLETS=${RAW_SELECTED_TRIPLETS:-}
export CACHED_DATA_DIR=${CACHED_DATA_DIR:-/mnt/cpfs/jiachengliu/dataset/CORNE/cache/singleturn_object_removal_wan2.1_1.3b_sam_strict_keyframe_cache_v1}
export CACHED_DATA_META=${CACHED_DATA_META:-${CACHED_DATA_DIR}/manifest.json}
export SHARED_PROMPT_CACHE=${SHARED_PROMPT_CACHE:-}
export CACHED_START_INDEX=${CACHED_START_INDEX:-0}
export CACHED_NUM_SAMPLES=${CACHED_NUM_SAMPLES:-}
export CACHED_NUM_WORKERS=${CACHED_NUM_WORKERS:-2}
export CACHED_PREFETCH_FACTOR=${CACHED_PREFETCH_FACTOR:-2}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/gamma2}
export LORA_PATH=${LORA_PATH:-/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/nonlin_gamma2/checkpoint-1500/lora_diffusion_pytorch_model.safetensors}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ENABLE_REFINEMENT=${ENABLE_REFINEMENT:-0}
export REFINEMENT_LORA_PATH=${REFINEMENT_LORA_PATH:-}
export REFINEMENT_LORA_ALPHA=${REFINEMENT_LORA_ALPHA:-1.0}
export REFINEMENT_GUIDANCE_SCALE=${REFINEMENT_GUIDANCE_SCALE:-1.0}

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}
SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-${SAMPLE_SIZE:-480}}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-${SAMPLE_SIZE:-832}}
SINGLETURN_CACHE_CORRUPTION_FRAMES=${SINGLETURN_CACHE_CORRUPTION_FRAMES:-2}
SINGLETURN_CACHE_RESTORATION_FRAMES=${SINGLETURN_CACHE_RESTORATION_FRAMES:-5}
SINGLETURN_CACHE_INTERPOLATION_GAMMA=${SINGLETURN_CACHE_INTERPOLATION_GAMMA:-2.0}
SEED=${SEED:-0}
FPS=${FPS:-4}
DTYPE=${DTYPE:-bf16}
ACCELERATE_MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION:-bf16}

if [[ "${DTYPE}" == "fp16" ]]; then
  ACCELERATE_MIXED_PRECISION=fp16
elif [[ "${DTYPE}" == "fp32" ]]; then
  ACCELERATE_MIXED_PRECISION=no
fi

if [[ -z "${RAW_DATA_DIR}" && -z "${IMAGE_PATH}" ]]; then
  if [[ -z "${CACHED_SAMPLE_PATH}" && -z "${CACHED_DATA_META}" ]]; then
    echo "Set RAW_DATA_DIR, or IMAGE_PATH/MASK_PATH, or CACHED_SAMPLE_PATH/CACHED_DATA_META." >&2
    exit 1
  fi
fi

cmd_base=(
  scripts/wan2.1/singleturn_edit_infer.py
  --pretrained_model_name_or_path "$MODEL_NAME"
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --sample_size "$SAMPLE_HEIGHT" "$SAMPLE_WIDTH"
  --singleturn_cache_corruption_frames "$SINGLETURN_CACHE_CORRUPTION_FRAMES"
  --singleturn_cache_restoration_frames "$SINGLETURN_CACHE_RESTORATION_FRAMES"
  --singleturn_cache_interpolation_gamma "$SINGLETURN_CACHE_INTERPOLATION_GAMMA"
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

if [[ -n "${RAW_DATA_DIR}" ]]; then
  if [[ -n "${IMAGE_PATH}" || -n "${MASK_PATH}" || -n "${CACHED_SAMPLE_PATH}" ]]; then
    echo "RAW_DATA_DIR mode does not accept IMAGE_PATH/MASK_PATH or CACHED_SAMPLE_PATH." >&2
    exit 1
  fi
  cmd+=(--raw_data_dir "$RAW_DATA_DIR")
  if [[ -n "${RAW_SELECTED_TRIPLETS}" ]]; then
    cmd+=(--raw_selected_triplets "$RAW_SELECTED_TRIPLETS")
  fi
  cmd+=(--cached_start_index "$CACHED_START_INDEX")
  if [[ -n "${CACHED_NUM_SAMPLES}" ]]; then
    cmd+=(--cached_num_samples "$CACHED_NUM_SAMPLES")
  fi
elif [[ -n "${IMAGE_PATH}" ]]; then
  if [[ -z "${MASK_PATH}" ]]; then
    echo "MASK_PATH must be set in image mode." >&2
    exit 1
  fi
  if [[ -n "${PROMPT}" ]]; then
    echo "PROMPT is ignored in CORNE object-removal image mode." >&2
  fi
  cmd+=(--image_path "$IMAGE_PATH" --mask_path "$MASK_PATH")
elif [[ -n "${CACHED_SAMPLE_PATH}" || -n "${CACHED_DATA_META}" ]]; then
  if [[ -n "${MASK_PATH}" ]]; then
    echo "Cached mode does not accept MASK_PATH." >&2
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
  echo "Set RAW_DATA_DIR, or IMAGE_PATH/MASK_PATH, or CACHED_SAMPLE_PATH/CACHED_DATA_META." >&2
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
"${cmd[@]}"
