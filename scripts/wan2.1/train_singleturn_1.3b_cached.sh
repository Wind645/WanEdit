#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

export PYTHON_BIN=${PYTHON_BIN:-/home/data/zhikai/miniconda3/envs/videocof/bin/python}
export ACCELERATE_BIN=${ACCELERATE_BIN:-/home/data/zhikai/miniconda3/envs/videocof/bin/accelerate}
export MODEL_NAME=${MODEL_NAME:-/home/data/zhikai/VideoCoF/models/Wan2.1-T2V-1.3B}
export CACHED_DATA_DIR=${CACHED_DATA_DIR:-/home/data/nas_hdd/CORNE_extracted/cache/singleturn_object_removal_wan2.1_1.3b_sam_strict_keyframe_cache_v1_60000_mixedmask_fixed}
export CACHED_DATA_META=${CACHED_DATA_META:-${CACHED_DATA_DIR}/manifest.json}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,5}
export OUTPUT_DIR=${OUTPUT_DIR:-/home/data/nas_hdd/CORNE_extracted/ckpt/f11_test_loss_scalar}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export WANDB_MODE=${WANDB_MODE:-online}
export REPORT_TO=${REPORT_TO:-wandb}
export TRACKER_PROJECT_NAME=${TRACKER_PROJECT_NAME:-wan2.1-singleturn-object-removal-sam-strict-keyframe-cache-v1-60000-mixedmask-fixed}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
export SAVE_STATE=${SAVE_STATE:-1}
export LORA_INIT_PATH=${LORA_INIT_PATH:-}
export SINGLETURN_VALIDATION_IMAGE_PATH=${SINGLETURN_VALIDATION_IMAGE_PATH:-}
export SINGLETURN_VALIDATION_MASK_PATH=${SINGLETURN_VALIDATION_MASK_PATH:-}
export SINGLETURN_VALIDATION_NEGATIVE_PROMPT=${SINGLETURN_VALIDATION_NEGATIVE_PROMPT:-}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-}

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-3}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-10}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-100}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
SEED=${SEED:-42}
SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
CORRUPTION_FRAME=${CORRUPTION_FRAME:-2}
RESTORATION_FRAME=${RESTORATION_FRAME:-5}
SINGLETURN_VALIDATION_GUIDANCE_SCALE=${SINGLETURN_VALIDATION_GUIDANCE_SCALE:-5.0}
SINGLETURN_VALIDATION_NUM_INFERENCE_STEPS=${SINGLETURN_VALIDATION_NUM_INFERENCE_STEPS:-50}
SINGLETURN_VALIDATION_FPS=${SINGLETURN_VALIDATION_FPS:-4}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter does not exist: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -x "${ACCELERATE_BIN}" ]]; then
  echo "Accelerate launcher does not exist: ${ACCELERATE_BIN}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "Model directory does not exist: ${MODEL_NAME}" >&2
  exit 1
fi

if [[ ! -f "${CACHED_DATA_META}" ]]; then
  echo "Cached manifest does not exist: ${CACHED_DATA_META}" >&2
  exit 1
fi

if [[ "${NPROC_PER_NODE}" -le 0 || "${TRAIN_BATCH_SIZE}" -le 0 || "${GRADIENT_ACCUMULATION_STEPS}" -le 0 ]]; then
  echo "NPROC_PER_NODE, TRAIN_BATCH_SIZE, and GRADIENT_ACCUMULATION_STEPS must all be positive." >&2
  exit 1
fi

cmd=(
  "$ACCELERATE_BIN" launch
  --num_processes "$NPROC_PER_NODE"
  --num_machines 1
  --dynamo_backend no
  --mixed_precision bf16
  "$REPO_ROOT/scripts/wan2.1/train_lora.py"
  --config_path "$REPO_ROOT/config/wan2.1/wan_civitai.yaml"
  --pretrained_model_name_or_path "$MODEL_NAME"
  --singleturn_mode
  --cached_data_dir "$CACHED_DATA_DIR"
  --cached_data_meta "$CACHED_DATA_META"
  --singleturn_sample_size "$SAMPLE_HEIGHT" "$SAMPLE_WIDTH"
  --corruption_frame "$CORRUPTION_FRAME"
  --restoration_frame "$RESTORATION_FRAME"
  --report_to "$REPORT_TO"
  --tracker_project_name "$TRACKER_PROJECT_NAME"
  --tracker_entity "$WANDB_ENTITY"
  --rank 128
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --learning_rate "$LEARNING_RATE"
  --seed "$SEED"
  --output_dir "$OUTPUT_DIR"
  --gradient_checkpointing
  --mixed_precision bf16
  --adam_weight_decay 3e-2
  --adam_epsilon 1e-10
  --max_grad_norm 0.05
  --uniform_sampling
)

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  cmd+=(
    --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
  )
fi

if [[ -n "${LORA_INIT_PATH}" ]]; then
  cmd+=(
    --lora_init_path "$LORA_INIT_PATH"
  )
fi

if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
  cmd+=(
    --max_train_steps "$MAX_TRAIN_STEPS"
  )
fi

if [[ "${SAVE_STATE}" == "1" ]]; then
  cmd+=(
    --save_state
  )
fi

if [[ -n "${SINGLETURN_VALIDATION_IMAGE_PATH}" || -n "${SINGLETURN_VALIDATION_MASK_PATH}" ]]; then
  cmd+=(
    --singleturn_validation_image_path "$SINGLETURN_VALIDATION_IMAGE_PATH"
    --singleturn_validation_mask_path "$SINGLETURN_VALIDATION_MASK_PATH"
    --singleturn_validation_negative_prompt "$SINGLETURN_VALIDATION_NEGATIVE_PROMPT"
    --singleturn_validation_guidance_scale "$SINGLETURN_VALIDATION_GUIDANCE_SCALE"
    --singleturn_validation_num_inference_steps "$SINGLETURN_VALIDATION_NUM_INFERENCE_STEPS"
    --singleturn_validation_fps "$SINGLETURN_VALIDATION_FPS"
  )
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
