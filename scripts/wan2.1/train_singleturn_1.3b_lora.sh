#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
export SINGLETURN_DATA_DIR=${SINGLETURN_DATA_DIR:-/home/data/nas_hdd/Singleturn}
export SINGLETURN_MANIFEST=${SINGLETURN_MANIFEST:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OUTPUT_DIR=${OUTPUT_DIR:-experiments/wan2.1_1.3b_singleturn_lora}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-Edit the source image according to this instruction: {prompt}}
export WANDB_MODE=${WANDB_MODE:-online}
export REPORT_TO=${REPORT_TO:-wandb}
export TRACKER_PROJECT_NAME=${TRACKER_PROJECT_NAME:-wan2.1-singleturn-lora}
export SINGLETURN_VALIDATION_IMAGE_PATH=${SINGLETURN_VALIDATION_IMAGE_PATH:-}
export SINGLETURN_VALIDATION_PROMPT=${SINGLETURN_VALIDATION_PROMPT:-}
export SINGLETURN_VALIDATION_NEGATIVE_PROMPT=${SINGLETURN_VALIDATION_NEGATIVE_PROMPT:-}

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-500}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
SEED=${SEED:-42}
SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
SINGLETURN_VALIDATION_GUIDANCE_SCALE=${SINGLETURN_VALIDATION_GUIDANCE_SCALE:-5.0}
SINGLETURN_VALIDATION_NUM_INFERENCE_STEPS=${SINGLETURN_VALIDATION_NUM_INFERENCE_STEPS:-50}
SINGLETURN_VALIDATION_FPS=${SINGLETURN_VALIDATION_FPS:-4}

cmd=(
  accelerate launch
  --use_deepspeed
  --deepspeed_config_file config/1.3b_lora_zero_stage2_1node.json
  --num_processes "$NPROC_PER_NODE"
  --num_machines 1
  --dynamo_backend no
  --mixed_precision bf16
  scripts/wan2.1/train_lora.py
  --config_path config/wan2.1/wan_civitai.yaml
  --pretrained_model_name_or_path "$MODEL_NAME"
  --singleturn_mode
  --train_data_dir "$SINGLETURN_DATA_DIR"
  --singleturn_sample_size "$SAMPLE_HEIGHT" "$SAMPLE_WIDTH"
  --prompt_template "$PROMPT_TEMPLATE"
  --report_to "$REPORT_TO"
  --tracker_project_name "$TRACKER_PROJECT_NAME"
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
  --vae_mini_batch "$TRAIN_BATCH_SIZE"
  --max_grad_norm 0.05
  --uniform_sampling
  --low_vram
  --use_deepspeed
)

if [[ -n "${SINGLETURN_MANIFEST}" ]]; then
  cmd+=(--train_data_manifest "$SINGLETURN_MANIFEST")
fi

if [[ -n "${SINGLETURN_VALIDATION_IMAGE_PATH}" || -n "${SINGLETURN_VALIDATION_PROMPT}" ]]; then
  cmd+=(
    --singleturn_validation_image_path "$SINGLETURN_VALIDATION_IMAGE_PATH"
    --singleturn_validation_prompt "$SINGLETURN_VALIDATION_PROMPT"
    --singleturn_validation_negative_prompt "$SINGLETURN_VALIDATION_NEGATIVE_PROMPT"
    --singleturn_validation_guidance_scale "$SINGLETURN_VALIDATION_GUIDANCE_SCALE"
    --singleturn_validation_num_inference_steps "$SINGLETURN_VALIDATION_NUM_INFERENCE_STEPS"
    --singleturn_validation_fps "$SINGLETURN_VALIDATION_FPS"
  )
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
