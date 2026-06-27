#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
if [[ -z "${CACHED_DATA_DIR:-}" ]]; then
  if [[ -d /home/data/nas_hdd/instructpix2pix ]]; then
    export CACHED_DATA_DIR=/home/data/nas_hdd/instructpix2pix/cache/singleturn_wan2.1_1.3b
  else
    export CACHED_DATA_DIR=cache/singleturn_wan2.1_1.3b
  fi
fi
export CACHED_DATA_META=${CACHED_DATA_META:-${CACHED_DATA_DIR}/manifest.json}
export SINGLETURN_RECONSTRUCTION_PROMPT_CACHE=${SINGLETURN_RECONSTRUCTION_PROMPT_CACHE:-${CACHED_DATA_DIR}/null_prompt_embeds.pt}
export SINGLETURN_NULL_PROMPT=${SINGLETURN_NULL_PROMPT:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OUTPUT_DIR=${OUTPUT_DIR:-/home/data/nas_hdd/instructpix2pix/ckpt_reconstruction}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export WANDB_MODE=${WANDB_MODE:-online}
export REPORT_TO=${REPORT_TO:-wandb}
export TRACKER_PROJECT_NAME=${TRACKER_PROJECT_NAME:-wan2.1-singleturn-reconstruction-lora-cached}
export WANDB_ENTITY=${WANDB_ENTITY:-}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-18}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-100}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
SEED=${SEED:-42}
SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
DEEPSPEED_CONFIG_FILE=${DEEPSPEED_CONFIG_FILE:-/tmp/wan2.1_singleturn_reconstruction_cached_zero2_${USER}_$$.json}

if [[ ! -f "${CACHED_DATA_META}" ]]; then
  echo "Cached manifest does not exist: ${CACHED_DATA_META}" >&2
  exit 1
fi
if [[ ! -f "${SINGLETURN_RECONSTRUCTION_PROMPT_CACHE}" ]]; then
  echo "Null prompt cache does not exist, generating: ${SINGLETURN_RECONSTRUCTION_PROMPT_CACHE}"
  python scripts/wan2.1/precompute_singleturn_prompt_cache.py \
    --pretrained_model_name_or_path "$MODEL_NAME" \
    --output_path "$SINGLETURN_RECONSTRUCTION_PROMPT_CACHE" \
    --prompt "$SINGLETURN_NULL_PROMPT" \
    --dtype bf16
fi

TOTAL_TRAIN_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE))

cat > "${DEEPSPEED_CONFIG_FILE}" <<EOF
{
    "bf16": {
        "enabled": true
    },
    "train_micro_batch_size_per_gpu": ${TRAIN_BATCH_SIZE},
    "train_batch_size": ${TOTAL_TRAIN_BATCH_SIZE},
    "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS},
    "gradient_clipping": 0.05,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true
    },
    "steps_per_print": 100,
    "wall_clock_breakdown": false
}
EOF

cmd=(
  accelerate launch
  --use_deepspeed
  --deepspeed_config_file "$DEEPSPEED_CONFIG_FILE"
  --num_processes "$NPROC_PER_NODE"
  --num_machines 1
  --dynamo_backend no
  --mixed_precision bf16
  scripts/wan2.1/train_lora.py
  --config_path config/wan2.1/wan_civitai.yaml
  --pretrained_model_name_or_path "$MODEL_NAME"
  --singleturn_mode
  --singleturn_reconstruction_mode
  --cached_data_dir "$CACHED_DATA_DIR"
  --cached_data_meta "$CACHED_DATA_META"
  --singleturn_reconstruction_prompt_cache "$SINGLETURN_RECONSTRUCTION_PROMPT_CACHE"
  --singleturn_sample_size "$SAMPLE_HEIGHT" "$SAMPLE_WIDTH"
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
  --use_deepspeed
)

echo "Running: ${cmd[*]}"
"${cmd[@]}"
