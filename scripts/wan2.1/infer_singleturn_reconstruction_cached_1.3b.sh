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
export SHARED_PROMPT_CACHE=${SHARED_PROMPT_CACHE:-${CACHED_DATA_DIR}/null_prompt_embeds.pt}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/singleturn_reconstruction_cached}
export LORA_PATH=${LORA_PATH:-}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
export NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
export SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
export SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
export DTYPE=${DTYPE:-bf16}
export SEED=${SEED:-0}
export FPS=${FPS:-4}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export CACHED_START_INDEX=${CACHED_START_INDEX:-0}
export CACHED_NUM_SAMPLES=${CACHED_NUM_SAMPLES:-2}
export CACHED_NUM_WORKERS=${CACHED_NUM_WORKERS:-2}
export CACHED_PREFETCH_FACTOR=${CACHED_PREFETCH_FACTOR:-2}

if [[ ! -f "${CACHED_DATA_META}" ]]; then
  echo "Cached manifest does not exist: ${CACHED_DATA_META}" >&2
  exit 1
fi

if [[ ! -f "${SHARED_PROMPT_CACHE}" ]]; then
  echo "Null prompt cache does not exist, generating: ${SHARED_PROMPT_CACHE}"
  python scripts/wan2.1/precompute_singleturn_prompt_cache.py \
    --pretrained_model_name_or_path "$MODEL_NAME" \
    --output_path "$SHARED_PROMPT_CACHE" \
    --prompt "" \
    --dtype "$DTYPE"
fi

exec bash scripts/wan2.1/infer_singleturn_1.3b.sh
