#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
export IMAGE_PATH=${IMAGE_PATH:-}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/singleturn_reconstruction_image}
export LORA_PATH=${LORA_PATH:-}
export PROMPT=${PROMPT:-}
export PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-{prompt}}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
export NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
export SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
export SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
export DTYPE=${DTYPE:-bf16}
export SEED=${SEED:-0}
export FPS=${FPS:-4}

if [[ -z "${IMAGE_PATH}" ]]; then
  echo "IMAGE_PATH must be set." >&2
  exit 1
fi

exec bash scripts/wan2.1/infer_singleturn_1.3b.sh
