#!/usr/bin/env bash
set -euo pipefail

export RUN_DIR=${RUN_DIR:-}
export MANIFEST_PATH=${MANIFEST_PATH:-}
export SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-480}
export SAMPLE_WIDTH=${SAMPLE_WIDTH:-832}
export MASK_BLEND_THRESHOLD=${MASK_BLEND_THRESHOLD:-0.5}
export MASK_BLEND_DILATE_KERNEL_SIZE=${MASK_BLEND_DILATE_KERNEL_SIZE:-31}
export MASK_BLEND_BLUR_KERNEL_SIZE=${MASK_BLEND_BLUR_KERNEL_SIZE:-15}
export MASK_BLEND_BLUR_SIGMA=${MASK_BLEND_BLUR_SIGMA:-4.0}
export INPLACE=${INPLACE:-1}
export OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-edgeblur}

if [[ -z "${RUN_DIR}" && -z "${MANIFEST_PATH}" ]]; then
  RUN_DIR=/mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/outputs/rordval343_rr14b_ckpt2100_objectmask/run-20260804-094907-a3e116a2
fi

if [[ -n "${RUN_DIR}" && -n "${MANIFEST_PATH}" ]]; then
  echo "Set only one of RUN_DIR or MANIFEST_PATH." >&2
  exit 1
fi

cmd=(
  python
  scripts/wan2.1/patch_singleturn_mask_blend_edge_blur.py
  --sample_size "${SAMPLE_HEIGHT}" "${SAMPLE_WIDTH}"
  --mask_blend_threshold "${MASK_BLEND_THRESHOLD}"
  --mask_blend_dilate_kernel_size "${MASK_BLEND_DILATE_KERNEL_SIZE}"
  --mask_blend_blur_kernel_size "${MASK_BLEND_BLUR_KERNEL_SIZE}"
  --mask_blend_blur_sigma "${MASK_BLEND_BLUR_SIGMA}"
  --output_suffix "${OUTPUT_SUFFIX}"
)

if [[ "${INPLACE}" == "1" ]]; then
  cmd+=(--inplace)
fi

if [[ -n "${MANIFEST_PATH}" ]]; then
  cmd+=(--manifest_path "${MANIFEST_PATH}")
else
  cmd+=(--run_dir "${RUN_DIR}")
fi

exec "${cmd[@]}"
