#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_DIR=${OUTPUT_DIR:-outputs/scribble_preview}
export SEED=${SEED:-0}
export NUM_SAMPLES=${NUM_SAMPLES:-6}
export PREVIEW=${PREVIEW:-1}

mask_dirs=(
  /mnt/cpfs/jiachengliu/dataset/CORNE/mask-check
  /mnt/cpfs/jiachengliu/dataset/CORNE/mask_sam
)

for mask_dir in "${mask_dirs[@]}"; do
  if [[ ! -d "${mask_dir}" ]]; then
    echo "Missing mask dir: ${mask_dir}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

for mask_dir in "${mask_dirs[@]}"; do
  tag=$(basename "${mask_dir}")
  out_dir="${OUTPUT_DIR}/${tag}"
  mkdir -p "${out_dir}"
  python scripts/wan2.1/simulate_user_scribble.py \
    --mask_dir "${mask_dir}" \
    --output_dir "${out_dir}" \
    --seed "${SEED}" \
    --num_samples "${NUM_SAMPLES}" \
    --preview
done
