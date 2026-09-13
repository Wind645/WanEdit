#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT=${EVAL_ROOT:-/mnt/cpfs/jiachengliu/code/object_removal/Evaluation_content_crop}
PYTHON_BIN=${PYTHON_BIN:-/mnt/cpfs/jiachengliu/envs/evaluation/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR to the inference output root (contains rank0..rankN).}
RAW_DATA_DIR=${RAW_DATA_DIR:?Set RAW_DATA_DIR (contains gt/, input/, object_mask/, condition_mask/).}
GPUS="${GPUS:-0 1 2 3 4 5 6}"
RUN_NAME=${RUN_NAME:-eval_$(date +%Y%m%d_%H%M%S)}
MASK_TYPE=${MASK_TYPE:-object_mask}
PRED_SUFFIX=${PRED_SUFFIX:-_frame8_pure.png}
# METRIC_GROUPS="${METRIC_GROUPS:-psnr,ssim,lpips fid cmmd as cfd remove}"
METRIC_GROUPS="${METRIC_GROUPS:-psnr,ssim,lpips fid remove cfd}"
# METRIC_GROUPS="${METRIC_GROUPS:-cfd}"
FID_BATCH_SIZE=${FID_BATCH_SIZE:-1}
CMMD_BATCH_SIZE=${CMMD_BATCH_SIZE:-32}

EVAL_DIR=${OUTPUT_DIR}/eval_inputs/${RUN_NAME}
PRED_COLLECT_DIR=${EVAL_DIR}/prediction
GT_COLLECT_DIR=${EVAL_DIR}/gt
MASK_COLLECT_DIR=${EVAL_DIR}/mask
EVAL_OUTPUT_ROOT=${EVAL_DIR}/results

source "${EVAL_ROOT}/model_env.sh"

if [[ -e "${EVAL_DIR}" ]]; then
    echo "ERROR: ${EVAL_DIR} already exists. Use a different RUN_NAME." >&2
    exit 1
fi

mkdir -p "${PRED_COLLECT_DIR}" "${GT_COLLECT_DIR}" "${MASK_COLLECT_DIR}" "${EVAL_OUTPUT_ROOT}/logs"

echo "=== Evaluation Setup ==="
echo "Output dir:      ${OUTPUT_DIR}"
echo "Raw data dir:    ${RAW_DATA_DIR}"
echo "Mask type:       ${MASK_TYPE}"
echo "Pred suffix:     ${PRED_SUFFIX}"
echo "Metric groups:   ${METRIC_GROUPS}"
echo "FID batch size:  ${FID_BATCH_SIZE}"
echo "GPUs:            ${GPUS}"
echo "Eval workspace:  ${EVAL_DIR}"
echo ""

echo "Collecting predictions, GT, and masks..."

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${RAW_DATA_DIR}" "${PRED_COLLECT_DIR}" "${GT_COLLECT_DIR}" "${MASK_COLLECT_DIR}" "${MASK_TYPE}" "${PRED_SUFFIX}" << 'PYEOF'
import json, sys
from pathlib import Path
from PIL import Image

output_dir = Path(sys.argv[1])
raw_data_dir = Path(sys.argv[2])
pred_collect = Path(sys.argv[3])
gt_collect = Path(sys.argv[4])
mask_collect = Path(sys.argv[5])
mask_type = sys.argv[6]
pred_suffix = sys.argv[7]

# Build prediction index: triplet -> file path
pred_index = {}
for pred_file in output_dir.rglob(f"*{pred_suffix}"):
    if "_tail_" in pred_file.name:
        continue
    triplet = pred_file.name.replace(pred_suffix, "")
    pred_index[triplet] = pred_file
print(f"Found {len(pred_index)} predictions in {output_dir}")

manifests = []
for mf in sorted(output_dir.glob("rank*/raw_infer_manifest.json")):
    with open(mf) as f:
        manifests.extend(json.load(f))
if not manifests:
    root_mf = output_dir / "raw_infer_manifest.json"
    if root_mf.exists():
        with open(root_mf) as f:
            manifests.extend(json.load(f))

if not manifests:
    print("WARNING: No raw_infer_manifest.json found. Falling back to filename matching.")
    for triplet in sorted(pred_index.keys()):
        manifests.append({"rel_triplet": triplet})

print(f"Total samples from manifests: {len(manifests)}")

collected = 0
for record in manifests:
    triplet = record["rel_triplet"]

    # Prediction
    pred_src = pred_index.get(triplet)
    if pred_src is None or not pred_src.exists():
        print(f"WARNING: missing prediction for {triplet}")
        continue

    # GT
    gt_src = None
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = raw_data_dir / "gt" / f"{triplet}{ext}"
        if candidate.exists():
            gt_src = candidate
            break
    if gt_src is None:
        print(f"WARNING: missing GT for {triplet}")
        continue

    # Mask
    mask_src = None
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = raw_data_dir / mask_type / f"{triplet}{ext}"
        if candidate.exists():
            mask_src = candidate
            break
    if mask_src is None:
        print(f"WARNING: missing mask for {triplet}")
        continue

    # Use GT size as reference (upsample prediction to GT if needed)
    gt_img = Image.open(gt_src)
    gt_size = gt_img.size
    pred_img = Image.open(pred_src)
    pred_size = pred_img.size

    # Prediction: resize to GT size if needed (LANCZOS upsample)
    dst = pred_collect / f"{triplet}.png"
    if pred_size != gt_size:
        pred_img.convert("RGB").resize(gt_size, resample=Image.LANCZOS).save(dst)
    elif pred_src.suffix == ".png":
        dst.symlink_to(pred_src)
    else:
        pred_img.save(dst)

    # GT: keep original size
    dst = gt_collect / f"{triplet}.png"
    if gt_src.suffix == ".png":
        dst.symlink_to(gt_src)
    else:
        gt_img.save(dst)

    # Mask: resize to GT size if needed (use NEAREST for masks)
    # Invert mask for anomalous sample 000088 (bench300 dataset quirk)
    mask_img = Image.open(mask_src)
    invert_mask = (triplet == "000088")
    dst = mask_collect / f"{triplet}.png"
    if invert_mask:
        from PIL import ImageChops
        mask_bin = mask_img.convert("L").point(lambda v: 255 if v > 127 else 0)
        mask_bin = ImageChops.invert(mask_bin)
        if mask_bin.size != gt_size:
            mask_bin = mask_bin.resize(gt_size, resample=Image.NEAREST)
        mask_bin.save(dst)
    elif mask_img.size != gt_size:
        mask_img.resize(gt_size, resample=Image.NEAREST).save(dst)
    elif mask_src.suffix == ".png":
        dst.symlink_to(mask_src)
    else:
        mask_img.save(dst)

    collected += 1

print(f"Collected: {collected}/{len(manifests)}")
if collected == 0:
    print("ERROR: No samples collected.", file=sys.stderr)
    sys.exit(1)
PYEOF

PRED_COUNT=$(ls "${PRED_COLLECT_DIR}" | wc -l)
GT_COUNT=$(ls "${GT_COLLECT_DIR}" | wc -l)
MASK_COUNT=$(ls "${MASK_COLLECT_DIR}" | wc -l)
echo "Predictions: ${PRED_COUNT}, GT: ${GT_COUNT}, Masks: ${MASK_COUNT}"

if [[ "${PRED_COUNT}" -ne "${GT_COUNT}" ]]; then
    echo "ERROR: prediction/GT count mismatch!" >&2
    exit 1
fi

echo ""
echo "=== Launching parallel evaluation ==="

gpu_list=(${GPUS})
if [[ "${#gpu_list[@]}" -eq 0 ]]; then
    echo "ERROR: GPUS is empty" >&2
    exit 1
fi

pids=()
names=()
slot=0

wait_for_all() {
    local failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[done] ${names[$i]}"
        else
            echo "[FAIL] ${names[$i]}" >&2
            failed=1
        fi
    done
    pids=()
    names=()
    if [[ "${failed}" -ne 0 ]]; then
        echo "At least one eval job failed. Check logs under ${EVAL_OUTPUT_ROOT}/logs" >&2
        exit 1
    fi
}

for metrics in ${METRIC_GROUPS}; do
    gpu="${gpu_list[$slot]}"
    label="${metrics//,/_}"
    log_path="${EVAL_OUTPUT_ROOT}/logs/${label}_gpu${gpu}.log"
    out_dir="${EVAL_OUTPUT_ROOT}/${label}"

    echo "[launch] metrics=${metrics} gpu=${gpu} log=${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${EVAL_ROOT}/unify_eval.py" \
        "${PRED_COLLECT_DIR}" \
        "${GT_COLLECT_DIR}" \
        --mask_dir "${MASK_COLLECT_DIR}" \
        --spatial_protocol full_frame \
        --output_dir "${out_dir}" \
        --metrics "${metrics}" \
        --fid_batch_size "${FID_BATCH_SIZE}" \
        --cmmd_batch_size "${CMMD_BATCH_SIZE}" \
        --device cuda:0 \
        >"${log_path}" 2>&1 &

    pids+=("$!")
    names+=("${label}_gpu${gpu}")
    slot=$((slot + 1))

    if [[ "${slot}" -ge "${#gpu_list[@]}" ]]; then
        wait_for_all
        slot=0
    fi
done

if [[ "${#pids[@]}" -gt 0 ]]; then
    wait_for_all
fi

echo ""
echo "=== All metrics complete ==="
echo "Results: ${EVAL_OUTPUT_ROOT}"

# Print summary from each metric group
for metrics in ${METRIC_GROUPS}; do
    label="${metrics//,/_}"
    summary_file="${EVAL_OUTPUT_ROOT}/${label}/summary.json"
    if [[ -f "${summary_file}" ]]; then
        echo ""
        echo "--- ${label} ---"
        "${PYTHON_BIN}" -c "
import json
with open('${summary_file}') as f:
    d = json.load(f)
for k, v in d.get('summary', {}).items():
    if isinstance(v, float):
        print(f'  {k}: {v:.4f}')
    elif isinstance(v, dict):
        print(f'  {k}: {v}')
    else:
        print(f'  {k}: {v}')
"
    fi
done
