#!/bin/bash
# Step 2 (MMFlash only): score how much every response token of the training set depends on the image.
#
#   d_t = KL( p(. | prefix, image) || p(. | prefix) )   on the target model, plus the entropy of the with-image
#   distribution (the confidence gate of data.visual_score_confidence_gate).
#
# Reads data.train_data_path, data.max_length and the target from the training config, runs one shard per GPU and
# writes them into data.visual_score_path (or OUTPUT_DIR), which training then reads. Re-running resumes every shard;
# a completeness check against the training set runs at the end.
#
#   GPUS=0,1,2,3 bash scripts/score_visual_kl.sh configs/train/qwen3.5-4b-mmflash.yaml
#   MAX_ROWS=200 GPUS=0 bash scripts/score_visual_kl.sh configs/train/qwen3.5-4b-mmflash.yaml   # smoke test
set -uo pipefail
if [ $# -lt 1 ]; then
    sed -n '2,14p' "$0"; exit 1
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG=$1
export PYTHONUNBUFFERED=1 PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
read -r CFG_OUT TRAIN_DATA < <(python - "${CONFIG}" <<'PY'
import sys, yaml
data = (yaml.safe_load(open(sys.argv[1])) or {}).get("data") or {}
print(data.get("visual_score_path") or "-", data.get("train_data_path") or "-")
PY
)
OUTPUT_DIR="${OUTPUT_DIR:-${CFG_OUT}}"
if [ "${OUTPUT_DIR}" = "-" ]; then echo "set data.visual_score_path in ${CONFIG} (or OUTPUT_DIR=...)" >&2; exit 1; fi
if [ ! -f "${TRAIN_DATA}" ]; then echo "data.train_data_path does not exist: ${TRAIN_DATA}" >&2; exit 1; fi
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
N=${#GPU_LIST[@]}
LOG_DIR="${OUTPUT_DIR}/logs"; mkdir -p "${LOG_DIR}"
ROWS=$(wc -l < "${TRAIN_DATA}")
echo "[score] ${TRAIN_DATA} (${ROWS} rows) -> ${OUTPUT_DIR}, ${N} shard(s) on GPU(s) ${GPU_LIST[*]}"
PIDS=()
for i in "${!GPU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" python scripts/score_visual_kl.py --config "${CONFIG}" --output-dir "${OUTPUT_DIR}" \
        --shard-index "${i}" --num-shards "${N}" --expected-rows "${ROWS}" ${MAX_ROWS:+--max-rows "${MAX_ROWS}"} \
        > "${LOG_DIR}/shard${i}.log" 2>&1 &
    PIDS+=("$!")
done
status=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "[score] shard ${i} failed, see ${LOG_DIR}/shard${i}.log" >&2; status=1; }
done
[ "${status}" -eq 0 ] || exit 1
[ -n "${MAX_ROWS:-}" ] && { echo "[score] smoke run done (MAX_ROWS=${MAX_ROWS}); skipping the completeness check"; exit 0; }
python scripts/check_visual_kl_complete.py --config "${CONFIG}" --sidecar "${OUTPUT_DIR}"
