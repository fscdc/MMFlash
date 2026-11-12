#!/bin/bash
# Step 4: export a training checkpoint as a draft directory that SGLang and vLLM load directly.
#
#   bash scripts/export_draft.sh <checkpoint> <draft config json> <output dir> [block size, default 16]
#   bash scripts/export_draft.sh outputs/qwen3.5-4b-mmflash/qwen3.5-4b-mmflash-step100000 \
#       configs/qwen3.5-4b-mmflash.json outputs/qwen3.5-4b-mmflash/export
set -euo pipefail
if [ $# -lt 3 ]; then
    sed -n '2,7p' "$0"; exit 1
fi
CHECKPOINT=$1 DRAFT_CONFIG=$2 OUTPUT_DIR=$3 BLOCK_SIZE=${4:-16}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m specforge.cli export --to hf --checkpoint "${CHECKPOINT}" --draft-config "${DRAFT_CONFIG}" --output-dir "${OUTPUT_DIR}"
python scripts/normalize_dflash_export.py --config "${OUTPUT_DIR}/config.json" --block-size "${BLOCK_SIZE}"
echo "[export] ${OUTPUT_DIR} is ready: --speculative-draft-model-path ${OUTPUT_DIR} (SGLang) / \"model\": \"${OUTPUT_DIR}\" (vLLM)"
