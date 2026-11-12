#!/bin/bash
# Step 3: train a draft. The YAML fully describes the run, including which GPUs host the SGLang capture server and
# which train (deployment.disaggregated.managed_local); `specforge train` starts and supervises every process.
#
#   bash scripts/train.sh configs/train/qwen3.5-4b-mmflash.yaml [dotted.overrides=value ...]
#   bash scripts/train.sh configs/train/qwen3.5-4b-mmflash.yaml training.max_steps=100   # a quick smoke run
#
# Needs the spec-capture patch in the SGLang that serves the target (bash scripts/apply_sglang_spec_capture_patch.sh)
# and, for Gemma 4 targets, the gemma4-spec patch (bash scripts/apply_sglang_gemma4_patch.sh).
set -euo pipefail
if [ $# -lt 1 ]; then
    sed -n '2,10p' "$0"; exit 1
fi
CONFIG=$1; shift
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# one compile cache per host: concurrent JIT compiles into a shared cache directory can race on network filesystems
CACHE_ROOT="${SPECFORGE_CACHE_ROOT:-${HOME}/.cache/mmflash}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton-$(hostname -s)}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/inductor-$(hostname -s)}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
exec python -m specforge.cli train --config "${CONFIG}" "$@"
