#!/bin/bash
# Image benchmarks as in the paper: ChartQA, CharXiv (reasoning), MMStar, MMBench, DynaMath, MathVista and
# MathVerse, 200 questions each, one request in flight, greedy decoding, at most 4096 new tokens, reasoning off.
#
#   METHOD=mmflash MODEL=Qwen/Qwen3.5-4B DRAFT=FSCCS/Qwen3.5-4B-MMFlash bash scripts/eval_image.sh
#   METHOD=origin  MODEL=Qwen/Qwen3.5-4B bash scripts/eval_image.sh          # the autoregressive baseline
#
# METHOD: origin (no speculation) or mmflash (any DFLASH-format draft, e.g. a DFlash baseline export). Each run
# writes results/<NAME>_results.jsonl: per benchmark the accept length, decode throughput, accuracy and per-question
# details. Speedup = decode throughput of the draft run / that of the origin run on the same GPU.
# Optional: TEMPERATURE=1 (samples with top-p 0.95, top-k 20), GPU, PORT, BLOCK_SIZE, BENCHES, NAME, RESULTS_DIR.
# Run `bash scripts/apply_sglang_bench_patches.sh` once beforehand so that the server reports per-request decode
# latency (the throughput the tables use); without it bench_mm.py falls back to end-to-end latency.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
METHOD="${METHOD:-mmflash}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DRAFT="${DRAFT:-FSCCS/Qwen3.5-4B-MMFlash}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
TEMPERATURE="${TEMPERATURE:-0}"
PORT="${PORT:-30000}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
BENCHES="${BENCHES:-chartqa:200 charxiv:200 mmstar:200 mmbench-origin:200 dynamath:200 mathvista:200 mathverse:200}"
TAG="$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]')"
NAME="${NAME:-${METHOD}_${TAG}_concurrency1_temp${TEMPERATURE}_4096}"

case "${MODEL,,}" in
    *gemma*) SERVER_ARGS=(--attention-backend triton --reasoning-parser gemma4)
             export SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION=1 ;;
    *)       SERVER_ARGS=(--attention-backend fa3 --reasoning-parser qwen3) ;;
esac
case "${METHOD}" in
    origin)  SPEC_ARGS=(); BENCH_BLOCK=0 ;;
    mmflash|dflash) SPEC_ARGS=(--speculative-algorithm DFLASH --speculative-draft-model-path "${DRAFT}"
                               --speculative-dflash-block-size "${BLOCK_SIZE}"); BENCH_BLOCK="${BLOCK_SIZE}" ;;
    *) echo "METHOD must be origin or mmflash" >&2; exit 1 ;;
esac

RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/${NAME}_server.log"
setsid python -m sglang.launch_server --model-path "${MODEL}" --mem-fraction-static 0.7 --tp 1 --trust-remote-code \
    --cuda-graph-max-bs 128 --mm-attention-backend sdpa --dtype bfloat16 "${SERVER_ARGS[@]}" "${SPEC_ARGS[@]}" \
    --host 127.0.0.1 --port "${PORT}" > "${LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill -- -"${SERVER_PID}" 2> /dev/null' EXIT
for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null && break
    kill -0 "${SERVER_PID}" 2> /dev/null || { echo "server exited, see ${LOG}" >&2; tail -n 30 "${LOG}" >&2; exit 1; }
    sleep 5
done

SAMPLING=(--temperature "${TEMPERATURE}" --top-p 0.95 --top-k 20)
# the client needs no GPU, and sglang.test.test_utils (imported by the benchmarks) parses the first character of
# CUDA_VISIBLE_DEVICES as an integer at import time, which fails on GPU UUIDs such as the ones PBS hands out
# shellcheck disable=SC2086
env -u CUDA_VISIBLE_DEVICES python benchmarks/bench_mm.py --model "${MODEL}" --base-url "http://127.0.0.1:${PORT}" --concurrency 1 \
    --block-size "${BENCH_BLOCK}" --benchmark-list ${BENCHES} --reasoning off "${SAMPLING[@]}" --max-tokens 4096 \
    --name "${NAME}" --output-dir "${RESULTS_DIR}"
