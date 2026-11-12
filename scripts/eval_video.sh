#!/bin/bash
# Video benchmarks as in the paper: LongVideoBench, MVBench and Video-MME at 48 frames (about 40k-token prompts),
# 200 questions each, one request in flight, greedy decoding, at most 4096 new tokens, reasoning off.
#
#   METHOD=origin  MODEL=Qwen/Qwen3.5-4B bash scripts/eval_video.sh
#   METHOD=mmflash MODEL=Qwen/Qwen3.5-4B DRAFT=<draft> bash scripts/eval_video.sh
#   METHOD=mmflash MODEL=Qwen/Qwen3.5-4B DRAFT=<draft> DRAFT_SPARSE="sink=4,text=1,stride=32,window=256" \
#       bash scripts/eval_video.sh       # sparse draft context (needs bash scripts/apply_sglang_bench_patches.sh)
#
# DRAFT_SPARSE serves the draft the sparse view of the context it was trained with (configs/train/*-video-sparse.yaml;
# python scripts/draft_sparse_env.py <yaml> prints the matching value). The first run downloads the benchmark videos
# into the Hugging Face cache (python scripts/prefetch_video_benches.py fetches them ahead of time).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
METHOD="${METHOD:-mmflash}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DRAFT="${DRAFT:-FSCCS/Qwen3.5-4B-MMFlash}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
PORT="${PORT:-30000}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
BENCHES="${BENCHES:-longvideobench:200 mvbench:200 videomme:200}"
export LVB_NUM_FRAMES=48 VIDEOMME_NUM_FRAMES=48 MVBENCH_NUM_FRAMES=48 MVBENCH_FRAME_PIXELS=1280x720 LVB_FPS_CAP=0
TAG="$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]')"
SPARSE_ARGS=()
if [ -n "${DRAFT_SPARSE:-}" ]; then
    export SGLANG_DFLASH_DRAFT_SPARSE="${DRAFT_SPARSE}"
    WINDOW=$(printf '%s' "${DRAFT_SPARSE}" | tr ',' '\n' | sed -n 's/^window=//p')
    [ -n "${WINDOW}" ] && [ "${WINDOW}" -gt 0 ] && SPARSE_ARGS=(--speculative-draft-window-size "$(( WINDOW > BLOCK_SIZE ? WINDOW : BLOCK_SIZE ))")
else
    unset SGLANG_DFLASH_DRAFT_SPARSE
fi
NAME="${NAME:-video_${METHOD}_${TAG}_concurrency1_temp0_4096${DRAFT_SPARSE:+_sparse}}"
case "${METHOD}" in
    origin)  SPEC_ARGS=(); BENCH_BLOCK=0 ;;
    mmflash|dflash) SPEC_ARGS=(--speculative-algorithm DFLASH --speculative-draft-model-path "${DRAFT}"
                               --speculative-dflash-block-size "${BLOCK_SIZE}" "${SPARSE_ARGS[@]}"); BENCH_BLOCK="${BLOCK_SIZE}" ;;
    *) echo "METHOD must be origin or mmflash" >&2; exit 1 ;;
esac

RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/${NAME}_server.log"
setsid python -m sglang.launch_server --model-path "${MODEL}" --mem-fraction-static 0.7 --tp 1 --trust-remote-code \
    --cuda-graph-max-bs 128 --attention-backend fa3 --mm-attention-backend sdpa --dtype bfloat16 \
    --reasoning-parser qwen3 "${SPEC_ARGS[@]}" --host 127.0.0.1 --port "${PORT}" > "${LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill -- -"${SERVER_PID}" 2> /dev/null' EXIT
for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null && break
    kill -0 "${SERVER_PID}" 2> /dev/null || { echo "server exited, see ${LOG}" >&2; tail -n 30 "${LOG}" >&2; exit 1; }
    sleep 5
done
# the client needs no GPU, and sglang.test.test_utils (imported by the benchmarks) parses the first character of
# CUDA_VISIBLE_DEVICES as an integer at import time, which fails on GPU UUIDs such as the ones PBS hands out
# shellcheck disable=SC2086
env -u CUDA_VISIBLE_DEVICES python benchmarks/bench_mm.py --model "${MODEL}" --base-url "http://127.0.0.1:${PORT}" --concurrency 1 \
    --block-size "${BENCH_BLOCK}" --benchmark-list ${BENCHES} --reasoning off --temperature 0 --top-p 0.95 --top-k 20 \
    --max-tokens 4096 --name "${NAME}" --output-dir "${RESULTS_DIR}"
