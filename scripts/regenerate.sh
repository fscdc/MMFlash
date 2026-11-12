#!/bin/bash
# Step 1: regenerate the training responses with the target model.
#
# A draft learns to imitate its target, so every assistant turn of the training set is rewritten by the target
# itself (greedy, at most 4096 new tokens, thinking off). What this does:
#   1. one SGLang server per GPU, each under a supervisor that relaunches it if it dies;
#   2. scripts/regenerate_train_data.py --resume, repeated until no row fails (a round only retries the rows that
#      failed: rows already written, skipped or rejected are recognized by their id);
#   3. a tally that must add up to the input.
# Re-running the script on an interrupted output finishes it.
#
#   MODEL=Qwen/Qwen3.5-4B INPUT=data/train.jsonl OUTPUT=data/regen/qwen3.5-4b.jsonl GPUS="0 1 2 3" \
#       bash scripts/regenerate.sh
#
# INPUT is a conversation JSONL (scripts/prepare_data_mm.py converts several public multimodal datasets).
# ALIGN_PROMPTS=1 (default) first rewrites math and terse-answer prompts to the step-by-step "\boxed{}" instruction
# the benchmarks use; the released drafts were trained on data regenerated this way. ALIGN_PROMPTS=0 keeps the
# prompts as they are. EXTRA_SERVER_ARGS is appended to every server command (e.g. "--tp 2" for the 27B target on
# 80 GB GPUs, or "--disable-radix-cache").
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
INPUT="${INPUT:?set INPUT to the conversation JSONL to regenerate}"
OUTPUT="${OUTPUT:?set OUTPUT to the regenerated JSONL to write}"
read -r -a GPU_IDS <<< "${GPUS:-0}"
CONCURRENCY="${CONCURRENCY:-64}"          # in-flight requests per server
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"             # --resume rounds; later rounds only retry failed rows
SERVER_TIMEOUT="${SERVER_TIMEOUT:-900}"   # seconds to wait for a server to come up
PORT_BASE="${PORT_BASE:-40000}"
ALIGN_PROMPTS="${ALIGN_PROMPTS:-1}"
LOG_DIR="${LOG_DIR:-logs/regen-$(date +%Y%m%d-%H%M%S)}"
# one compile cache per GPU: servers that JIT-compile the same kernels at the same moment into one shared cache
# directory can race and crash
CACHE_ROOT="${SPECFORGE_CACHE_ROOT:-${HOME}/.cache/mmflash}"

case "${MODEL,,}" in
    *gemma*) SERVER_ARGS=(--attention-backend triton --reasoning-parser gemma4) ;;
    *)       SERVER_ARGS=(--attention-backend fa3 --reasoning-parser qwen3) ;;
esac
read -r -a EXTRA <<< "${EXTRA_SERVER_ARGS:-}"
REGEN_ARGS=(--reasoning disable)
[ "${ALIGN_PROMPTS}" = 1 ] && REGEN_ARGS+=(--align-prompts)

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT}")"
ERROR_FILE="${OUTPUT%.jsonl}_error.jsonl"
SKIPPED_FILE="${OUTPUT%.jsonl}_skipped.jsonl"
REJECTED_FILE="${OUTPUT%.jsonl}_rejected.jsonl"
STOP_FILE="${LOG_DIR}/stop-servers"
rm -f "${STOP_FILE}"
SERVER_ADDRESSES=()
SUPERVISOR_PIDS=()

count_lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }

cleanup() {
    echo "[cleanup] stopping the SGLang servers"
    touch "${STOP_FILE}"
    for pid_file in "${LOG_DIR}"/server_gpu*.pid; do
        [ -f "${pid_file}" ] && kill "$(cat "${pid_file}")" 2> /dev/null
    done
    for pid in "${SUPERVISOR_PIDS[@]}"; do wait "${pid}" 2> /dev/null; done
}
trap cleanup EXIT INT TERM

# launch one server and relaunch it whenever it exits before we asked it to; rows that fail while it is down are
# retried by the next --resume round
supervise_server() {
    local gpu_id=$1 port=$2 attempt=0 pid
    local cache_dir="${CACHE_ROOT}/triton-$(hostname -s)-gpu${gpu_id}"
    local log="${LOG_DIR}/server_gpu${gpu_id}_port${port}.log"
    local pid_file="${LOG_DIR}/server_gpu${gpu_id}.pid"
    mkdir -p "${cache_dir}" "${cache_dir}-inductor"
    while [ ! -f "${STOP_FILE}" ]; do
        attempt=$((attempt + 1))
        echo "[server] GPU ${gpu_id} port ${port}: launch #${attempt} ($(date '+%F %T'))"
        TRITON_CACHE_DIR="${cache_dir}" TORCHINDUCTOR_CACHE_DIR="${cache_dir}-inductor" CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python -m sglang.launch_server --model-path "${MODEL}" --mem-fraction-static 0.7 --tp 1 --trust-remote-code \
            --cuda-graph-max-bs 128 --mm-attention-backend sdpa --dtype bfloat16 --max-running-requests "${CONCURRENCY}" \
            "${SERVER_ARGS[@]}" "${EXTRA[@]}" --host 127.0.0.1 --port "${port}" >> "${log}" 2>&1 &
        pid=$!
        echo "${pid}" > "${pid_file}"
        wait "${pid}"
        rm -f "${pid_file}"
        [ -f "${STOP_FILE}" ] && break
        echo "[server] GPU ${gpu_id}: exited ($(date '+%F %T')); relaunching in 15 s, see ${log}" >&2
        sleep 15
    done
}

wait_for_servers() {
    local addr start
    for addr in "${SERVER_ADDRESSES[@]}"; do
        start=$(date +%s)
        until curl -sf -m 5 "http://${addr}/health" > /dev/null 2>&1; do
            if (( $(date +%s) - start >= SERVER_TIMEOUT )); then
                echo "[error] server ${addr} did not come up; see ${LOG_DIR}" >&2
                return 1
            fi
            sleep 10
        done
    done
}

echo "[info] ${MODEL} on GPU(s) ${GPU_IDS[*]}, ${CONCURRENCY} requests in flight per server; logs in ${LOG_DIR}"
echo "[info] ${INPUT} -> ${OUTPUT} (already written: $(count_lines "${OUTPUT}"), failed: $(count_lines "${ERROR_FILE}"))"
for idx in "${!GPU_IDS[@]}"; do
    port=$((PORT_BASE + idx * 10))
    SERVER_ADDRESSES+=("127.0.0.1:${port}")
    supervise_server "${GPU_IDS[$idx]}" "${port}" &
    SUPERVISOR_PIDS+=("$!")
done
wait_for_servers || exit 1

prev_failed=-1
for round in $(seq 1 "${MAX_ROUNDS}"); do
    wait_for_servers || exit 1   # a server may be mid-restart
    echo "[round ${round}/${MAX_ROUNDS}] $(date '+%F %T'), log ${LOG_DIR}/regen.log"
    python scripts/regenerate_train_data.py --model "${MODEL}" --concurrency "${CONCURRENCY}" \
        --max-tokens "${MAX_TOKENS}" --server-address "${SERVER_ADDRESSES[@]}" --temperature 0.0 --top-p 0.95 \
        --top-k 20 --input-file-path "${INPUT}" --output-file-path "${OUTPUT}" --resume "${REGEN_ARGS[@]}" \
        >> "${LOG_DIR}/regen.log" 2>&1 || { echo "[error] regeneration failed, see ${LOG_DIR}/regen.log" >&2; exit 1; }
    failed=$(count_lines "${ERROR_FILE}")
    echo "[round ${round}] $(count_lines "${OUTPUT}") regenerated, ${failed} failed"
    [ "${failed}" -eq 0 ] && break
    if [ "${prev_failed}" -ge 0 ] && [ "${failed}" -ge "${prev_failed}" ]; then
        echo "[error] no progress since the last round; the servers are probably unhealthy, see ${LOG_DIR}" >&2
        exit 1
    fi
    prev_failed=${failed}
done

total=$(count_lines "${INPUT}")
success=$(count_lines "${OUTPUT}") failed=$(count_lines "${ERROR_FILE}")
skipped=$(count_lines "${SKIPPED_FILE}") rejected=$(count_lines "${REJECTED_FILE}")
echo "[done] input ${total} = regenerated ${success} + failed ${failed} + skipped ${skipped} + rejected ${rejected}"
[ "${failed}" -eq 0 ] || { echo "[done] ${failed} rows still failed; re-run this script to retry them" >&2; exit 1; }
python scripts/validate_regenerated_data.py --data-path "${OUTPUT}" --expect-non-reasoning > "${LOG_DIR}/validate.log" 2>&1 \
    && echo "[done] ${OUTPUT} passed scripts/validate_regenerated_data.py" \
    || echo "[warn] scripts/validate_regenerated_data.py flagged rows, see ${LOG_DIR}/validate.log" >&2
