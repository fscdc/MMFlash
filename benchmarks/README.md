# Benchmarks

Clients that measure speculative decoding against a running OpenAI-compatible server. They never launch a server:
`scripts/eval_image.sh`, `scripts/eval_video.sh` and `scripts/eval_vllm_concurrency.sh` start one and call them.

| Script | What it runs |
|---|---|
| `bench_mm.py` | Image and video benchmarks (`mm_benchmarker/`): ChartQA, CharXiv, MMStar, MMBench, DynaMath, MathVista, MathVerse, LongVideoBench, MVBench, Video-MME, and more |
| `bench_text.py` | Text benchmarks (`benchmarker/`): GSM8K, MATH-500, HumanEval, MBPP, MT-Bench, and more |
| `vllm/bench_mm_vllm.py` | `bench_mm.py` against a vLLM server; `vllm/phase_plugin` adds the server-side prefill/decode accounting |

Both clients send one non-streaming `POST /v1/chat/completions` per question, with images as base64 `data:` URIs,
and write `<output-dir>/<name>_results.jsonl`. For every benchmark the file holds the acceptance length τ (the mean
over requests of the server's `spec_accept_length`), the decode throughput, the accuracy and per-question details.
A rerun with the same `--name` skips the benchmarks the file already holds.

```bash
# a server is already running on port 30000 (see the main README)
python benchmarks/bench_mm.py --model Qwen/Qwen3.5-4B --base-url http://127.0.0.1:30000 \
    --benchmark-list chartqa:200 charxiv:200 mmstar:200 mmbench-origin:200 dynamath:200 mathvista:200 mathverse:200 \
    --concurrency 1 --block-size 16 --reasoning off --temperature 0 --top-p 0.95 --top-k 20 --max-tokens 4096 \
    --name mmflash_qwen3.5-4b
```

`<benchmark>:<n>` evaluates the first `n` questions of the benchmark. `mmbench-origin` keeps MMBench's own
answer instruction; the other image benchmarks use a shared step-by-step `\boxed{}` instruction. The decode
throughput uses the per-request decode latency that `scripts/apply_sglang_bench_patches.sh` adds to SGLang; on a
stock server it falls back to the end-to-end latency.
