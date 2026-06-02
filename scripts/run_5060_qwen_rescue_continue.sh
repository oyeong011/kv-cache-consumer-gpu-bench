#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results logs
rm -f results/results_5060_qwen25_1p5b_rescue_continue.csv logs/results_5060_qwen25_1p5b_rescue_continue.log

.venv/bin/python run_kv_cache_bench.py \
  --model-id /home/ssu/models/Qwen2.5-1.5B-Instruct \
  --seq-lens 2048,4096,8192 \
  --batch-sizes 2,4,8 \
  --cache-modes dynamic,quantized \
  --max-new-tokens 64 \
  --dtype fp16 \
  --out results/results_5060_qwen25_1p5b_rescue_continue.csv \
  --warmup \
  --telemetry-interval 0.2 \
  --quantized-backend hqq \
  --continue-after-prefill-oom \
  2>&1 | tee logs/results_5060_qwen25_1p5b_rescue_continue.log
