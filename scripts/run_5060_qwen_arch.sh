#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results logs plots_5060_arch
rm -f results/results_5060_qwen25_1p5b_arch.csv logs/results_5060_qwen25_1p5b_arch.log

.venv/bin/python run_kv_cache_bench.py \
  --model-id /home/ssu/models/Qwen2.5-1.5B-Instruct \
  --seq-lens 512,1024,2048,4096,8192 \
  --batch-sizes 1,2,4,8 \
  --cache-modes dynamic,quantized,no_cache \
  --max-new-tokens 64 \
  --dtype fp16 \
  --out results/results_5060_qwen25_1p5b_arch.csv \
  --warmup \
  --telemetry-interval 0.2 \
  --quantized-backend hqq \
  2>&1 | tee logs/results_5060_qwen25_1p5b_arch.log

.venv/bin/python analyze_results.py \
  --csv results/results_5060_qwen25_1p5b_arch.csv \
  --outdir plots_5060_arch
