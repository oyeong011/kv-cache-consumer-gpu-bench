#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results plots_5080
rm -f results/results_5080_qwen25_1p5b.csv
python run_kv_cache_bench.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --seq-lens 512,1024,2048,4096,8192 \
  --batch-sizes 1,2,4,8 \
  --cache-modes dynamic,quantized,offloaded,no_cache \
  --max-new-tokens 64 \
  --dtype fp16 \
  --out results/results_5080_qwen25_1p5b.csv \
  --warmup
python analyze_results.py --csv results/results_5080_qwen25_1p5b.csv --outdir plots_5080
