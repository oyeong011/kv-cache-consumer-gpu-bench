#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results plots_5060
python run_kv_cache_bench.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --seq-lens 128,256,512,1024,2048,3072 \
  --batch-sizes 1,2,4 \
  --cache-modes dynamic,quantized,offloaded,no_cache \
  --max-new-tokens 32 \
  --dtype fp16 \
  --out results/results_5060_tinyllama.csv \
  --warmup
python analyze_results.py --csv results/results_5060_tinyllama.csv --outdir plots_5060
