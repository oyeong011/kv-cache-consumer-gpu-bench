#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p plots_5080 analysis/two_gpu plots_two_gpu

.venv/bin/python analyze_results.py \
  --csv results/results_5080_qwen25_1p5b.csv \
  --outdir plots_5080

.venv/bin/python scripts/analyze_two_gpu_bottleneck.py \
  --csv-5060 results/results_5060_qwen25_1p5b_arch.csv \
  --csv-5080 results/results_5080_qwen25_1p5b.csv \
  --csv-5060-rescue results/results_5060_qwen25_1p5b_rescue_continue.csv \
  --outdir analysis/two_gpu \
  --plotdir plots_two_gpu
