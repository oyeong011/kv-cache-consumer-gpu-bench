# KV-cache Consumer GPU Benchmark

This repository measures **KV-cache memory pressure** for Hugging Face causal language models on consumer GPUs.
It is **not an Oaken reproduction**. Oaken's core contribution includes accelerator-level HW/SW co-design, dedicated quant/dequant hardware, and memory-management logic. This repo instead measures the root pressure that motivates Oaken-like work: KV-cache growth, latency degradation, and OOM boundaries during long-context inference.

## Two-GPU architectural bottleneck analysis

The latest documented comparison is in:

- `docs/kv_cache_two_gpu_comparison.md`
- `docs/kv_cache_architectural_bottleneck_analysis.md`

It compares a newly rerun RTX 5060 Qwen2.5-1.5B grid with the existing RTX 5080 Qwen2.5-1.5B grid, regenerates plots from CSV, and separates directly backed evidence from limitations such as missing RTX 5080 telemetry and quantized-backend mismatch.

## Key RTX 5080 result

Initial measured result on `Qwen/Qwen2.5-1.5B-Instruct` with an NVIDIA GeForce RTX 5080:

- 80 total cases, 76 successful cases, 4 OOM cases.
- OOM occurred at `batch_size=8`, `seq_len=8192` for all cache modes.
- `kv_actual_over_theory = 1.0` for every successful row, validating the KV-cache footprint formula against actual `past_key_values` tensors.
- Mean throughput on common non-OOM cases relative to `dynamic`:
  - `quantized`: 0.744×
  - `offloaded`: 0.594×
  - `no_cache`: 0.094×
- Mean peak CUDA allocated delta on common non-OOM cases relative to `dynamic`:
  - `quantized`: 0.786×
  - `offloaded`: 0.705×
  - `no_cache`: 0.739×

Representative long-context case, `batch_size=4`, `seq_len=8192`:

| cache mode | tokens/s | latency ms | peak delta bytes |
| --- | ---: | ---: | ---: |
| dynamic | 124.96 | 2048.69 | 3121613312 |
| quantized | 108.24 | 2365.16 | 2446330368 |
| offloaded | 45.45 | 5632.32 | 2215643648 |
| no_cache | 2.95 | 86730.71 | 2204519936 |

Interpretation: `dynamic` is the throughput baseline; `quantized` is the most balanced memory-saving strategy in this sweep; `offloaded` saves memory with larger throughput loss; `no_cache` is not viable as a long-context decode performance strategy. Detailed numbers are in `analysis/rtx5080_qwen25_analysis.md`.

## Research questions

1. Does actual `past_key_values` tensor footprint match the KV-cache theory?
2. Where do context length and batch size hit OOM boundaries on consumer GPUs?
3. How do `dynamic`, `quantized`, `offloaded`, and `no_cache` strategies trade memory for latency?
4. When does quantized/offloaded cache help capacity, and when does overhead reduce throughput?

## KV-cache formula

```text
KV bytes = 2 × num_layers × batch_size × seq_len × num_key_value_heads × head_dim × bytes_per_element
```

Use `num_key_value_heads`, not `num_attention_heads`, because modern GQA/MQA models store fewer KV heads than attention query heads.

## Files

```text
run_kv_cache_bench.py          # benchmark harness
analyze_results.py             # plot generator
requirements.txt               # Python requirements
PROMPT_FOR_CODEX.md            # original repo-generation prompt
scripts/run_5060_tinyllama.sh  # RTX 5060-style run
scripts/run_5080_qwen.sh       # RTX 5080-style run
scripts/push_to_github_template.sh
```

## Setup

```bash
cd /home/ssu/kv_cache_consumer_gpu_bench
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Needed only for --cache-modes quantized with backend quanto/hqq:
pip install optimum-quanto quanto hqq
```

If you only want `dynamic`, `offloaded`, and `no_cache`, the optional `optimum-quanto quanto hqq` install can be skipped. Current Transformers releases may require `optimum-quanto` for `cache_implementation="quantized"` with the quanto backend.

## RTX 5060 8GB example

```bash
bash scripts/run_5060_tinyllama.sh
```

Equivalent explicit command:

```bash
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
```

## RTX 5080 16GB example

```bash
bash scripts/run_5080_qwen.sh
```

Equivalent explicit command:

```bash
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
```

## CSV columns

The benchmark writes one row per `(cache_mode, batch_size, seq_len)` case:

- `model_id`
- `gpu_name`
- `dtype`
- `cache_mode`
- `batch_size`
- `seq_len`
- `max_new_tokens`
- `status`: `ok`, `oom`, or `error`
- `error`
- `latency_ms`
- `tokens_per_sec`
- `generated_tokens_total`
- `theoretical_kv_bytes`
- `actual_prefill_kv_bytes`
- `kv_actual_over_theory`
- `base_allocated_bytes`
- `peak_allocated_bytes`
- `peak_delta_bytes`
- `peak_reserved_bytes`
- `free_before_bytes`
- `free_after_bytes`
- `total_vram_bytes`
- `num_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`

OOM rows remain in the CSV and are excluded from plots.

## Interpretation guide

- `kv_actual_over_theory ≈ 1.0`: the formula explains the model's actual KV tensor footprint.
- `tokens_per_sec` falls as `seq_len` grows: decode becomes increasingly pressured by KV-cache reads.
- `dynamic` OOM but `quantized`/`offloaded` passes: cache compression/offload is useful at that capacity boundary.
- `quantized` uses less memory but has lower throughput: quant/dequant overhead is visible.
- `offloaded` avoids OOM but slows down: it is a capacity workaround, not a pure performance optimization.
- `no_cache` degrades sharply at long context: KV reuse is essential for autoregressive decode.

## GitHub upload

With GitHub CLI authenticated:

```bash
cd /home/ssu/kv_cache_consumer_gpu_bench
bash scripts/push_to_github_template.sh oyeong011 kv-cache-consumer-gpu-bench
```

Manual fallback:

```bash
git init
git add .
git commit -m "Measure KV-cache pressure on consumer GPUs"
git branch -M main
git remote add origin git@github.com:oyeong011/kv-cache-consumer-gpu-bench.git
git push -u origin main
```

## Report sentence

최근 Oaken과 같은 KV-cache 최적화 연구를 읽은 뒤, 해당 논문의 전용 hardware/software co-design을 consumer GPU에서 단순 재현하는 것은 핵심 성능 주장을 검증하기 어렵다고 판단했습니다. 대신 RTX 5060/5080 환경에서 Hugging Face 기반 causal LM의 KV-cache memory footprint, decode latency, OOM boundary를 직접 계측하는 실험을 구성했습니다. 이론적 KV-cache 크기와 실제 `past_key_values` tensor 크기를 비교하고, dynamic/quantized/offloaded/no-cache 전략의 memory-latency trade-off를 측정하여 긴 context inference에서 memory hierarchy와 data movement가 성능 병목으로 이어지는 지점을 분석하고 있습니다.

## Smoke validation

The benchmark pipeline was validated on an NVIDIA GeForce RTX 5080 with CUDA enabled. The first smoke test confirmed that both `dynamic` and `no_cache` generation paths run successfully, and the measured `actual_prefill_kv_bytes` matched the theoretical KV-cache size with `kv_actual_over_theory = 1.0`.

The second smoke test confirmed that `offloaded` and `quantized` cache modes also run successfully after installing `optimum-quanto`. An artificial high-pressure case with `batch_size=512` and `seq_len=8192` triggered CUDA OOM and was correctly recorded as `status=oom` in the CSV output.

## Initial RTX 5080 Qwen sweep

An initial full RTX 5080 sweep was run with `Qwen/Qwen2.5-1.5B-Instruct` over:

- `seq_len`: 512, 1024, 2048, 4096, 8192
- `batch_size`: 1, 2, 4, 8
- `cache_mode`: `dynamic`, `quantized`, `offloaded`, `no_cache`
- `max_new_tokens`: 64
- `dtype`: fp16

Summary:

- Total rows: 80
- Successful rows: 76
- OOM rows: 4
- OOM boundary observed at `batch_size=8`, `seq_len=8192` for all four cache modes.
- Mean `kv_actual_over_theory` across successful rows was `1.0` for every cache mode.

This confirms the benchmark can measure KV-cache tensor footprint, decode throughput, latency, peak CUDA allocator deltas, and capacity-boundary OOM rows on the RTX 5080 environment.

Result artifacts:

- `results/results_5080_qwen25_1p5b.csv`
- `plots_5080/throughput_vs_seq_len.png`
- `plots_5080/latency_vs_seq_len.png`
- `plots_5080/peak_delta_memory_vs_seq_len.png`
- `plots_5080/actual_kv_vs_theoretical_kv.png`
- `plots_5080/kv_actual_over_theory_vs_seq_len.png`

## RTX 5080 numeric analysis

The detailed analysis is stored in `analysis/rtx5080_qwen25_analysis.md`.

Key measured findings from the RTX 5080 Qwen2.5 sweep:

- Status: 80 total cases, 76 successful cases, 4 OOM cases.
- OOM boundary: `batch_size=8`, `seq_len=8192` for `dynamic`, `quantized`, `offloaded`, and `no_cache`.
- KV formula check: all successful rows had `kv_actual_over_theory = 1.0`.
- Throughput on common non-OOM cases relative to `dynamic`:
  - `quantized`: 0.744× mean throughput
  - `offloaded`: 0.594× mean throughput
  - `no_cache`: 0.094× mean throughput
- Peak CUDA allocated delta on common non-OOM cases relative to `dynamic`:
  - `quantized`: 0.786× mean peak delta
  - `offloaded`: 0.705× mean peak delta
  - `no_cache`: 0.739× mean peak delta

Representative long-context case, `batch_size=4`, `seq_len=8192`:

| cache mode | tokens/s | latency ms | peak delta bytes |
| --- | ---: | ---: | ---: |
| dynamic | 124.96 | 2048.69 | 3121613312 |
| quantized | 108.24 | 2365.16 | 2446330368 |
| offloaded | 45.45 | 5632.32 | 2215643648 |
| no_cache | 2.95 | 86730.71 | 2204519936 |

Interpretation: `dynamic` remains the throughput baseline, `quantized` reduces peak memory with a smaller throughput penalty than offload in this long-context example, `offloaded` reduces memory but pays a larger throughput cost, and `no_cache` is not viable as a performance strategy for long-context decode.

Caveat: the quantized run sometimes generated fewer than `max_new_tokens` because generation can stop early on EOS. `tokens_per_sec` is therefore the safer normalized throughput metric for cross-mode comparison; raw latency should be interpreted with generated token counts in mind.
