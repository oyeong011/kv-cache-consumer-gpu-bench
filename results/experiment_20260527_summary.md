# RTX 5080 KV-cache experiment summary - 2026-05-27

## Scope

- Harness: `/home/ssu/kv_cache_consumer_gpu_bench/run_kv_cache_bench.py`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- GPU: `NVIDIA GeForce RTX 5080`
- dtype: `fp16`
- cache modes: `dynamic`, `quantized`, `offloaded`, `no_cache`
- batch sizes: `1,2,4,8`
- sequence lengths: `512,1024,2048,4096,8192`
- max generated tokens: `64`

Raw output:

- CSV: `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b_20260527.csv`
- Plots: `/home/ssu/kv_cache_consumer_gpu_bench/plots_5080_20260527/`

## Environment caveat

This run was executed while another GPU process was resident:

- `.../linux-x86_64/release/kit/kit`: about `2280 MiB`
- Xorg / GNOME shell: about `138 MiB`

Because of that, the observed OOM boundary is stricter than the earlier clean RTX 5080 sweep stored in `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`.
Use this run as evidence that runtime GPU occupancy and allocator state can shift capacity boundaries, not as a replacement for the earlier clean-room result.

## Status summary

- Total cases: `80`
- OK: `68`
- OOM: `12`
- Error: `0`
- KV formula check: every successful row had `kv_actual_over_theory = 1.0`

OOM rows appeared at the same grid positions for all four cache modes:

- `batch_size=4`, `seq_len=8192`
- `batch_size=8`, `seq_len=4096`
- `batch_size=8`, `seq_len=8192`

## Throughput ratios on common successful cases

Mean throughput relative to `dynamic` across common non-OOM rows:

| cache mode | relative throughput |
| --- | ---: |
| dynamic | 1.000x |
| quantized | 0.740x |
| offloaded | 0.623x |
| no_cache | 0.102x |

Interpretation:

- `dynamic` remains the fastest successful serving baseline.
- `quantized` reduces peak CUDA allocation but pays quant/dequant overhead.
- `offloaded` reduces GPU allocation more aggressively but loses more throughput.
- `no_cache` is an ablation/lower-bound, not a practical long-context serving strategy.

## Representative long-context cases

`batch_size=4`, `seq_len=4096`:

| cache mode | tokens/s | latency ms | peak delta GiB |
| --- | ---: | ---: | ---: |
| dynamic | 190.01 | 1347.28 | 1.454 |
| quantized | 164.50 | 1556.21 | 1.139 |
| offloaded | 85.89 | 2980.39 | 1.032 |
| no_cache | 5.94 | 43087.04 | 1.037 |

`batch_size=2`, `seq_len=8192`:

| cache mode | tokens/s | latency ms | peak delta GiB |
| --- | ---: | ---: | ---: |
| dynamic | 92.50 | 1383.86 | 1.454 |
| quantized | 80.60 | 1588.19 | 1.139 |
| offloaded | 39.60 | 3231.99 | 1.032 |
| no_cache | 2.72 | 47095.01 | 1.028 |

## RTX 5060 comparison notes

The closest local RTX 5060 Qwen evidence is stored under `/home/ssu/oaken/results/`:

- `/home/ssu/oaken/results/rtx5060_qwen25_15b_dynamic_boundary.csv`
- `/home/ssu/oaken/results/rtx5060_qwen25_15b_rescue_cases.csv`
- `/home/ssu/oaken/results/rtx5060_qwen25_15b_sanity.csv`

Those 5060 runs used a chunked cache-growth harness, so compare the direction of the boundary/rescue behavior rather than treating throughput numbers as directly apples-to-apples with this Hugging Face `generate()` sweep.

Observed local 5060 Qwen facts from those files:

- Dynamic cache passed through `batch_size=8`, `seq_len=8192`.
- Dynamic cache OOM occurred at `batch_size=8`, `seq_len=12288` and `16384`.
- Quantized cache rescued both `batch_size=8`, `seq_len=12288` and `16384`.
- Qwen config was validated as GQA/MQA: `num_key_value_heads=2`, `head_dim=128`, `max_position_embeddings=32768`.

The useful system question is therefore not only "which GPU is faster?" but:

- when is the bottleneck pure VRAM capacity,
- when does quant/dequant overhead dominate,
- when does offload trade GPU memory pressure for transfer/host-memory pressure,
- and how much do runtime occupancy plus allocator fragmentation move the apparent OOM boundary?

