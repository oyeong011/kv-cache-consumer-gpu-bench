# KV-cache two-GPU comparison

Status: combined from CSV artifacts on 2026-06-02 after rebasing onto the existing remote two-machine evidence.

This repository now contains two complementary evidence tracks. They are intentionally kept separate so RTX 5060-backed and RTX 5080-backed claims are not merged beyond what the files support.

## Evidence tracks

### Track 1: autoregressive generate same-grid comparison

- Purpose: compare RTX 5060 and RTX 5080 on the same Hugging Face `generate()` grid where possible.
- Model: Qwen2.5-1.5B-Instruct (`/home/ssu/models/Qwen2.5-1.5B-Instruct` on RTX 5060, `Qwen/Qwen2.5-1.5B-Instruct` in the existing RTX 5080 CSV).
- Grid: batch sizes `1,2,4,8`; sequence lengths `512,1024,2048,4096,8192`; `max_new_tokens=64`; dtype `fp16`.
- Cache modes in comparison: `dynamic`, `quantized`, `no_cache`.
- RTX 5060 status: **newly rerun** in `results/results_5060_qwen25_1p5b_arch.csv`.
- RTX 5080 status: **verified from existing file** `results/results_5080_qwen25_1p5b.csv` with mtime `2026-05-20T20:01:19`.
- Derived CSVs: `analysis/two_gpu/*.csv`.
- Plots: `plots_two_gpu/*.png` and per-GPU plots in `plots_5060_arch/` and `plots_5080/`.

### Track 2: chunked cache-growth / long-boundary comparison

- Purpose: compare longer cache-growth boundaries and newly rerun RTX 5080 telemetry against existing RTX 5060 chunked evidence.
- Harness: `scripts/run_chunked_kv_cache_sweep_telemetry.py` and `scripts/build_two_gpu_arch_analysis.py`.
- Grid coverage: dynamic rows over batch sizes `1,2,4,8` and sequence lengths `1024,2048,4096,8192,12288,16384` where files exist; quantized/no_cache only where matched rows exist.
- RTX 5060 status: **verified from existing file**.
- RTX 5080 status: **newly rerun** in `results/two_gpu_arch/`.
- Derived CSVs: `results/two_gpu_arch/two_gpu_qwen25_matched_grid.csv`, `results/two_gpu_arch/two_gpu_qwen25_derived_metrics.csv`, and `results/two_gpu_arch/two_gpu_qwen25_status_summary.csv`.
- Plots: `plots_two_gpu_arch/*.png`.

## Hardware/software environment

| Machine | Track | Evidence status | Hostname | GPU | VRAM evidence | Driver/CUDA/PyTorch evidence |
|---|---|---|---|---|---|---|
| RTX 5060 | generate | newly rerun | `ssu-22663-09` | NVIDIA GeForce RTX 5060 | `8151 MiB` from `logs/env_5060_qwen25_arch.log`; CSV total `8073838592` bytes | driver `590.48.01`, NVIDIA-SMI CUDA `13.1`, PyTorch `2.12.0+cu130`, torch CUDA `13.0`, Transformers `5.8.1`, HQQ `0.2.8.post1` |
| RTX 5080 | generate | verified existing file | not captured | NVIDIA GeForce RTX 5080 | CSV total `16600596480` bytes | not recorded in `results/results_5080_qwen25_1p5b.csv` |
| RTX 5060 | chunked | verified existing file | see remote inventory docs | RTX 5060 class | in `results/rtx5060_*` files | utilization/power telemetry missing in those existing CSVs |
| RTX 5080 | chunked | newly rerun | see `logs/rtx5080_*` files | RTX 5080 class | in `results/two_gpu_arch/*.csv` and logs | coarse telemetry newly captured where available |

## Collected metrics

- Track 1 has CSV-backed `latency_ms`, `tokens_per_sec`, `peak_allocated_bytes`, `peak_delta_bytes`, and `status`/OOM fields for both GPUs. The newly rerun RTX 5060 files also include best-effort `nvidia-smi` columns: `gpu_util_mean_pct`, `gpu_util_max_pct`, `memory_util_mean_pct`, `memory_util_max_pct`, `power_draw_mean_w`, and `power_draw_max_w`.
- Track 2 includes coarse RTX 5080 telemetry where `nvidia-smi` exposed it. The existing RTX 5060 chunked rows do not have utilization/power telemetry.
- No file contains Nsight-grade DRAM GB/s bandwidth utilization. Treat `nvidia-smi` memory utilization as a coarse counter, not a direct bandwidth measurement.

## Commands

### Track 1 RTX 5060 generate full grid

```bash
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
  --quantized-backend hqq
```

### Track 1 RTX 5060 prefill-OOM continuation check

```bash
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
  --continue-after-prefill-oom
```

### Track 1 RTX 5080 existing generate grid

The existing CSV is backed by `scripts/run_5080_qwen.sh`:

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
```

### Track 2 RTX 5080 chunked reruns

Representative commands retained from the remote evidence:

```bash
.venv/bin/python scripts/run_chunked_kv_cache_sweep_telemetry.py \
  --model /home/ssu/models/Qwen2.5-1.5B-Instruct \
  --gpu-name rtx5080-16gb --dtype fp16 \
  --batch-sizes 1 2 4 8 \
  --seq-lens 1024 2048 4096 8192 12288 16384 \
  --cache-modes dynamic \
  --output results/two_gpu_arch/rtx5080_qwen25_15b_dynamic_grid_telemetry_venv.csv \
  --chunk-size 128 --vram-sample-interval 0.1

.venv/bin/python scripts/run_chunked_kv_cache_sweep_telemetry.py \
  --model /home/ssu/models/Qwen2.5-1.5B-Instruct \
  --gpu-name rtx5080-16gb --dtype fp16 \
  --pairs 1:1024 1:2048 8:12288 8:16384 \
  --cache-modes quantized \
  --output results/two_gpu_arch/rtx5080_qwen25_15b_quantized_matched_quanto_telemetry_venv.csv \
  --chunk-size 128 --quant-backend quanto --quant-bits 4 --vram-sample-interval 0.1

.venv/bin/python scripts/build_two_gpu_arch_analysis.py
```

### Plot/table regeneration for Track 1

```bash
.venv/bin/python analyze_results.py --csv results/results_5060_qwen25_1p5b_arch.csv --outdir plots_5060_arch
.venv/bin/python analyze_results.py --csv results/results_5080_qwen25_1p5b.csv --outdir plots_5080
.venv/bin/python scripts/analyze_two_gpu_bottleneck.py \
  --csv-5060 results/results_5060_qwen25_1p5b_arch.csv \
  --csv-5080 results/results_5080_qwen25_1p5b.csv \
  --csv-5060-rescue results/results_5060_qwen25_1p5b_rescue_continue.csv \
  --outdir analysis/two_gpu \
  --plotdir plots_two_gpu
```

## Artifact inventory

| Path | Status | Purpose |
|---|---|---|
| `results/results_5060_qwen25_1p5b_arch.csv` | newly rerun | Track 1 RTX 5060 60-row same-grid sweep |
| `results/results_5060_qwen25_1p5b_rescue_continue.csv` | newly rerun | Track 1 RTX 5060 targeted prefill-OOM continuation check |
| `results/results_5080_qwen25_1p5b.csv` | verified existing file | Track 1 RTX 5080 80-row sweep; comparison uses 60 dynamic/quantized/no_cache rows |
| `analysis/two_gpu/*.csv` | regenerated from CSV | Track 1 aligned rows, speedups, ratios, OOM tables |
| `plots_two_gpu/*.png` | regenerated from CSV | Track 1 comparison and rescue plots |
| `results/two_gpu_arch/*.csv` | existing remote/newly rerun mix | Track 2 matched-grid and RTX 5080 telemetry outputs |
| `plots_two_gpu_arch/*.png` | regenerated in remote evidence | Track 2 architectural plots |
| `docs/kv_cache_experiment_inventory.md` | verified existing remote doc | broader artifact inventory from the previous two-machine pass |
| `docs/kv_cache_two_machine_results.md` | verified existing remote doc | separated RTX 5060/RTX 5080 evidence from the previous two-machine pass |

## Status summary

### Track 1 generate grid

- RTX 5060 full grid: 60 rows; `dynamic`, `quantized`, and `no_cache` each have 14 `ok` and 6 `oom` rows in the default prefill-measured harness.
- RTX 5080 comparison subset: 60 rows; each of `dynamic`, `quantized`, and `no_cache` has 19 `ok` and 1 `oom` row.
- Same-grid status table: `analysis/two_gpu/same_grid_status.csv`.
- OOM map: `plots_two_gpu/oom_boundary_map_both_gpus.png`.

Default-harness maximum successful sequence length by batch:

| GPU | Cache modes | b=1 | b=2 | b=4 | b=8 |
|---|---|---:|---:|---:|---:|
| RTX 5060 | dynamic / quantized / no_cache | 8192 | 4096 | 2048 | 1024 |
| RTX 5080 | dynamic / quantized / no_cache | 8192 | 8192 | 8192 | 4096 |

The Track 1 default OOM rows are conservative because `run_kv_cache_bench.py` measures unquantized prefill KV bytes before `generate()`. The RTX 5060 continuation check shows both dynamic and HQQ-quantized generation succeeded through `batch=8, seq_len=4096` and both OOMed at `batch=8, seq_len=8192`; see `analysis/two_gpu/rtx5060_rescue_boundary_table.csv` and `plots_two_gpu/rtx5060_rescue_boundary_map.png`.

### Track 2 chunked long-boundary grid

The chunked matched-grid summary is in `results/two_gpu_arch/two_gpu_qwen25_status_summary.csv`. It includes dynamic, quantized, and no-cache rows where matched files exist. This track reports an RTX 5060 capacity-rescue case at longer sequence lengths: dynamic OOM at `batch=8, seq_len=12288/16384` while quantized rows complete in the existing RTX 5060 evidence. That claim is backed by `results/two_gpu_arch/two_gpu_qwen25_matched_grid.csv`, not by the Track 1 generate CSV.

## Derived ratios

### Track 1 generate ratios

- Expected bandwidth ratio: `960 / 448 = 2.142857`.
- Expected CUDA-core ratio: `10752 / 3840 = 2.8`.
- Dynamic median RTX 5080/5060 speedup: `1.68x`.
- No-cache median RTX 5080/5060 speedup: `2.62x`.
- Quantized median RTX 5080/5060 speedup: `1.34x`, but backend-mismatched (RTX 5060 HQQ vs RTX 5080 inferred Quanto).
- RTX 5060 HQQ median `memory_saving_ratio`: `0.784`; median `throughput_loss_ratio`: `0.839`.
- RTX 5080 inferred-Quanto median `memory_saving_ratio`: `0.784`; median `throughput_loss_ratio`: `0.733`.

### Track 2 chunked ratios

- Derived metrics: `results/two_gpu_arch/two_gpu_qwen25_derived_metrics.csv`.
- Median dynamic RTX 5080/5060 speedup over common successful chunked rows: `2.053x`.
- Median quantized RTX 5080/5060 speedup over common successful chunked rows: `7.561x`; this is backend- and harness-sensitive, so do not treat it as clean hardware scaling.

## Plot files

Track 1:

- `plots_two_gpu/throughput_vs_seq_len_5060_vs_5080.png`
- `plots_two_gpu/peak_memory_vs_seq_len_dynamic_vs_quantized.png`
- `plots_two_gpu/speedup_5080_over_5060_vs_seq_len.png`
- `plots_two_gpu/quantized_memory_saving_vs_throughput_loss.png`
- `plots_two_gpu/oom_boundary_map_both_gpus.png`
- `plots_two_gpu/rtx5060_rescue_boundary_map.png`

Track 2:

- `plots_two_gpu_arch/throughput_vs_seq_len_dynamic_5060_vs_5080.png`
- `plots_two_gpu_arch/throughput_vs_seq_len_quantized_5060_vs_5080.png`
- `plots_two_gpu_arch/peak_memory_vs_seq_len_dynamic_vs_quantized.png`
- `plots_two_gpu_arch/speedup_5080_over_5060_vs_seq_len.png`
- `plots_two_gpu_arch/quantized_memory_saving_vs_throughput_loss.png`
- `plots_two_gpu_arch/oom_boundary_map_5060_5080.png`

## Limitations

- Track 1 has RTX 5060 telemetry but not RTX 5080 telemetry.
- Track 2 has RTX 5080 coarse telemetry but not RTX 5060 coarse telemetry.
- Quantized cross-GPU comparisons are not backend-identical across all files: RTX 5060 rows commonly use HQQ, while RTX 5080 successful quantized reruns use/infer Quanto.
- The generate harness and chunked cache-growth harness answer related but not identical questions; claims above specify which track backs them.
- No Nsight Compute/PyTorch-profiler kernel metrics are committed, so bottleneck attribution remains latency/throughput/memory based.
