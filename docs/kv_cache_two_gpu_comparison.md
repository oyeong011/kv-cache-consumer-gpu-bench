# KV-cache two-GPU comparison

Status: generated from CSV artifacts on 2026-06-02.

## Compared grid

- Model: `/home/ssu/models/Qwen2.5-1.5B-Instruct` / Qwen2.5-1.5B-Instruct.
- Harness: chunked cache-growth harness in `scripts/run_chunked_kv_cache_sweep_telemetry.py` for new RTX 5080 runs; older RTX 5060 runs used the predecessor chunked harness.
- RTX 5060 evidence is `verified from existing file`; RTX 5080 matched runs are `newly rerun`.
- Dynamic cache was compared over batch sizes 1,2,4,8 and sequence lengths 1024,2048,4096,8192,12288,16384 where rows exist.
- Quantized cache was compared only on pairs that exist on both GPUs: batch=1 seq=1024/2048 and batch=8 seq=12288/16384.
- no_cache was compared only on batch=8 seq=12288/16384 because those are the existing RTX 5060 no_cache rows.

## Artifact outputs

- Normalized CSV: `results/two_gpu_arch/two_gpu_qwen25_matched_grid.csv`
- Derived metrics CSV: `results/two_gpu_arch/two_gpu_qwen25_derived_metrics.csv`
- Status summary CSV: `results/two_gpu_arch/two_gpu_qwen25_status_summary.csv`
- Plots: `plots_two_gpu_arch/`

## New RTX 5080 commands used

```bash
.venv/bin/python scripts/run_chunked_kv_cache_sweep_telemetry.py   --model /home/ssu/models/Qwen2.5-1.5B-Instruct   --gpu-name rtx5080-16gb --dtype fp16   --batch-sizes 1 2 4 8   --seq-lens 1024 2048 4096 8192 12288 16384   --cache-modes dynamic   --output results/two_gpu_arch/rtx5080_qwen25_15b_dynamic_grid_telemetry_venv.csv   --chunk-size 128 --vram-sample-interval 0.1

.venv/bin/python scripts/run_chunked_kv_cache_sweep_telemetry.py   --model /home/ssu/models/Qwen2.5-1.5B-Instruct   --gpu-name rtx5080-16gb --dtype fp16   --batch-sizes 1 8 --seq-lens 1024 2048 12288 16384   --pairs 1:1024 1:2048 8:12288 8:16384   --cache-modes quantized   --output results/two_gpu_arch/rtx5080_qwen25_15b_quantized_matched_quanto_telemetry_venv.csv   --chunk-size 128 --quant-backend quanto --quant-bits 4 --vram-sample-interval 0.1

.venv/bin/python scripts/run_chunked_kv_cache_sweep_telemetry.py   --model /home/ssu/models/Qwen2.5-1.5B-Instruct   --gpu-name rtx5080-16gb --dtype fp16   --batch-sizes 8 --seq-lens 12288 16384   --pairs 8:12288 8:16384   --cache-modes no_cache   --output results/two_gpu_arch/rtx5080_qwen25_15b_no_cache_matched_telemetry_venv.csv   --chunk-size 128 --vram-sample-interval 0.1

.venv/bin/python scripts/build_two_gpu_arch_analysis.py
```

The failed system-Python quantized attempts are retained as explicit `ERROR` rows in `results/two_gpu_arch/two_gpu_qwen25_matched_grid.csv`; the successful quantized run used the repository venv with Quanto installed.

## Plot files

- `plots_two_gpu_arch/throughput_vs_seq_len_dynamic_5060_vs_5080.png`
- `plots_two_gpu_arch/throughput_vs_seq_len_quantized_5060_vs_5080.png`
- `plots_two_gpu_arch/peak_memory_vs_seq_len_dynamic_vs_quantized.png`
- `plots_two_gpu_arch/speedup_5080_over_5060_vs_seq_len.png`
- `plots_two_gpu_arch/quantized_memory_saving_vs_throughput_loss.png`
- `plots_two_gpu_arch/oom_boundary_map_5060_5080.png`

## Status summary

| gpu      | cache_mode   | status_norm   |   rows |
|:---------|:-------------|:--------------|-------:|
| RTX 5060 | dynamic      | OK            |     22 |
| RTX 5060 | dynamic      | OOM           |      2 |
| RTX 5060 | no_cache     | OK            |      2 |
| RTX 5060 | quantized    | OK            |      4 |
| RTX 5080 | dynamic      | OK            |     24 |
| RTX 5080 | no_cache     | OK            |      2 |
| RTX 5080 | quantized    | ERROR         |      8 |
| RTX 5080 | quantized    | OK            |      4 |

## Derived ratios

- `bandwidth_ratio_expected = 960 / 448 = 2.143`
- `cuda_core_ratio_expected = 10752 / 3840 = 2.800`
- Median dynamic RTX 5080/5060 speedup over common successful rows: 2.053x
- Median quantized RTX 5080/5060 speedup over common successful rows: 7.561x

## Missing telemetry

RTX 5060 utilization, memory-bandwidth-utilization, and power telemetry are missing because the existing RTX 5060 CSVs did not record `nvidia-smi` utilization/power columns. RTX 5080 telemetry columns were newly collected where `nvidia-smi` exposed them, but they are coarse process-level samples rather than Nsight-grade kernel metrics.
