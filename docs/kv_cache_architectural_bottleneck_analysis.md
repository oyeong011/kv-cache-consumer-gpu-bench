# KV-cache architectural bottleneck analysis

## Question

When quantized KV-cache is used, is the RTX 5060 vs RTX 5080 performance gap mainly explained by compute capacity, memory bandwidth, VRAM capacity, or quant/dequant overhead?

## Short answer

The strongest directly backed conclusion is **capacity sensitivity plus memory-side/overhead effects**, not pure CUDA-core scaling.

- In the same-grid `generate()` track, RTX 5060 hits OOM earlier than RTX 5080; dynamic speedup is `~1.68x` median, below both the expected memory-bandwidth ratio `2.14x` and CUDA-core ratio `2.8x`.
- In the chunked long-boundary track, dynamic speedup is `~2.05x` median, closer to the expected bandwidth ratio than the core ratio, and long-context RTX 5060 rows show a quantized capacity-rescue case.
- Quantized cache consistently saves measured peak memory, but it also reduces throughput and is backend-sensitive. HQQ/Quanto backend differences prevent a clean quantized 5080/5060 hardware-scaling claim.

## Evidence boundaries

Two evidence tracks are committed and should not be conflated:

1. **Track 1: autoregressive `generate()` same-grid comparison**
   - CSVs: `results/results_5060_qwen25_1p5b_arch.csv`, `results/results_5060_qwen25_1p5b_rescue_continue.csv`, `results/results_5080_qwen25_1p5b.csv`.
   - Derived outputs: `analysis/two_gpu/*.csv`, `plots_two_gpu/*.png`.
   - RTX 5060 telemetry exists; RTX 5080 telemetry is missing.
2. **Track 2: chunked cache-growth / long-boundary comparison**
   - CSVs: `results/two_gpu_arch/*.csv` plus existing RTX 5060 files referenced by `docs/kv_cache_experiment_inventory.md`.
   - Derived outputs: `plots_two_gpu_arch/*.png`.
   - RTX 5080 coarse telemetry exists; RTX 5060 telemetry is missing.

Neither track is an Oaken reproduction. Both are empirical consumer-GPU KV-cache pressure measurements.

## Capacity-bound evidence

Capacity is the clearest hardware difference.

### Track 1 generate grid

- RTX 5060 full grid: each of `dynamic`, `quantized`, and `no_cache` has 14 `ok` and 6 `oom` rows.
- RTX 5080 comparison subset: each mode has 19 `ok` and 1 `oom` row.
- Default-harness maximum successful sequence length:
  - RTX 5060: `8192/4096/2048/1024` for batches `1/2/4/8`.
  - RTX 5080: `8192/8192/8192/4096` for batches `1/2/4/8`.

Caveat: the default Track 1 harness measures unquantized prefill KV bytes before generation. A targeted RTX 5060 continuation pass shows dynamic and HQQ-quantized generation both succeed through `batch=8, seq_len=4096` and both OOM at `batch=8, seq_len=8192`. Thus, in Track 1, quantized cache reduces peak memory but does **not** extend the tested high-context generation boundary beyond dynamic.

### Track 2 chunked long-boundary grid

The chunked matched grid in `results/two_gpu_arch/two_gpu_qwen25_matched_grid.csv` reports a stronger capacity-rescue result: RTX 5060 dynamic OOMs at `batch=8, seq_len=12288` and `batch=8, seq_len=16384`, while quantized rows complete those same pairs. RTX 5080 dynamic also completes those pairs, so quantization is less necessary for capacity on the 16GB card in that track.

Conclusion: quantized cache **can** help by avoiding OOM in the chunked long-boundary evidence, but that rescue did not appear in the Track 1 `generate()` continuation boundary. State which track backs the claim.

## Memory bandwidth / memory hierarchy evidence

Expected bandwidth ratio: `960 / 448 = 2.142857`.

- Track 1 dynamic median RTX 5080/5060 speedup: `1.68x`.
- Track 2 dynamic median RTX 5080/5060 speedup: `2.053x`.

Both are below the CUDA-core ratio and Track 2 is close to the nominal bandwidth ratio. This supports a memory-side/cache-traffic interpretation, but true DRAM bandwidth was not measured with Nsight/CUPTI, so this remains an inference from throughput and memory behavior.

## Compute capacity evidence

Expected CUDA-core ratio: `10752 / 3840 = 2.8`.

- Track 1 dynamic cache does not approach `2.8x`.
- Track 1 no-cache has median speedup `~2.62x`, closer to the CUDA-core ratio, but no-cache is absolutely slow and algorithmically different because it recomputes instead of reusing KV.
- Track 2 dynamic speedup `~2.05x` also stays below core ratio.

Conclusion: dynamic KV-cache decode is not explained by pure CUDA-core scaling. No-cache looks more compute-side, but it is not a practical long-context decode strategy in these measurements.

## Quant/dequant overhead evidence

Quantized cache saves memory and costs throughput.

Track 1 generate ratios:

- RTX 5060 HQQ median `memory_saving_ratio`: `0.784`.
- RTX 5060 HQQ median `throughput_loss_ratio`: `0.839`.
- RTX 5080 inferred-Quanto median `memory_saving_ratio`: `0.784`.
- RTX 5080 inferred-Quanto median `throughput_loss_ratio`: `0.733`.

Track 2 chunked ratios:

- `results/two_gpu_arch/two_gpu_qwen25_derived_metrics.csv` contains the committed derived metrics.
- Quantized cross-GPU speedup is large on the matched successful rows, but the docs and CSVs retain failed/error quantized attempts and backend differences. Do not read that as clean CUDA-core scaling.

Conclusion: quant/dequant/cache-management overhead is material. Backend mismatch (`hqq` vs `quanto`) prevents attributing all quantized behavior to GPU architecture alone.

## Direct answers

### Does quantized cache help mainly by avoiding OOM?

- **Track 2:** yes for the committed RTX 5060 chunked long-boundary evidence at `batch=8, seq_len=12288/16384`.
- **Track 1:** no for the tested `generate()` continuation boundary; dynamic and HQQ-quantized share the same observed high boundary (`batch=8, seq_len=8192` OOM).

### Does RTX 5080 scale according to CUDA-core ratio or memory-bandwidth ratio?

No cleanly. Dynamic speedup is `~1.68x` in Track 1 and `~2.05x` in Track 2. Both are below the `2.8x` CUDA-core ratio. Track 2 is close to the `2.14x` bandwidth ratio, but without hardware counters this is suggestive rather than definitive.

### Does quantization introduce overhead that prevents core scaling?

Likely yes. Quantized rows reduce peak memory but reduce throughput versus dynamic within each GPU/harness, and backend differences make quantized cross-GPU speedups unstable. This is evidence for overhead, not a proof that overhead is the only limiter.

### Is the bottleneck compute-side, memory-side, capacity-side, or overhead-bound?

- **Capacity-side:** strongest conclusion at long contexts.
- **Memory-side/cache-traffic:** likely for dynamic cache, especially in Track 2 where speedup is close to the bandwidth ratio and below the core ratio.
- **Compute-side:** more plausible for no-cache, not for dynamic cache.
- **Overhead-bound:** plausible for quantized cache due quant/dequant and backend behavior.

## Additional profiling needed

To strengthen the architectural claim, run matched dynamic/quantized/no-cache rows on both GPUs with the same container, same quantized backend, and the following profiling:

1. Nsight Compute for achieved DRAM bandwidth, L2 hit rate, SM occupancy, tensor-core/FP16 utilization, and per-kernel memory traffic.
2. Nsight Systems or PyTorch profiler to separate prefill/cache-update time, decode attention, matmul time, quantize, and dequantize kernels.
3. Synchronized power traces and allocator snapshots.
4. Repeated natural-prompt runs in addition to synthetic token grids.

Until those counters exist for both GPUs, this repository should state bottleneck conclusions as latency/throughput/memory based rather than definitive hardware-counter attribution.

## Key files

Track 1:

- `analysis/two_gpu/speedup_5080_over_5060.csv`
- `analysis/two_gpu/dynamic_quantized_ratios.csv`
- `analysis/two_gpu/rtx5060_rescue_boundary_table.csv`
- `plots_two_gpu/speedup_5080_over_5060_vs_seq_len.png`
- `plots_two_gpu/quantized_memory_saving_vs_throughput_loss.png`
- `plots_two_gpu/oom_boundary_map_both_gpus.png`

Track 2:

- `results/two_gpu_arch/two_gpu_qwen25_matched_grid.csv`
- `results/two_gpu_arch/two_gpu_qwen25_derived_metrics.csv`
- `plots_two_gpu_arch/speedup_5080_over_5060_vs_seq_len.png`
- `plots_two_gpu_arch/quantized_memory_saving_vs_throughput_loss.png`
- `plots_two_gpu_arch/oom_boundary_map_5060_5080.png`
