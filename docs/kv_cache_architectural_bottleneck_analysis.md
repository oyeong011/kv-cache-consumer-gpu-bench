# KV-cache architectural bottleneck analysis

## Question

When quantized KV-cache is used, is the RTX 5060 vs RTX 5080 performance gap mainly explained by compute capacity, memory bandwidth, VRAM capacity, or quant/dequant overhead?

## Short answer

The observed behavior is **primarily capacity-bound at the long-context boundary**, with a **memory-side/cache-traffic component** and visible **quant/dequant overhead**. It does **not** scale like a pure CUDA-core-count workload: the common dynamic-cache median speedup is 2.05x versus the expected CUDA-core ratio of 2.80x, and it is closer to—but still not identical to—the expected memory-bandwidth ratio of 2.14x. Quantized-cache speedup is even less stable because backend differences and quant/dequant work interfere with straightforward core scaling.

## Evidence boundaries

- Do not read this as a full Oaken reproduction.
- RTX 5060 and RTX 5080 are separated in `results/two_gpu_arch/two_gpu_qwen25_matched_grid.csv`.
- RTX 5060 telemetry for utilization, bandwidth utilization, and power is missing; conclusions use latency, throughput, peak memory, and OOM status plus coarse RTX 5080 telemetry only.
- RTX 5060 quantized rows used `hqq`; RTX 5080 successful quantized rerun used `quanto` because the system Python lacked quantization packages and the venv provided Quanto. Backend mismatch makes quantized cross-GPU throughput a suggestive, not definitive, architectural comparison.

## Does quantized cache help mainly by avoiding OOM?

Yes for the strongest RTX 5060 Qwen evidence: dynamic cache OOMs at batch=8 seq_len=12288 and 16384, while quantized cache completes those same pairs. That is a capacity-bound rescue result. On RTX 5080, dynamic also completes those pairs, so quantization is not needed for capacity there; it reduces peak allocated memory but can reduce throughput.

## Does RTX 5080 scale according to CUDA core ratio or memory bandwidth ratio?

No. The dynamic-cache median speedup over common successful rows is 2.05x. This is below the CUDA-core ratio (2.80x) and closer to the memory-bandwidth ratio (2.14x), but not a clean match. That pattern suggests the benchmark is not purely compute-bound; KV-cache growth and memory movement are important.

## Does quantization introduce overhead that prevents core scaling?

Likely yes, but this run cannot isolate it perfectly. Quantized rows reduce cache tensor footprint and rescue 5060 long-context OOM cases, but quantized throughput does not scale cleanly with RTX 5080 CUDA cores. The backend mismatch (`hqq` on existing 5060 rows versus `quanto` on new 5080 rows) and the lack of per-kernel profiling mean the overhead conclusion should remain conservative: quant/dequant overhead is visible as a plausible limiter, not proven as the sole limiter.

## Compute-side vs memory-side vs capacity-side interpretation

- **Capacity-bound:** strongest conclusion. RTX 5060 dynamic OOM at long context while quantized succeeds; RTX 5080 dynamic succeeds on the same pairs.
- **Memory-side:** likely. Speedups are closer to memory-bandwidth ratio than CUDA-core ratio, and throughput declines as sequence length grows.
- **Compute-side:** not dominant as a pure explanation because speedup does not approach CUDA-core ratio.
- **Overhead-bound:** plausible for quantized cache, especially where quantized throughput falls below dynamic despite lower memory footprint.

## Additional profiling needed

To make this architectural claim stronger, run:

1. Nsight Compute kernel-level profiling for dynamic and quantized rows at the same batch/seq pairs.
2. PyTorch profiler with CUDA activities to separate prefill/cache-update time from model matmul time.
3. CUPTI/Nsight Systems memory throughput counters, including DRAM throughput, L2 hit rate, achieved occupancy, SM utilization, and kernel launch overhead.
4. Same quantized backend on both GPUs, preferably in the same container image.
5. Fixed decode-style generation benchmark in addition to chunked cache-growth to distinguish cache construction from autoregressive decode.

## Key files

- `results/two_gpu_arch/two_gpu_qwen25_matched_grid.csv`
- `results/two_gpu_arch/two_gpu_qwen25_derived_metrics.csv`
- `plots_two_gpu_arch/speedup_5080_over_5060_vs_seq_len.png`
- `plots_two_gpu_arch/quantized_memory_saving_vs_throughput_loss.png`
- `plots_two_gpu_arch/oom_boundary_map_5060_5080.png`
