# KV-cache two-machine results

This report keeps RTX 5060 and RTX 5080 evidence separate. It does not claim RTX 5080 results from RTX 5060 data, and it only discusses Qwen RTX 5080 where Qwen files exist in this repository.

## Result-label legend

- `verified from existing file`: benchmark/log data already existed and was re-read.
- `newly rerun`: generated during this pass, limited to environment probe and plots regenerated from CSV.
- `missing / not reproducible`: could not be run on the attached machine or source file is absent.

## RTX 5060 8GB section

| Field | Value | Label | Evidence |
| --- | --- | --- | --- |
| Hostname | `rtx5060-8gb` from CSV `gpu_name`; OS hostname not otherwise available in imported CSV | verified from existing file | `results/rtx5060_qwen25_15b_dynamic_boundary.csv` |
| GPU | NVIDIA GeForce RTX 5060 | verified from existing file | `results/rtx5060_qwen25_15b_dynamic_boundary.csv`, `logs/env_rtx5060_existing_file.txt` |
| VRAM | 8151 MiB | verified from existing file | `logs/env_rtx5060_existing_file.txt` |
| Driver / CUDA / PyTorch | Driver 590.48.01; CUDA 13.1 by `nvidia-smi`; PyTorch 2.11.0+cu130 | verified from existing file | `logs/env_rtx5060_existing_file.txt` |
| Attached-session rerun | Not run: current host is RTX 5080, not RTX 5060 | missing / not reproducible | `logs/env_rtx5080_current.txt` |

### Experiment A: pure KV-cache growth baseline on RTX 5060

| Field | Value |
| --- | --- |
| Exact command used | `missing / not reproducible`: the exact shell transcript was not present in the imported artifacts. Reproducible equivalent from CSV columns and the source harness is `python scripts/run_chunked_kv_cache_sweep.py --model /home/ssu/models/Qwen2.5-1.5B-Instruct --cache-modes dynamic --batch-sizes 1,2,4,8 --seq-lens 1024,2048,4096,8192,12288,16384 --dtype fp16 --chunk-size 128 --out results/rtx5060_qwen25_15b_dynamic_boundary.csv`. |
| Model | `/home/ssu/models/Qwen2.5-1.5B-Instruct` |
| Cache mode | dynamic |
| Batch sizes | 1, 2, 4, 8 |
| Sequence lengths | 1024, 2048, 4096, 8192, 12288, 16384 |
| KV theory/actual | dynamic successful rows have `kv_actual_over_theory=1.0`; Qwen uses `kv_formula_type=gqa_mqa`, `num_key_value_heads=2`, `head_dim=128` |
| Output CSV paths | `results/rtx5060_qwen25_15b_dynamic_boundary.csv`, `results/rtx5060_qwen25_15b_combined.csv` |
| Plot paths | `plots_5060_qwen25_15b/peak_memory_vs_seq_len.png`, `plots_5060_qwen25_15b/throughput_vs_peak_memory.png` |
| Result label | verified from existing file for CSV; newly rerun for plots |

OOM cases:

| cache_mode   |   batch_size |   seq_len | status   |
|:-------------|-------------:|----------:|:---------|
| dynamic      |            8 |     12288 | OOM      |
| dynamic      |            8 |     16384 | OOM      |
### Experiment B: Qwen2.5-1.5B quantized rescue / Oaken-inspired boundary on RTX 5060

| Field | Value |
| --- | --- |
| Exact command used | `missing / not reproducible`: the exact shell transcript was not present in the imported artifacts. Reproducible equivalent from CSV columns and the source harness is `python scripts/run_chunked_kv_cache_sweep.py --model /home/ssu/models/Qwen2.5-1.5B-Instruct --cache-modes quantized,no_cache --batch-sizes 8 --seq-lens 12288,16384 --dtype fp16 --chunk-size 128 --out results/rtx5060_qwen25_15b_rescue_cases.csv`. |
| Model | `/home/ssu/models/Qwen2.5-1.5B-Instruct` |
| Cache modes | quantized, no_cache lower-bound ablation |
| Batch sizes | 8 |
| Sequence lengths | 12288, 16384 |
| Output CSV paths | `results/rtx5060_qwen25_15b_rescue_cases.csv`, `plots_5060_qwen25_15b/dynamic_oom_rescue_cases.csv` |
| Plot paths | `plots_5060_qwen25_15b/peak_memory_vs_seq_len.png`, `plots_5060_qwen25_15b/throughput_vs_peak_memory.png` |
| Result label | verified from existing file for CSV; newly rerun for plots |

Rescue rows:

| cache_mode   |   batch_size |   seq_len | status   |   throughput_tokens_per_sec |   peak_vram_used_mib |   kv_actual_over_theory |
|:-------------|-------------:|----------:|:---------|----------------------------:|---------------------:|------------------------:|
| quantized    |            8 |     12288 | OK       |                     850.672 |                 5903 |                0.288737 |
| quantized    |            8 |     16384 | OK       |                     647.77  |                 6795 |                0.286865 |
| no_cache     |            8 |     12288 | OK       |                   11788     |                 3475 |              nan        |
| no_cache     |            8 |     16384 | OK       |                   11768.8   |                 3475 |              nan        |
### Additional RTX 5060 OPT-1.3B boundary/rescue evidence

| Field | Value |
| --- | --- |
| Model | `facebook/opt-1.3b` / `/home/ssu/models/opt-1.3b` |
| Cache modes | dynamic, quantized, no_cache lower-bound ablation |
| Batch sizes | 1, 2, 4, 8 |
| Sequence lengths | 1024, 2048, 4096, 6144, 8192 |
| Output CSV paths | `results/rtx5060_opt13b_dynamic_boundary.csv`, `results/rtx5060_opt13b_rescue_cases.csv`, `results/rtx5060_opt13b_combined.csv` |
| Plot paths | `plots_5060_opt13b/peak_memory_vs_seq_len.png`, `plots_5060_opt13b/throughput_vs_peak_memory.png` |
| Result label | verified from existing file for CSV; newly rerun for plots |

OPT-1.3B OOM cases:

| cache_mode   |   batch_size |   seq_len | status   |
|:-------------|-------------:|----------:|:---------|
| dynamic      |            4 |      8192 | OOM      |
| dynamic      |            8 |      4096 | OOM      |
| dynamic      |            8 |      6144 | OOM      |
| dynamic      |            8 |      8192 | OOM      |
| quantized    |            8 |      6144 | OOM      |
| quantized    |            8 |      8192 | OOM      |
## RTX 5080 section

| Field | Value | Label | Evidence |
| --- | --- | --- | --- |
| Hostname | `ssu-04` | newly rerun | `logs/env_rtx5080_current.txt` |
| GPU | NVIDIA GeForce RTX 5080 | newly rerun / verified from existing file | `logs/env_rtx5080_current.txt`, `results/results_5080_qwen25_1p5b.csv` |
| VRAM | 16303 MiB (`nvidia-smi`), 16600596480 bytes in PyTorch CSV | verified from existing file / newly rerun | `logs/env_rtx5080_current.txt`, `results/results_5080_qwen25_1p5b.csv` |
| Driver / CUDA / PyTorch | Driver 580.142; CUDA 13.0; PyTorch 2.11.0+cu130 | newly rerun | `logs/env_rtx5080_current.txt` |

### Experiment A/B supporting RTX 5080 Qwen2.5 cache-policy sweep

| Field | Value |
| --- | --- |
| Exact command used | `python run_kv_cache_bench.py --model-id Qwen/Qwen2.5-1.5B-Instruct --seq-lens 512,1024,2048,4096,8192 --batch-sizes 1,2,4,8 --cache-modes dynamic,quantized,offloaded,no_cache --max-new-tokens 64 --dtype fp16 --out results/results_5080_qwen25_1p5b.csv --warmup` |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Cache modes | dynamic, quantized, offloaded, no_cache |
| Batch sizes | 1, 2, 4, 8 |
| Sequence lengths | 512, 1024, 2048, 4096, 8192 |
| Output CSV paths | `results/results_5080_qwen25_1p5b.csv`, `results/results_5080_qwen25_1p5b_20260527.csv` |
| Plot paths | `plots_5080/throughput_vs_seq_len.png`, `plots_5080/latency_vs_seq_len.png`, `plots_5080/peak_delta_memory_vs_seq_len.png`, `plots_5080/actual_kv_vs_theoretical_kv.png`, `plots_5080/kv_actual_over_theory_vs_seq_len.png`, `plots_5080_20260527/throughput_vs_seq_len.png`, `plots_5080_20260527/latency_vs_seq_len.png`, `plots_5080_20260527/peak_delta_memory_vs_seq_len.png`, `plots_5080_20260527/actual_kv_vs_theoretical_kv.png`, `plots_5080_20260527/kv_actual_over_theory_vs_seq_len.png`, `plots_dynamic_vs_quantized_20260527/q_over_dynamic_throughput_ratio.png`, `plots_dynamic_vs_quantized_20260527/q_over_dynamic_memory_ratio.png`, `plots_dynamic_vs_quantized_20260527/dynamic_vs_quantized_tps_batch4.png`, `plots_dynamic_vs_quantized_20260527/dynamic_vs_quantized_memory_batch4.png` |
| Result label | verified from existing file for CSV; newly rerun for plots |

Clean 5080 Qwen OOM cases (`results/results_5080_qwen25_1p5b.csv`):

| cache_mode   |   batch_size |   seq_len | status   |
|:-------------|-------------:|----------:|:---------|
| dynamic      |            8 |      8192 | oom      |
| offloaded    |            8 |      8192 | oom      |
| no_cache     |            8 |      8192 | oom      |
| quantized    |            8 |      8192 | oom      |
Resident-process 5080 Qwen OOM cases (`results/results_5080_qwen25_1p5b_20260527.csv`):

| cache_mode   |   batch_size |   seq_len | status   |
|:-------------|-------------:|----------:|:---------|
| dynamic      |            4 |      8192 | oom      |
| dynamic      |            8 |      4096 | oom      |
| dynamic      |            8 |      8192 | oom      |
| quantized    |            4 |      8192 | oom      |
| quantized    |            8 |      4096 | oom      |
| quantized    |            8 |      8192 | oom      |
| offloaded    |            4 |      8192 | oom      |
| offloaded    |            8 |      4096 | oom      |
| offloaded    |            8 |      8192 | oom      |
| no_cache     |            4 |      8192 | oom      |
| no_cache     |            8 |      4096 | oom      |
| no_cache     |            8 |      8192 | oom      |
### Experiment B: RTX 5080 OPT-family larger-model boundary confirmation

| Field | Value |
| --- | --- |
| Exact command used: original eval | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_perplexity.py -m opt -s 6.7b -t wikitext --quant-method none --gpu-start-idx 0 --gpu-count 1` |
| Exact command used: Oaken profiling | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python oaken_preprocess_activation.py -m opt -s 6.7b -t wikitext -f 0.04 0.9 0.06 -o quantizer/oaken/opt-6.7b.json --gpu-start-idx 0 --gpu-count 1` |
| Exact command used: Oaken eval | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_perplexity.py -m opt -s 6.7b -t wikitext -q quantizer/oaken/opt-6.7b.json --quant-method oaken --gpu-start-idx 0 --gpu-count 1` |
| Model | `facebook/opt-6.7b` / `/data/models/opt-6.7b` |
| Cache/quantization mode | original FP16 baseline; Oaken-inspired quantized eval/profiling artifact |
| Batch sizes / sequence lengths | Not a batch/sequence KV sweep; Wikitext eval chunks from Oaken artifact. Marked not applicable for the CSV grid requirement. |
| OOM cases | baseline without allocator tuning OOM; Oaken eval OOM after 6/141 Wikitext chunks; original eval and profiling completed with `expandable_segments:True` |
| Output CSV paths | `results/rtx5080_opt6_7b/original-vram.csv`, `results/rtx5080_opt6_7b/profile-vram.csv`, `results/rtx5080_opt6_7b/eval-oom-vram.csv`, `results/rtx5080_opt6_7b/original-no-expandable-oom-vram.csv` |
| Plot paths | `plots_5080_opt6_7b/original-vram.png`, `plots_5080_opt6_7b/profile-vram.png`, `plots_5080_opt6_7b/eval-oom-vram.png`, `plots_5080_opt6_7b/original-no-expandable-oom-vram.png` |
| Result label | verified from existing file for logs/CSV; newly rerun for plots |

RTX 5080 OPT-family evidence is not used to claim Qwen behavior, and RTX 5080 Qwen CSVs are not used to claim OPT-6.7B behavior.

## Publication checks

- All referenced CSV/PNG/log/doc paths are intended to exist in this repository.
- Plots were regenerated from CSV files only.
- RTX 5060 and RTX 5080 claims are separated by section.
