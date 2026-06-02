# KV-cache experiment inventory
Status: updated 2026-06-02. Result labels use the required vocabulary: `verified from existing file`, `newly rerun`, or `missing / not reproducible`.
## Relevant commits
```text
4a3f5e0 Compare dynamic and quantized KV-cache behavior
10ad633 Surface RTX 5080 findings first
2b7a818 Explain RTX 5080 KV-cache trade-offs
751aaf7 Add RTX 5080 KV-cache benchmark results
f18aa81 Keep GitHub publishing on main
09d5dbf Measure KV-cache pressure on consumer GPUs
```

Publication commit: the commit containing this inventory file (`Publish separated KV-cache GPU evidence`).
## Artifact inventory
| Path | Type | Label | Notes |
| --- | --- | --- | --- |
| `analysis/rtx5080_key_examples.csv` | csv | verified from existing file | exists in this repo |
| `analysis/rtx5080_kv_theory_check.csv` | csv | verified from existing file | exists in this repo |
| `analysis/rtx5080_latency_summary.csv` | csv | verified from existing file | exists in this repo |
| `analysis/rtx5080_oom_cases.csv` | csv | verified from existing file | exists in this repo |
| `analysis/rtx5080_peak_memory_summary.csv` | csv | verified from existing file | exists in this repo |
| `analysis/rtx5080_qwen25_analysis.md` | log/doc | verified from existing file | exists in this repo |
| `analysis/rtx5080_ratios_vs_dynamic.csv` | csv | verified from existing file | exists in this repo |
| `analysis/rtx5080_status_count.csv` | csv | verified from existing file | exists in this repo |
| `analysis/rtx5080_throughput_summary.csv` | csv | verified from existing file | exists in this repo |
| `analyze_results.py` | script | verified from existing file | exists in this repo |
| `docs/kv_cache_experiment_inventory.md` | log/doc | newly rerun | exists in this repo |
| `docs/kv_cache_two_machine_results.md` | log/doc | newly rerun | exists in this repo |
| `logs/env_rtx5060_existing_file.txt` | log/doc | verified from existing file | exists in this repo |
| `logs/env_rtx5080_current.txt` | log/doc | newly rerun | exists in this repo |
| `plot_dynamic_vs_quantized.py` | script | verified from existing file | exists in this repo |
| `plots_5060_opt13b/dynamic_oom_rescue_cases.csv` | csv | newly rerun | exists in this repo |
| `plots_5060_opt13b/oom_cases.csv` | csv | newly rerun | exists in this repo |
| `plots_5060_opt13b/peak_memory_vs_seq_len.png` | png | newly rerun | exists in this repo |
| `plots_5060_opt13b/status_boundary_matrix.csv` | csv | newly rerun | exists in this repo |
| `plots_5060_opt13b/throughput_vs_peak_memory.png` | png | newly rerun | exists in this repo |
| `plots_5060_qwen25_15b/dynamic_oom_rescue_cases.csv` | csv | newly rerun | exists in this repo |
| `plots_5060_qwen25_15b/oom_cases.csv` | csv | newly rerun | exists in this repo |
| `plots_5060_qwen25_15b/peak_memory_vs_seq_len.png` | png | newly rerun | exists in this repo |
| `plots_5060_qwen25_15b/status_boundary_matrix.csv` | csv | newly rerun | exists in this repo |
| `plots_5060_qwen25_15b/throughput_vs_peak_memory.png` | png | newly rerun | exists in this repo |
| `plots_5080/actual_kv_vs_theoretical_kv.png` | png | newly rerun | exists in this repo |
| `plots_5080/kv_actual_over_theory_vs_seq_len.png` | png | newly rerun | exists in this repo |
| `plots_5080/latency_vs_seq_len.png` | png | newly rerun | exists in this repo |
| `plots_5080/peak_delta_memory_vs_seq_len.png` | png | newly rerun | exists in this repo |
| `plots_5080/summary_by_mode.csv` | csv | newly rerun | exists in this repo |
| `plots_5080/throughput_vs_seq_len.png` | png | newly rerun | exists in this repo |
| `plots_5080_20260527/actual_kv_vs_theoretical_kv.png` | png | newly rerun | exists in this repo |
| `plots_5080_20260527/kv_actual_over_theory_vs_seq_len.png` | png | newly rerun | exists in this repo |
| `plots_5080_20260527/latency_vs_seq_len.png` | png | newly rerun | exists in this repo |
| `plots_5080_20260527/peak_delta_memory_vs_seq_len.png` | png | newly rerun | exists in this repo |
| `plots_5080_20260527/summary_by_mode.csv` | csv | newly rerun | exists in this repo |
| `plots_5080_20260527/throughput_vs_seq_len.png` | png | newly rerun | exists in this repo |
| `plots_5080_opt6_7b/eval-oom-vram.png` | png | newly rerun | exists in this repo |
| `plots_5080_opt6_7b/original-no-expandable-oom-vram.png` | png | newly rerun | exists in this repo |
| `plots_5080_opt6_7b/original-vram.png` | png | newly rerun | exists in this repo |
| `plots_5080_opt6_7b/profile-vram.png` | png | newly rerun | exists in this repo |
| `plots_dynamic_vs_quantized_20260527/dynamic_vs_quantized_comparison.csv` | csv | newly rerun | exists in this repo |
| `plots_dynamic_vs_quantized_20260527/dynamic_vs_quantized_memory_batch4.png` | png | newly rerun | exists in this repo |
| `plots_dynamic_vs_quantized_20260527/dynamic_vs_quantized_tps_batch4.png` | png | newly rerun | exists in this repo |
| `plots_dynamic_vs_quantized_20260527/q_over_dynamic_memory_ratio.png` | png | newly rerun | exists in this repo |
| `plots_dynamic_vs_quantized_20260527/q_over_dynamic_throughput_ratio.png` | png | newly rerun | exists in this repo |
| `results/experiment_20260527_summary.md` | log/doc | verified from existing file | exists in this repo |
| `results/results_5080_qwen25_1p5b.csv` | csv | verified from existing file | exists in this repo |
| `results/results_5080_qwen25_1p5b_20260527.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5060_opt13b_combined.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5060_opt13b_dynamic_boundary.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5060_opt13b_rescue_cases.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5060_qwen25_15b_combined.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5060_qwen25_15b_dynamic_boundary.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5060_qwen25_15b_rescue_cases.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5060_qwen25_15b_sanity.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5080_opt6_7b/eval-oom-vram.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5080_opt6_7b/hardware.txt` | log/doc | verified from existing file | exists in this repo |
| `results/rtx5080_opt6_7b/logs.md` | log/doc | verified from existing file | exists in this repo |
| `results/rtx5080_opt6_7b/original-no-expandable-oom-vram.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5080_opt6_7b/original-vram.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5080_opt6_7b/profile-vram.csv` | csv | verified from existing file | exists in this repo |
| `results/rtx5080_opt6_7b/summary.md` | log/doc | verified from existing file | exists in this repo |
| `results/smoke_5080_qwen_quantized.csv` | csv | verified from existing file | exists in this repo |
| `results/smoke_cache_modes.csv` | csv | verified from existing file | exists in this repo |
| `results/smoke_oom.csv` | csv | verified from existing file | exists in this repo |
| `results/smoke_quantized.csv` | csv | verified from existing file | exists in this repo |
| `results/smoke_tinyllama.csv` | csv | verified from existing file | exists in this repo |
| `run_kv_cache_bench.py` | script | verified from existing file | exists in this repo |
| `scripts/plot_kv_cache_sweep.py` | script | verified from existing file | exists in this repo |
| `scripts/plot_opt_vram.py` | script | newly rerun | exists in this repo |
| `scripts/push_to_github_template.sh` | script | verified from existing file | exists in this repo |
| `scripts/run_5060_tinyllama.sh` | script | verified from existing file | exists in this repo |
| `scripts/run_5080_qwen.sh` | script | verified from existing file | exists in this repo |
| `scripts/run_chunked_kv_cache_sweep.py` | script | verified from existing file | exists in this repo |

## Source provenance for imported evidence
- RTX 5060 Qwen/OPT CSVs and 5060 environment log were imported from `/home/ssu/oaken/results/` and are labeled `verified from existing file`; they were not rerun on this RTX 5080 host.
- RTX 5080 OPT-6.7B boundary logs/VRAM CSVs were imported from `/home/ssu/oaken/results/rtx5080/opt-6.7b/` and are labeled `verified from existing file`.
- RTX 5080 Qwen benchmark CSVs already existed in this repository and are labeled `verified from existing file`.
- All PNG plots listed here were regenerated from CSV files during this publication pass and are labeled `newly rerun`; no plot was treated as source data.

## Missing / not reproducible items
- RTX 5060 benchmark reruns are `missing / not reproducible` in this session because the attached machine is `ssu-04` with RTX 5080, not RTX 5060. Existing RTX 5060 files are used without merging their claims into RTX 5080 sections.
- Exact RTX 5060 shell transcripts are `missing / not reproducible`; the results document provides reconstructed equivalent commands from CSV columns and the copied harness.
- RTX 5080 Qwen benchmark rerun was not performed during this pass; the existing two CSVs are used as file-backed evidence and plots were regenerated from them.
