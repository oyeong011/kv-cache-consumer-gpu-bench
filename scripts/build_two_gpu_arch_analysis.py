#!/usr/bin/env python3
"""Build two-GPU architectural bottleneck CSVs, plots, and markdown docs.

Inputs are benchmark CSVs only. RTX 5060 files are existing artifacts; RTX 5080
files are the same chunked harness rerun on the attached RTX 5080 host.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "two_gpu_arch"
PLOTS = ROOT / "plots_two_gpu_arch"
DOCS = ROOT / "docs"
BANDWIDTH_RATIO_EXPECTED = 960 / 448
CUDA_CORE_RATIO_EXPECTED = 10752 / 3840

COMMON_COLS = [
    "gpu", "gpu_name", "detected_gpu_name", "model", "resolved_model", "dtype", "cache_mode",
    "batch_size", "seq_len", "chunk_size", "status", "success", "error_type", "error_message",
    "elapsed_sec", "latency_sec", "throughput_tokens_per_sec", "peak_memory_allocated_mib",
    "peak_vram_used_mib", "avg_gpu_util_percent", "max_gpu_util_percent",
    "avg_memory_util_percent", "max_memory_util_percent", "avg_power_draw_w", "max_power_draw_w",
    "kv_actual_over_theory", "kv_theory_mb", "kv_actual_mb", "quant_backend", "measurement", "source_csv", "result_label",
]


def read_csv(path: str | Path, gpu: str, label: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(ROOT / path)
    df["gpu"] = gpu
    df["source_csv"] = path.as_posix()
    df["result_label"] = label
    for col in COMMON_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df["latency_sec"] = pd.to_numeric(df["elapsed_sec"], errors="coerce")
    df["throughput_tokens_per_sec"] = pd.to_numeric(df["throughput_tokens_per_sec"], errors="coerce")
    df["peak_memory_allocated_mib"] = pd.to_numeric(df["peak_memory_allocated_mib"], errors="coerce")
    df["peak_vram_used_mib"] = pd.to_numeric(df["peak_vram_used_mib"], errors="coerce")
    for col in ["avg_gpu_util_percent", "max_gpu_util_percent", "avg_memory_util_percent", "max_memory_util_percent", "avg_power_draw_w", "max_power_draw_w", "kv_actual_over_theory"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce").astype("Int64")
    df["seq_len"] = pd.to_numeric(df["seq_len"], errors="coerce").astype("Int64")
    return df[COMMON_COLS]


def build_normalized() -> pd.DataFrame:
    frames = []
    # Existing RTX 5060 Qwen dynamic/no_cache/rescue grid.
    frames.append(read_csv("results/rtx5060_qwen25_15b_combined.csv", "RTX 5060", "verified from existing file"))
    # Add 5060 quantized sanity pairs that are not in the combined rescue CSV.
    sanity = read_csv("results/rtx5060_qwen25_15b_sanity.csv", "RTX 5060", "verified from existing file")
    sanity = sanity[sanity["cache_mode"].eq("quantized")]
    frames.append(sanity)
    # New RTX 5080 same-harness attempts.
    frames.append(read_csv("results/two_gpu_arch/rtx5080_qwen25_15b_dynamic_grid_telemetry_venv.csv", "RTX 5080", "newly rerun"))
    frames.append(read_csv("results/two_gpu_arch/rtx5080_qwen25_15b_quantized_matched_quanto_telemetry_venv.csv", "RTX 5080", "newly rerun"))
    frames.append(read_csv("results/two_gpu_arch/rtx5080_qwen25_15b_no_cache_matched_telemetry_venv.csv", "RTX 5080", "newly rerun"))
    # Keep failed backend attempts as explicit missing/not-reproducible rows for backend choice evidence.
    frames.append(read_csv("results/two_gpu_arch/rtx5080_qwen25_15b_quantized_matched_telemetry.csv", "RTX 5080", "missing / not reproducible"))
    frames.append(read_csv("results/two_gpu_arch/rtx5080_qwen25_15b_quantized_matched_quanto_telemetry.csv", "RTX 5080", "missing / not reproducible"))
    df = pd.concat(frames, ignore_index=True)
    df["status_norm"] = df["status"].astype(str).str.upper()
    return df


def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status_norm"].eq("OK")].copy()
    # Prefer successful rows. If duplicates exist, newest/source order last wins; keep backend-specific failure rows out.
    ok = ok.sort_values(["gpu", "cache_mode", "batch_size", "seq_len", "result_label", "source_csv"])
    rows = []
    keys = ["cache_mode", "batch_size", "seq_len"]
    for key, group in ok.groupby(keys, dropna=False):
        gpus = {r["gpu"]: r for _, r in group.iterrows()}
        if "RTX 5060" in gpus and "RTX 5080" in gpus:
            r5060, r5080 = gpus["RTX 5060"], gpus["RTX 5080"]
            speedup = r5080["throughput_tokens_per_sec"] / r5060["throughput_tokens_per_sec"] if pd.notna(r5060["throughput_tokens_per_sec"]) and r5060["throughput_tokens_per_sec"] else np.nan
            rows.append({
                "cache_mode": key[0], "batch_size": key[1], "seq_len": key[2],
                "throughput_5060": r5060["throughput_tokens_per_sec"],
                "throughput_5080": r5080["throughput_tokens_per_sec"],
                "speedup_5080_over_5060": speedup,
                "bandwidth_ratio_expected": BANDWIDTH_RATIO_EXPECTED,
                "cuda_core_ratio_expected": CUDA_CORE_RATIO_EXPECTED,
                "peak_alloc_mib_5060": r5060["peak_memory_allocated_mib"],
                "peak_alloc_mib_5080": r5080["peak_memory_allocated_mib"],
                "avg_gpu_util_percent_5080": r5080["avg_gpu_util_percent"],
                "avg_memory_util_percent_5080": r5080["avg_memory_util_percent"],
                "avg_power_draw_w_5080": r5080["avg_power_draw_w"],
            })
    speed = pd.DataFrame(rows)

    # Per-GPU quantized/dynamic ratios where both rows exist for a pair.
    ratio_rows = []
    for (gpu, batch, seq), group in ok.groupby(["gpu", "batch_size", "seq_len"], dropna=False):
        by_mode = {r["cache_mode"]: r for _, r in group.iterrows()}
        if "dynamic" in by_mode and "quantized" in by_mode:
            dyn, q = by_mode["dynamic"], by_mode["quantized"]
            ratio_rows.append({
                "gpu": gpu, "batch_size": batch, "seq_len": seq,
                "memory_saving_ratio": q["peak_memory_allocated_mib"] / dyn["peak_memory_allocated_mib"],
                "throughput_loss_ratio": q["throughput_tokens_per_sec"] / dyn["throughput_tokens_per_sec"],
                "dynamic_peak_memory_mib": dyn["peak_memory_allocated_mib"],
                "quantized_peak_memory_mib": q["peak_memory_allocated_mib"],
                "dynamic_throughput": dyn["throughput_tokens_per_sec"],
                "quantized_throughput": q["throughput_tokens_per_sec"],
                "bandwidth_ratio_expected": BANDWIDTH_RATIO_EXPECTED,
                "cuda_core_ratio_expected": CUDA_CORE_RATIO_EXPECTED,
            })
    ratios = pd.DataFrame(ratio_rows)

    if speed.empty:
        speed["metric_type"] = []
    else:
        speed["metric_type"] = "cross_gpu_speedup"
    if ratios.empty:
        ratios["metric_type"] = []
    else:
        ratios["metric_type"] = "quantized_vs_dynamic"
    return pd.concat([speed, ratios], ignore_index=True, sort=False)


def plot_throughput(df: pd.DataFrame) -> None:
    ok = df[df["status_norm"].eq("OK") & df["cache_mode"].isin(["dynamic", "quantized"])].copy()
    for mode in ["dynamic", "quantized"]:
        plt.figure(figsize=(10, 6))
        sub = ok[ok["cache_mode"].eq(mode)]
        for (gpu, batch), g in sub.groupby(["gpu", "batch_size"]):
            g = g.sort_values("seq_len")
            plt.plot(g["seq_len"], g["throughput_tokens_per_sec"], marker="o", label=f"{gpu} b={batch}")
        plt.title(f"Throughput vs sequence length ({mode})")
        plt.xlabel("Sequence length")
        plt.ylabel("Throughput (tokens/sec)")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncols=2)
        plt.tight_layout()
        plt.savefig(PLOTS / f"throughput_vs_seq_len_{mode}_5060_vs_5080.png", dpi=180)
        plt.close()


def plot_peak_memory(df: pd.DataFrame) -> None:
    ok = df[df["status_norm"].eq("OK") & df["cache_mode"].isin(["dynamic", "quantized"])].copy()
    plt.figure(figsize=(10, 6))
    for (gpu, mode), g in ok.groupby(["gpu", "cache_mode"]):
        # Use batch=8 where available, else batch=1 for quantized sanity.
        preferred = g[g["batch_size"].eq(8)]
        if preferred.empty:
            preferred = g[g["batch_size"].eq(1)]
        preferred = preferred.sort_values("seq_len")
        plt.plot(preferred["seq_len"], preferred["peak_memory_allocated_mib"], marker="o", label=f"{gpu} {mode}")
    plt.title("Peak allocated memory vs sequence length")
    plt.xlabel("Sequence length")
    plt.ylabel("Peak CUDA allocated memory (MiB)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS / "peak_memory_vs_seq_len_dynamic_vs_quantized.png", dpi=180)
    plt.close()


def plot_speedup(metrics: pd.DataFrame) -> None:
    speed = metrics[metrics["metric_type"].eq("cross_gpu_speedup")].copy()
    plt.figure(figsize=(10, 6))
    for (mode, batch), g in speed.groupby(["cache_mode", "batch_size"]):
        g = g.sort_values("seq_len")
        plt.plot(g["seq_len"], g["speedup_5080_over_5060"], marker="o", label=f"{mode} b={batch}")
    plt.axhline(BANDWIDTH_RATIO_EXPECTED, color="tab:orange", linestyle="--", label="expected bandwidth ratio 960/448")
    plt.axhline(CUDA_CORE_RATIO_EXPECTED, color="tab:red", linestyle=":", label="expected CUDA-core ratio 10752/3840")
    plt.title("RTX 5080 / RTX 5060 throughput speedup")
    plt.xlabel("Sequence length")
    plt.ylabel("Speedup")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(PLOTS / "speedup_5080_over_5060_vs_seq_len.png", dpi=180)
    plt.close()


def plot_tradeoff(metrics: pd.DataFrame) -> None:
    ratios = metrics[metrics["metric_type"].eq("quantized_vs_dynamic")].copy()
    plt.figure(figsize=(8, 6))
    for gpu, g in ratios.groupby("gpu"):
        plt.scatter(g["memory_saving_ratio"], g["throughput_loss_ratio"], s=80, label=gpu)
        for _, r in g.iterrows():
            plt.annotate(f"b{int(r['batch_size'])}/s{int(r['seq_len'])}", (r["memory_saving_ratio"], r["throughput_loss_ratio"]), fontsize=8)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title("Quantized memory saving vs throughput loss")
    plt.xlabel("quantized_peak_memory / dynamic_peak_memory")
    plt.ylabel("quantized_throughput / dynamic_throughput")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "quantized_memory_saving_vs_throughput_loss.png", dpi=180)
    plt.close()


def plot_oom_map(df: pd.DataFrame) -> None:
    grid = df[df["cache_mode"].isin(["dynamic", "quantized", "no_cache"])].copy()
    grid["ok_value"] = grid["status_norm"].map({"OK": 1, "OOM": -1}).fillna(0)
    panels = [(gpu, mode) for gpu in ["RTX 5060", "RTX 5080"] for mode in ["dynamic", "quantized", "no_cache"]]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=False, sharey=False)
    for ax, (gpu, mode) in zip(axes.ravel(), panels):
        sub = grid[(grid["gpu"].eq(gpu)) & (grid["cache_mode"].eq(mode))]
        if sub.empty:
            ax.set_title(f"{gpu} {mode}\nmissing")
            ax.axis("off")
            continue
        piv = sub.pivot_table(index="batch_size", columns="seq_len", values="ok_value", aggfunc="last").sort_index().sort_index(axis=1)
        ax.imshow(piv.values, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_title(f"{gpu} {mode}")
        ax.set_xticks(range(len(piv.columns)), [str(int(c)) for c in piv.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index)), [str(int(i)) for i in piv.index])
        ax.set_xlabel("seq_len")
        ax.set_ylabel("batch")
        for y in range(len(piv.index)):
            for x in range(len(piv.columns)):
                val = piv.values[y, x]
                label = "OK" if val == 1 else "OOM" if val == -1 else "ERR/MISS"
                ax.text(x, y, label, ha="center", va="center", fontsize=7)
    fig.suptitle("OOM boundary map (green OK, red OOM, yellow error/missing)")
    fig.tight_layout()
    fig.savefig(PLOTS / "oom_boundary_map_5060_5080.png", dpi=180)
    plt.close(fig)


def status_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["gpu", "cache_mode", "status_norm"], dropna=False).size().reset_index(name="rows")


def write_docs(df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    speed = metrics[metrics["metric_type"].eq("cross_gpu_speedup")]
    ratios = metrics[metrics["metric_type"].eq("quantized_vs_dynamic")]
    dyn_speed = speed[speed["cache_mode"].eq("dynamic")]["speedup_5080_over_5060"].dropna()
    q_speed = speed[speed["cache_mode"].eq("quantized")]["speedup_5080_over_5060"].dropna()
    dyn_median = dyn_speed.median() if not dyn_speed.empty else np.nan
    q_median = q_speed.median() if not q_speed.empty else np.nan
    status = status_table(df)

    summary = f"""# KV-cache two-GPU comparison

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
  --batch-sizes 1 8 --seq-lens 1024 2048 12288 16384 \
  --pairs 1:1024 1:2048 8:12288 8:16384 \
  --cache-modes quantized \
  --output results/two_gpu_arch/rtx5080_qwen25_15b_quantized_matched_quanto_telemetry_venv.csv \
  --chunk-size 128 --quant-backend quanto --quant-bits 4 --vram-sample-interval 0.1

.venv/bin/python scripts/run_chunked_kv_cache_sweep_telemetry.py \
  --model /home/ssu/models/Qwen2.5-1.5B-Instruct \
  --gpu-name rtx5080-16gb --dtype fp16 \
  --batch-sizes 8 --seq-lens 12288 16384 \
  --pairs 8:12288 8:16384 \
  --cache-modes no_cache \
  --output results/two_gpu_arch/rtx5080_qwen25_15b_no_cache_matched_telemetry_venv.csv \
  --chunk-size 128 --vram-sample-interval 0.1

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

{status.to_markdown(index=False)}

## Derived ratios

- `bandwidth_ratio_expected = 960 / 448 = {BANDWIDTH_RATIO_EXPECTED:.3f}`
- `cuda_core_ratio_expected = 10752 / 3840 = {CUDA_CORE_RATIO_EXPECTED:.3f}`
- Median dynamic RTX 5080/5060 speedup over common successful rows: {dyn_median:.3f}x
- Median quantized RTX 5080/5060 speedup over common successful rows: {q_median:.3f}x

## Missing telemetry

RTX 5060 utilization, memory-bandwidth-utilization, and power telemetry are missing because the existing RTX 5060 CSVs did not record `nvidia-smi` utilization/power columns. RTX 5080 telemetry columns were newly collected where `nvidia-smi` exposed them, but they are coarse process-level samples rather than Nsight-grade kernel metrics.
"""
    (DOCS / "kv_cache_two_gpu_comparison.md").write_text(summary)

    arch = f"""# KV-cache architectural bottleneck analysis

## Question

When quantized KV-cache is used, is the RTX 5060 vs RTX 5080 performance gap mainly explained by compute capacity, memory bandwidth, VRAM capacity, or quant/dequant overhead?

## Short answer

The observed behavior is **primarily capacity-bound at the long-context boundary**, with a **memory-side/cache-traffic component** and visible **quant/dequant overhead**. It does **not** scale like a pure CUDA-core-count workload: the common dynamic-cache median speedup is {dyn_median:.2f}x versus the expected CUDA-core ratio of {CUDA_CORE_RATIO_EXPECTED:.2f}x, and it is closer to—but still not identical to—the expected memory-bandwidth ratio of {BANDWIDTH_RATIO_EXPECTED:.2f}x. Quantized-cache speedup is even less stable because backend differences and quant/dequant work interfere with straightforward core scaling.

## Evidence boundaries

- Do not read this as a full Oaken reproduction.
- RTX 5060 and RTX 5080 are separated in `results/two_gpu_arch/two_gpu_qwen25_matched_grid.csv`.
- RTX 5060 telemetry for utilization, bandwidth utilization, and power is missing; conclusions use latency, throughput, peak memory, and OOM status plus coarse RTX 5080 telemetry only.
- RTX 5060 quantized rows used `hqq`; RTX 5080 successful quantized rerun used `quanto` because the system Python lacked quantization packages and the venv provided Quanto. Backend mismatch makes quantized cross-GPU throughput a suggestive, not definitive, architectural comparison.

## Does quantized cache help mainly by avoiding OOM?

Yes for the strongest RTX 5060 Qwen evidence: dynamic cache OOMs at batch=8 seq_len=12288 and 16384, while quantized cache completes those same pairs. That is a capacity-bound rescue result. On RTX 5080, dynamic also completes those pairs, so quantization is not needed for capacity there; it reduces peak allocated memory but can reduce throughput.

## Does RTX 5080 scale according to CUDA core ratio or memory bandwidth ratio?

No. The dynamic-cache median speedup over common successful rows is {dyn_median:.2f}x. This is below the CUDA-core ratio ({CUDA_CORE_RATIO_EXPECTED:.2f}x) and closer to the memory-bandwidth ratio ({BANDWIDTH_RATIO_EXPECTED:.2f}x), but not a clean match. That pattern suggests the benchmark is not purely compute-bound; KV-cache growth and memory movement are important.

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
"""
    (DOCS / "kv_cache_architectural_bottleneck_analysis.md").write_text(arch)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    df = build_normalized()
    metrics = derive_metrics(df)
    df.to_csv(OUT / "two_gpu_qwen25_matched_grid.csv", index=False)
    metrics.to_csv(OUT / "two_gpu_qwen25_derived_metrics.csv", index=False)
    status_table(df).to_csv(OUT / "two_gpu_qwen25_status_summary.csv", index=False)
    plot_throughput(df)
    plot_peak_memory(df)
    plot_speedup(metrics)
    plot_tradeoff(metrics)
    plot_oom_map(df)
    write_docs(df, metrics)
    print(f"wrote {OUT}, {PLOTS}, docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
