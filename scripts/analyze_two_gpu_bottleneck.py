#!/usr/bin/env python3
"""Regenerate two-GPU KV-cache bottleneck tables and plots from benchmark CSVs."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


COMPARE_MODES = ["dynamic", "quantized", "no_cache"]
BANDWIDTH_RATIO_EXPECTED = 960 / 448
CUDA_CORE_RATIO_EXPECTED = 10752 / 3840


NUMERIC_COLUMNS = [
    "batch_size",
    "seq_len",
    "max_new_tokens",
    "latency_ms",
    "tokens_per_sec",
    "generated_tokens_total",
    "theoretical_kv_bytes",
    "actual_prefill_kv_bytes",
    "kv_actual_over_theory",
    "base_allocated_bytes",
    "peak_allocated_bytes",
    "peak_delta_bytes",
    "peak_reserved_bytes",
    "free_before_bytes",
    "free_after_bytes",
    "total_vram_bytes",
    "telemetry_samples",
    "gpu_util_mean_pct",
    "gpu_util_max_pct",
    "memory_util_mean_pct",
    "memory_util_max_pct",
    "power_draw_mean_w",
    "power_draw_max_w",
]


OPTIONAL_COLUMNS = {
    "prefill_status": "",
    "prefill_error": "",
    "telemetry_samples": math.nan,
    "gpu_util_mean_pct": math.nan,
    "gpu_util_max_pct": math.nan,
    "memory_util_mean_pct": math.nan,
    "memory_util_max_pct": math.nan,
    "power_draw_mean_w": math.nan,
    "power_draw_max_w": math.nan,
    "quantized_nbits": "",
    "quantized_backend": "",
}


def bytes_to_gib(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") / (1024**3)


def normalize_gpu_name(value: object) -> str:
    text = str(value)
    if "5080" in text:
        return "RTX 5080"
    if "5060" in text:
        return "RTX 5060"
    return text or "unknown"


def load_csv(path: Path, source_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col, default in OPTIONAL_COLUMNS.items():
        if col not in df.columns:
            df[col] = default
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["source_csv"] = str(path)
    df["source_label"] = source_label
    df["gpu_label"] = df["gpu_name"].map(normalize_gpu_name)
    # Existing RTX 5080 quantized rows were produced by the repository script
    # before quantized_backend was a CSV column; that script hard-coded quanto.
    inferred = (df["gpu_label"] == "RTX 5080") & (df["cache_mode"] == "quantized") & (df["quantized_backend"].astype(str) == "")
    df.loc[inferred, "quantized_backend"] = "quanto_inferred_from_script"
    df.loc[inferred, "quantized_nbits"] = 4
    return df


def status_grid(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grid = all_rows[["cache_mode", "batch_size", "seq_len", "max_new_tokens"]].drop_duplicates()
    gpu_labels = ["RTX 5060", "RTX 5080"]
    for _, combo in grid.iterrows():
        if combo["cache_mode"] not in COMPARE_MODES:
            continue
        row = combo.to_dict()
        for gpu in gpu_labels:
            match = all_rows[
                (all_rows["gpu_label"] == gpu)
                & (all_rows["cache_mode"] == combo["cache_mode"])
                & (all_rows["batch_size"] == combo["batch_size"])
                & (all_rows["seq_len"] == combo["seq_len"])
                & (all_rows["max_new_tokens"] == combo["max_new_tokens"])
            ]
            if match.empty:
                row[f"{gpu}_status"] = "missing"
                row[f"{gpu}_tokens_per_sec"] = math.nan
                row[f"{gpu}_peak_delta_bytes"] = math.nan
            else:
                first = match.iloc[0]
                row[f"{gpu}_status"] = first["status"]
                row[f"{gpu}_tokens_per_sec"] = first["tokens_per_sec"]
                row[f"{gpu}_peak_delta_bytes"] = first["peak_delta_bytes"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cache_mode", "batch_size", "seq_len", "max_new_tokens"])


def speedup_table(all_rows: pd.DataFrame) -> pd.DataFrame:
    subset = all_rows[(all_rows["cache_mode"].isin(COMPARE_MODES)) & (all_rows["status"] == "ok")].copy()
    pivot = subset.pivot_table(
        index=["cache_mode", "batch_size", "seq_len", "max_new_tokens"],
        columns="gpu_label",
        values=["tokens_per_sec", "latency_ms", "peak_delta_bytes"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{gpu.replace(' ', '_')}" for metric, gpu in pivot.columns]
    pivot = pivot.reset_index()
    if "tokens_per_sec_RTX_5080" in pivot.columns and "tokens_per_sec_RTX_5060" in pivot.columns:
        pivot["speedup_5080_over_5060"] = pivot["tokens_per_sec_RTX_5080"] / pivot["tokens_per_sec_RTX_5060"]
    else:
        pivot["speedup_5080_over_5060"] = math.nan
    pivot["bandwidth_ratio_expected"] = BANDWIDTH_RATIO_EXPECTED
    pivot["cuda_core_ratio_expected"] = CUDA_CORE_RATIO_EXPECTED
    return pivot.sort_values(["cache_mode", "batch_size", "seq_len", "max_new_tokens"])


def dynamic_quantized_table(all_rows: pd.DataFrame) -> pd.DataFrame:
    subset = all_rows[(all_rows["cache_mode"].isin(["dynamic", "quantized"])) & (all_rows["status"] == "ok")].copy()
    rows = []
    for keys, group in subset.groupby(["gpu_label", "batch_size", "seq_len", "max_new_tokens"], dropna=False):
        by_mode = {row["cache_mode"]: row for _, row in group.iterrows()}
        if "dynamic" not in by_mode or "quantized" not in by_mode:
            continue
        dyn = by_mode["dynamic"]
        quant = by_mode["quantized"]
        rows.append(
            {
                "gpu_label": keys[0],
                "batch_size": keys[1],
                "seq_len": keys[2],
                "max_new_tokens": keys[3],
                "dynamic_tokens_per_sec": dyn["tokens_per_sec"],
                "quantized_tokens_per_sec": quant["tokens_per_sec"],
                "throughput_loss_ratio": quant["tokens_per_sec"] / dyn["tokens_per_sec"],
                "dynamic_peak_delta_bytes": dyn["peak_delta_bytes"],
                "quantized_peak_delta_bytes": quant["peak_delta_bytes"],
                "memory_saving_ratio": quant["peak_delta_bytes"] / dyn["peak_delta_bytes"],
                "dynamic_peak_allocated_bytes": dyn["peak_allocated_bytes"],
                "quantized_peak_allocated_bytes": quant["peak_allocated_bytes"],
                "quantized_backend": quant.get("quantized_backend", ""),
                "quantized_nbits": quant.get("quantized_nbits", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["gpu_label", "batch_size", "seq_len", "max_new_tokens"])


def summary_by_gpu_mode(all_rows: pd.DataFrame) -> pd.DataFrame:
    ok = all_rows[all_rows["status"] == "ok"].copy()
    ok["peak_delta_gib"] = bytes_to_gib(ok["peak_delta_bytes"])
    return ok.groupby(["gpu_label", "cache_mode"], as_index=False).agg(
        rows=("status", "count"),
        mean_tokens_per_sec=("tokens_per_sec", "mean"),
        median_latency_ms=("latency_ms", "median"),
        max_peak_delta_gib=("peak_delta_gib", "max"),
        mean_gpu_util_pct=("gpu_util_mean_pct", "mean"),
        mean_memory_util_pct=("memory_util_mean_pct", "mean"),
        mean_power_draw_w=("power_draw_mean_w", "mean"),
    )


def oom_boundary_table(statuses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in statuses.iterrows():
        for gpu in ["RTX 5060", "RTX 5080"]:
            rows.append(
                {
                    "gpu_label": gpu,
                    "cache_mode": row["cache_mode"],
                    "batch_size": row["batch_size"],
                    "seq_len": row["seq_len"],
                    "max_new_tokens": row["max_new_tokens"],
                    "status": row[f"{gpu}_status"],
                    "tokens_per_sec": row[f"{gpu}_tokens_per_sec"],
                    "peak_delta_bytes": row[f"{gpu}_peak_delta_bytes"],
                }
            )
    out = pd.DataFrame(rows)
    out["peak_delta_gib"] = bytes_to_gib(out["peak_delta_bytes"])
    return out.sort_values(["gpu_label", "cache_mode", "batch_size", "seq_len", "max_new_tokens"])


def save_throughput_plot(all_rows: pd.DataFrame, path: Path) -> None:
    ok = all_rows[(all_rows["status"] == "ok") & (all_rows["cache_mode"].isin(COMPARE_MODES))].copy()
    fig, axes = plt.subplots(1, len(COMPARE_MODES), figsize=(18, 5), sharey=False)
    colors = {"RTX 5060": "#1f77b4", "RTX 5080": "#d62728"}
    for ax, mode in zip(axes, COMPARE_MODES):
        mode_df = ok[ok["cache_mode"] == mode]
        for (gpu, batch), group in mode_df.groupby(["gpu_label", "batch_size"]):
            group = group.sort_values("seq_len")
            ax.plot(
                group["seq_len"],
                group["tokens_per_sec"],
                marker="o",
                color=colors.get(gpu),
                linestyle="-" if batch in (1, 4) else "--",
                alpha=0.85,
                label=f"{gpu}, b={batch}",
            )
        ax.set_title(mode)
        ax.set_xlabel("Sequence length")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Generated tokens/sec")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Throughput vs sequence length: RTX 5060 vs RTX 5080")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_peak_memory_plot(all_rows: pd.DataFrame, path: Path) -> None:
    ok = all_rows[(all_rows["status"] == "ok") & (all_rows["cache_mode"].isin(["dynamic", "quantized"]))].copy()
    ok["peak_delta_gib"] = bytes_to_gib(ok["peak_delta_bytes"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {"dynamic": "#2ca02c", "quantized": "#9467bd"}
    for ax, gpu in zip(axes, ["RTX 5060", "RTX 5080"]):
        gpu_df = ok[ok["gpu_label"] == gpu]
        for (mode, batch), group in gpu_df.groupby(["cache_mode", "batch_size"]):
            group = group.sort_values("seq_len")
            ax.plot(
                group["seq_len"],
                group["peak_delta_gib"],
                marker="o",
                color=colors.get(mode),
                linestyle="-" if batch in (1, 4) else "--",
                alpha=0.85,
                label=f"{mode}, b={batch}",
            )
        ax.set_title(gpu)
        ax.set_xlabel("Sequence length")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("CUDA peak allocated delta (GiB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Peak memory vs sequence length: dynamic vs quantized")
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_speedup_plot(speedups: pd.DataFrame, path: Path) -> None:
    data = speedups.dropna(subset=["speedup_5080_over_5060"]).copy()
    plt.figure(figsize=(10, 6))
    for (mode, batch), group in data.groupby(["cache_mode", "batch_size"]):
        group = group.sort_values("seq_len")
        plt.plot(group["seq_len"], group["speedup_5080_over_5060"], marker="o", label=f"{mode}, b={batch}")
    plt.axhline(BANDWIDTH_RATIO_EXPECTED, color="black", linestyle="--", linewidth=1, label=f"bandwidth ratio {BANDWIDTH_RATIO_EXPECTED:.2f}x")
    plt.axhline(CUDA_CORE_RATIO_EXPECTED, color="gray", linestyle=":", linewidth=1.5, label=f"CUDA-core ratio {CUDA_CORE_RATIO_EXPECTED:.2f}x")
    plt.xlabel("Sequence length")
    plt.ylabel("RTX 5080 throughput / RTX 5060 throughput")
    plt.title("Observed 5080/5060 speedup vs expected hardware ratios")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_saving_loss_plot(ratios: pd.DataFrame, path: Path) -> None:
    if ratios.empty:
        return
    colors = {"RTX 5060": "#1f77b4", "RTX 5080": "#d62728"}
    plt.figure(figsize=(8, 6))
    for gpu, group in ratios.groupby("gpu_label"):
        plt.scatter(
            group["memory_saving_ratio"],
            group["throughput_loss_ratio"],
            s=35 + group["batch_size"].astype(float) * 8,
            alpha=0.75,
            color=colors.get(gpu),
            label=gpu,
        )
        for _, row in group.iterrows():
            plt.annotate(
                f"b{int(row['batch_size'])}/s{int(row['seq_len'])}",
                (row["memory_saving_ratio"], row["throughput_loss_ratio"]),
                fontsize=6,
                alpha=0.7,
            )
    plt.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    plt.axvline(1.0, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("memory_saving_ratio = quantized peak delta / dynamic peak delta")
    plt.ylabel("throughput_loss_ratio = quantized throughput / dynamic throughput")
    plt.title("Quantized memory change vs throughput change")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_oom_boundary_plot(boundary: pd.DataFrame, path: Path) -> None:
    status_to_value = {"ok": 0, "oom": 1, "error": 2, "missing": 3}
    cmap = ListedColormap(["#2ca02c", "#d62728", "#ff7f0e", "#bdbdbd"])
    fig, axes = plt.subplots(len(["RTX 5060", "RTX 5080"]), len(COMPARE_MODES), figsize=(15, 7), sharex=True, sharey=True)
    for i, gpu in enumerate(["RTX 5060", "RTX 5080"]):
        for j, mode in enumerate(COMPARE_MODES):
            ax = axes[i][j]
            part = boundary[(boundary["gpu_label"] == gpu) & (boundary["cache_mode"] == mode)]
            batches = sorted(part["batch_size"].dropna().unique())
            seqs = sorted(part["seq_len"].dropna().unique())
            matrix = []
            labels = []
            for batch in batches:
                row_values = []
                row_labels = []
                for seq in seqs:
                    match = part[(part["batch_size"] == batch) & (part["seq_len"] == seq)]
                    status = str(match.iloc[0]["status"]) if not match.empty else "missing"
                    row_values.append(status_to_value.get(status, 2))
                    row_labels.append(status)
                matrix.append(row_values)
                labels.append(row_labels)
            ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3, aspect="auto")
            ax.set_title(f"{gpu} {mode}")
            ax.set_xticks(range(len(seqs)))
            ax.set_xticklabels([str(int(x)) for x in seqs], rotation=45)
            ax.set_yticks(range(len(batches)))
            ax.set_yticklabels([str(int(x)) for x in batches])
            for y, row_labels in enumerate(labels):
                for x, label in enumerate(row_labels):
                    ax.text(x, y, label, ha="center", va="center", fontsize=7, color="white" if label in {"oom", "error"} else "black")
            if i == len(["RTX 5060", "RTX 5080"]) - 1:
                ax.set_xlabel("Sequence length")
            if j == 0:
                ax.set_ylabel("Batch size")
    fig.suptitle("OOM boundary map (ok / oom / error / missing)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def rescue_boundary_table(rescue_rows: pd.DataFrame) -> pd.DataFrame:
    out = rescue_rows[rescue_rows["cache_mode"].isin(["dynamic", "quantized"])].copy()
    out = out[
        [
            "gpu_label",
            "cache_mode",
            "batch_size",
            "seq_len",
            "max_new_tokens",
            "status",
            "prefill_status",
            "tokens_per_sec",
            "peak_delta_bytes",
            "quantized_backend",
        ]
    ]
    out["peak_delta_gib"] = bytes_to_gib(out["peak_delta_bytes"])
    return out.sort_values(["cache_mode", "batch_size", "seq_len", "max_new_tokens"])


def save_rescue_boundary_plot(boundary: pd.DataFrame, path: Path) -> None:
    status_to_value = {"ok": 0, "oom": 1, "error": 2, "missing": 3}
    cmap = ListedColormap(["#2ca02c", "#d62728", "#ff7f0e", "#bdbdbd"])
    modes = ["dynamic", "quantized"]
    fig, axes = plt.subplots(1, len(modes), figsize=(10, 4), sharex=True, sharey=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        part = boundary[boundary["cache_mode"] == mode]
        batches = sorted(part["batch_size"].dropna().unique())
        seqs = sorted(part["seq_len"].dropna().unique())
        matrix = []
        labels = []
        for batch in batches:
            row_values = []
            row_labels = []
            for seq in seqs:
                match = part[(part["batch_size"] == batch) & (part["seq_len"] == seq)]
                status = str(match.iloc[0]["status"]) if not match.empty else "missing"
                prefill_status = str(match.iloc[0]["prefill_status"]) if not match.empty else "missing"
                label = status if prefill_status == "ok" else f"{status}\nprefill:{prefill_status}"
                row_values.append(status_to_value.get(status, 2))
                row_labels.append(label)
            matrix.append(row_values)
            labels.append(row_labels)
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3, aspect="auto")
        ax.set_title(f"RTX 5060 rescue pass: {mode}")
        ax.set_xticks(range(len(seqs)))
        ax.set_xticklabels([str(int(x)) for x in seqs], rotation=45)
        ax.set_yticks(range(len(batches)))
        ax.set_yticklabels([str(int(x)) for x in batches])
        ax.set_xlabel("Sequence length")
        for y, row_labels in enumerate(labels):
            for x, label in enumerate(row_labels):
                ax.text(x, y, label, ha="center", va="center", fontsize=7, color="white" if "oom" in label else "black")
    axes[0].set_ylabel("Batch size")
    fig.suptitle("RTX 5060 continue-after-prefill-OOM boundary check")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-5060", default="results/results_5060_qwen25_1p5b_arch.csv")
    parser.add_argument("--csv-5080", default="results/results_5080_qwen25_1p5b.csv")
    parser.add_argument("--csv-5060-rescue", default="results/results_5060_qwen25_1p5b_rescue_continue.csv")
    parser.add_argument("--outdir", default="analysis/two_gpu")
    parser.add_argument("--plotdir", default="plots_two_gpu")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    plotdir = Path(args.plotdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    df_5060 = load_csv(Path(args.csv_5060), "rtx5060_arch")
    df_5080 = load_csv(Path(args.csv_5080), "rtx5080_existing")
    all_rows = pd.concat([df_5060, df_5080], ignore_index=True)
    all_rows = all_rows[all_rows["cache_mode"].isin(COMPARE_MODES)].copy()
    all_rows["peak_delta_gib"] = bytes_to_gib(all_rows["peak_delta_bytes"])
    all_rows["peak_allocated_gib"] = bytes_to_gib(all_rows["peak_allocated_bytes"])

    aligned_path = outdir / "two_gpu_aligned_rows.csv"
    all_rows.to_csv(aligned_path, index=False)

    statuses = status_grid(all_rows)
    statuses.to_csv(outdir / "same_grid_status.csv", index=False)

    speedups = speedup_table(all_rows)
    speedups.to_csv(outdir / "speedup_5080_over_5060.csv", index=False)

    ratios = dynamic_quantized_table(all_rows)
    ratios.to_csv(outdir / "dynamic_quantized_ratios.csv", index=False)

    summary = summary_by_gpu_mode(all_rows)
    summary.to_csv(outdir / "summary_by_gpu_mode.csv", index=False)

    boundary = oom_boundary_table(statuses)
    boundary.to_csv(outdir / "oom_boundary_table.csv", index=False)

    pd.DataFrame(
        [
            {
                "bandwidth_ratio_expected": BANDWIDTH_RATIO_EXPECTED,
                "cuda_core_ratio_expected": CUDA_CORE_RATIO_EXPECTED,
                "bandwidth_ratio_note": "GDDR7 bandwidth: 960 GB/s RTX 5080 / 448 GB/s RTX 5060",
                "cuda_core_ratio_note": "CUDA cores: 10752 RTX 5080 / 3840 RTX 5060",
            }
        ]
    ).to_csv(outdir / "expected_hardware_ratios.csv", index=False)

    save_throughput_plot(all_rows, plotdir / "throughput_vs_seq_len_5060_vs_5080.png")
    save_peak_memory_plot(all_rows, plotdir / "peak_memory_vs_seq_len_dynamic_vs_quantized.png")
    save_speedup_plot(speedups, plotdir / "speedup_5080_over_5060_vs_seq_len.png")
    save_saving_loss_plot(ratios, plotdir / "quantized_memory_saving_vs_throughput_loss.png")
    save_oom_boundary_plot(boundary, plotdir / "oom_boundary_map_both_gpus.png")

    rescue_path = Path(args.csv_5060_rescue)
    if rescue_path.exists():
        rescue_rows = load_csv(rescue_path, "rtx5060_rescue_continue")
        rescue_rows = rescue_rows[rescue_rows["cache_mode"].isin(["dynamic", "quantized"])].copy()
        rescue_rows.to_csv(outdir / "rtx5060_rescue_continue_rows.csv", index=False)
        rescue_boundary = rescue_boundary_table(rescue_rows)
        rescue_boundary.to_csv(outdir / "rtx5060_rescue_boundary_table.csv", index=False)
        rescue_ratios = dynamic_quantized_table(rescue_rows)
        rescue_ratios.to_csv(outdir / "rtx5060_rescue_dynamic_quantized_ratios.csv", index=False)
        save_rescue_boundary_plot(rescue_boundary, plotdir / "rtx5060_rescue_boundary_map.png")

    print(f"Wrote CSV summaries to {outdir}")
    print(f"Wrote plots to {plotdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
