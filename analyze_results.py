#!/usr/bin/env python3
"""Create plots from KV-cache consumer GPU benchmark CSV results."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def bytes_to_gib(series: pd.Series) -> pd.Series:
    return series.astype(float) / (1024**3)


def save_line_plot(df: pd.DataFrame, y: str, ylabel: str, title: str, path: str) -> None:
    plt.figure(figsize=(9, 6))
    for (cache_mode, batch_size), group in df.groupby(["cache_mode", "batch_size"]):
        group = group.sort_values("seq_len")
        label = f"{cache_mode}, b={batch_size}"
        plt.plot(group["seq_len"].to_numpy(), group[y].to_numpy(), marker="o", label=label)
    plt.xlabel("Sequence length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_actual_vs_theory(df: pd.DataFrame, path: str) -> None:
    plt.figure(figsize=(7, 7))
    for cache_mode, group in df.groupby("cache_mode"):
        plt.scatter(
            bytes_to_gib(group["theoretical_kv_bytes"]).to_numpy(),
            bytes_to_gib(group["actual_prefill_kv_bytes"]).to_numpy(),
            label=cache_mode,
            alpha=0.75,
        )
    max_val = max(
        bytes_to_gib(df["theoretical_kv_bytes"]).max(),
        bytes_to_gib(df["actual_prefill_kv_bytes"]).max(),
    )
    plt.plot([0, max_val], [0, max_val], linestyle="--", color="black", linewidth=1, label="y=x")
    plt.xlabel("Theoretical KV size (GiB)")
    plt.ylabel("Actual prefill KV tensor size (GiB)")
    plt.title("Actual KV footprint vs theoretical footprint")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise SystemExit("No status=ok rows to plot. OOM/error rows remain in CSV for analysis.")

    numeric_cols = [
        "seq_len",
        "batch_size",
        "tokens_per_sec",
        "latency_ms",
        "peak_delta_bytes",
        "actual_prefill_kv_bytes",
        "theoretical_kv_bytes",
        "kv_actual_over_theory",
    ]
    for col in numeric_cols:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")

    ok["peak_delta_gib"] = bytes_to_gib(ok["peak_delta_bytes"])
    ok["actual_prefill_kv_gib"] = bytes_to_gib(ok["actual_prefill_kv_bytes"])
    ok["theoretical_kv_gib"] = bytes_to_gib(ok["theoretical_kv_bytes"])

    save_line_plot(
        ok,
        "tokens_per_sec",
        "Generated tokens / second",
        "Decode throughput vs sequence length",
        os.path.join(args.outdir, "throughput_vs_seq_len.png"),
    )
    save_line_plot(
        ok,
        "latency_ms",
        "Latency (ms)",
        "Generation latency vs sequence length",
        os.path.join(args.outdir, "latency_vs_seq_len.png"),
    )
    save_line_plot(
        ok,
        "peak_delta_gib",
        "CUDA peak allocated delta (GiB)",
        "Peak memory delta vs sequence length",
        os.path.join(args.outdir, "peak_delta_memory_vs_seq_len.png"),
    )
    save_line_plot(
        ok,
        "kv_actual_over_theory",
        "Actual / theoretical KV bytes",
        "KV actual/theory ratio vs sequence length",
        os.path.join(args.outdir, "kv_actual_over_theory_vs_seq_len.png"),
    )
    save_actual_vs_theory(ok, os.path.join(args.outdir, "actual_kv_vs_theoretical_kv.png"))

    summary_path = os.path.join(args.outdir, "summary_by_mode.csv")
    summary = ok.groupby(["cache_mode", "batch_size"], as_index=False).agg(
        rows=("status", "count"),
        mean_tokens_per_sec=("tokens_per_sec", "mean"),
        median_latency_ms=("latency_ms", "median"),
        max_peak_delta_gib=("peak_delta_gib", "max"),
        mean_kv_actual_over_theory=("kv_actual_over_theory", "mean"),
    )
    summary.to_csv(summary_path, index=False)
    print(f"Wrote plots and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
