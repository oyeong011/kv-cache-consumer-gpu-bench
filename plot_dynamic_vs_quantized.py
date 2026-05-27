#!/usr/bin/env python3
"""Plot direct dynamic-cache vs quantized-cache benchmark comparisons."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def bytes_to_gib(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") / (1024**3)


def comparison_frame(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["cache_mode"].isin(["dynamic", "quantized"])].copy()
    df["tokens_per_sec"] = pd.to_numeric(df["tokens_per_sec"], errors="coerce")
    df["peak_delta_gib"] = bytes_to_gib(df["peak_delta_bytes"])

    indexed = df.set_index(["batch_size", "seq_len", "cache_mode"])
    rows = []
    for key, group in df.groupby(["batch_size", "seq_len"]):
        batch_size, seq_len = key
        row = {"batch_size": batch_size, "seq_len": seq_len}
        for mode in ("dynamic", "quantized"):
            try:
                source = indexed.loc[(batch_size, seq_len, mode)]
            except KeyError:
                source = None
            if source is None:
                row[f"{mode}_status"] = "missing"
                row[f"{mode}_tps"] = float("nan")
                row[f"{mode}_peak_gib"] = float("nan")
            else:
                row[f"{mode}_status"] = source["status"]
                row[f"{mode}_tps"] = source["tokens_per_sec"]
                row[f"{mode}_peak_gib"] = source["peak_delta_gib"]
        row["q_tps_over_dyn"] = row["quantized_tps"] / row["dynamic_tps"]
        row["q_peak_over_dyn"] = row["quantized_peak_gib"] / row["dynamic_peak_gib"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["batch_size", "seq_len"])


def plot_ratio_lines(comp: pd.DataFrame, y: str, ylabel: str, title: str, path: str) -> None:
    ok = comp[(comp["dynamic_status"] == "ok") & (comp["quantized_status"] == "ok")].copy()
    plt.figure(figsize=(9, 5.5))
    for batch_size, group in ok.groupby("batch_size"):
        group = group.sort_values("seq_len")
        plt.plot(group["seq_len"], group[y], marker="o", linewidth=2, label=f"batch={batch_size}")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    plt.xlabel("Sequence length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Batch size")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_absolute_lines(comp: pd.DataFrame, batch_size: int, y_suffix: str, ylabel: str, title: str, path: str) -> None:
    group = comp[comp["batch_size"] == batch_size].sort_values("seq_len")
    plt.figure(figsize=(9, 5.5))
    for mode, label in (("dynamic", "Dynamic cache"), ("quantized", "Quantized cache")):
        ok = group[group[f"{mode}_status"] == "ok"]
        plt.plot(ok["seq_len"], ok[f"{mode}_{y_suffix}"], marker="o", linewidth=2, label=label)

        oom = group[group[f"{mode}_status"] != "ok"]
        if not oom.empty:
            ymax = max(group[f"dynamic_{y_suffix}"].max(), group[f"quantized_{y_suffix}"].max())
            plt.scatter(
                oom["seq_len"],
                [ymax * 1.04] * len(oom),
                marker="x",
                s=90,
                label=f"{label} OOM",
            )
    plt.xlabel("Sequence length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--absolute-batch", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    comp = comparison_frame(args.csv)
    comp.to_csv(os.path.join(args.outdir, "dynamic_vs_quantized_comparison.csv"), index=False)

    plot_ratio_lines(
        comp,
        "q_tps_over_dyn",
        "Quantized throughput / dynamic throughput",
        "Quantized vs dynamic throughput ratio",
        os.path.join(args.outdir, "q_over_dynamic_throughput_ratio.png"),
    )
    plot_ratio_lines(
        comp,
        "q_peak_over_dyn",
        "Quantized peak memory / dynamic peak memory",
        "Quantized vs dynamic peak memory ratio",
        os.path.join(args.outdir, "q_over_dynamic_memory_ratio.png"),
    )
    plot_absolute_lines(
        comp,
        args.absolute_batch,
        "tps",
        "Generated tokens / second",
        f"Dynamic vs quantized throughput, batch={args.absolute_batch}",
        os.path.join(args.outdir, f"dynamic_vs_quantized_tps_batch{args.absolute_batch}.png"),
    )
    plot_absolute_lines(
        comp,
        args.absolute_batch,
        "peak_gib",
        "Peak CUDA allocated delta (GiB)",
        f"Dynamic vs quantized peak memory, batch={args.absolute_batch}",
        os.path.join(args.outdir, f"dynamic_vs_quantized_memory_batch{args.absolute_batch}.png"),
    )
    print(f"Wrote dynamic-vs-quantized plots to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
