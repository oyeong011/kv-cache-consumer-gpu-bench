#!/usr/bin/env python3
"""Plot KV-cache sweep CSVs and derive boundary/rescue tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def memory_column(df: pd.DataFrame) -> str:
    if "peak_vram_used_mib" in df.columns and df["peak_vram_used_mib"].notna().any():
        return "peak_vram_used_mib"
    return "peak_memory_allocated_mib"


def write_empty_csv(path: Path, columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()


def plot_peak_memory(df: pd.DataFrame, output: Path, mem_col: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    success = df[df["success"].astype(str).str.lower() == "true"].copy()
    for (mode, batch), group in success.groupby(["cache_mode", "batch_size"]):
        group = group.sort_values("seq_len")
        ax.plot(group["seq_len"], group[mem_col], marker="o", linewidth=1.6, label=f"{mode} bs={int(batch)}")

    oom = df[df["status"].astype(str).str.upper() == "OOM"].copy()
    if not oom.empty:
        y = oom[mem_col].fillna(df[mem_col].max())
        ax.scatter(oom["seq_len"], y, marker="x", s=80, color="black", label="OOM")

    ax.set_title("Peak Memory vs Sequence Length")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(f"{mem_col} (MiB)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_throughput_memory(df: pd.DataFrame, output: Path, mem_col: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    success = df[df["success"].astype(str).str.lower() == "true"].copy()
    markers = {"dynamic": "o", "quantized": "s", "offloaded": "^", "no_cache": "D"}
    for mode, group in success.groupby("cache_mode"):
        ax.scatter(
            group[mem_col],
            group["throughput_tokens_per_sec"],
            s=42,
            marker=markers.get(mode, "o"),
            label=mode,
            alpha=0.8,
        )
    ax.set_title("Throughput vs Peak Memory")
    ax.set_xlabel(f"{mem_col} (MiB)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_status_matrix(df: pd.DataFrame, output: Path, mem_col: str) -> None:
    columns = [
        "model",
        "cache_mode",
        "batch_size",
        "seq_len",
        "status",
        "success",
        "error_type",
        mem_col,
        "throughput_tokens_per_sec",
        "final_cache_seq_len",
        "theoretical_dynamic_kv_mib",
    ]
    present = [column for column in columns if column in df.columns]
    df.sort_values(["cache_mode", "batch_size", "seq_len"])[present].to_csv(output, index=False)


def write_oom_cases(df: pd.DataFrame, output: Path, mem_col: str) -> pd.DataFrame:
    columns = [
        "model",
        "cache_mode",
        "batch_size",
        "seq_len",
        "status",
        "error_type",
        "error_message",
        mem_col,
        "theoretical_dynamic_kv_mib",
    ]
    present = [column for column in columns if column in df.columns]
    oom = df[df["status"].astype(str).str.upper() == "OOM"].copy()
    if oom.empty:
        write_empty_csv(output, present)
    else:
        oom.sort_values(["cache_mode", "batch_size", "seq_len"])[present].to_csv(output, index=False)
    return oom


def write_dynamic_rescue(df: pd.DataFrame, output: Path, mem_col: str) -> None:
    columns = [
        "model",
        "batch_size",
        "seq_len",
        "rescue_cache_mode",
        "rescue_status",
        "rescue_peak_memory_mib",
        "rescue_throughput_tokens_per_sec",
        "dynamic_error_type",
        "dynamic_error_message",
    ]
    dynamic_oom = df[
        (df["cache_mode"] == "dynamic") & (df["status"].astype(str).str.upper() == "OOM")
    ].copy()
    rows: list[dict[str, object]] = []
    for _, dyn in dynamic_oom.iterrows():
        candidates = df[
            (df["batch_size"] == dyn["batch_size"])
            & (df["seq_len"] == dyn["seq_len"])
            & (df["cache_mode"] != "dynamic")
            & (df["success"].astype(str).str.lower() == "true")
        ]
        for _, candidate in candidates.iterrows():
            rows.append(
                {
                    "model": candidate.get("model", dyn.get("model", "")),
                    "batch_size": int(candidate["batch_size"]),
                    "seq_len": int(candidate["seq_len"]),
                    "rescue_cache_mode": candidate["cache_mode"],
                    "rescue_status": candidate["status"],
                    "rescue_peak_memory_mib": candidate.get(mem_col, ""),
                    "rescue_throughput_tokens_per_sec": candidate.get("throughput_tokens_per_sec", ""),
                    "dynamic_error_type": dyn.get("error_type", ""),
                    "dynamic_error_message": dyn.get("error_message", ""),
                }
            )

    if rows:
        pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
    else:
        write_empty_csv(output, columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KV-cache sweep results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df = numeric(
        df,
        [
            "batch_size",
            "seq_len",
            "throughput_tokens_per_sec",
            "peak_memory_allocated_mib",
            "peak_memory_reserved_mib",
            "peak_vram_used_mib",
            "final_cache_seq_len",
            "theoretical_dynamic_kv_mib",
        ],
    )
    mem_col = memory_column(df)

    plot_peak_memory(df, args.output_dir / "peak_memory_vs_seq_len.png", mem_col)
    plot_throughput_memory(df, args.output_dir / "throughput_vs_peak_memory.png", mem_col)
    write_status_matrix(df, args.output_dir / "status_boundary_matrix.csv", mem_col)
    write_oom_cases(df, args.output_dir / "oom_cases.csv", mem_col)
    write_dynamic_rescue(df, args.output_dir / "dynamic_oom_rescue_cases.csv", mem_col)
    print(f"wrote plots and derived CSVs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
