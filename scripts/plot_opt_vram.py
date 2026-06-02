#!/usr/bin/env python3
"""Plot nvidia-smi VRAM sample CSV files.

This helper is intentionally CSV-only: it reads already-recorded VRAM sample
CSVs and emits one PNG per input CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def choose_memory_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        normalized = column.lower().strip()
        if "memory.used" in normalized or normalized in {"memory_used_mib", "used_mib", "memory.used [mib]"}:
            return column
    for column in reversed(df.columns):
        if pd.to_numeric(df[column], errors="coerce").notna().any():
            return column
    return None


def plot_csv(csv_path: Path, outdir: Path) -> Path | None:
    df = pd.read_csv(csv_path)
    column = choose_memory_column(df)
    if column is None:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    output = outdir / f"{csv_path.stem}.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(len(series)), series, marker="o")
    plt.grid(True, alpha=0.3)
    plt.xlabel("sample index")
    plt.ylabel("VRAM used (MiB)")
    plt.title(csv_path.name.replace("-vram.csv", " VRAM"))
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    outputs = [plot_csv(path, args.outdir) for path in sorted(args.input_dir.glob("*vram.csv"))]
    for output in outputs:
        if output is not None:
            print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
