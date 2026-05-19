#!/usr/bin/env python3
"""Benchmark KV-cache memory growth and cache strategy trade-offs on consumer GPUs.

This is intentionally a measurement harness, not an Oaken reproduction.
It compares raw theoretical KV-cache footprint with actual prefill
past_key_values tensor bytes and measures generate() latency / throughput /
CUDA peak allocator deltas for several Hugging Face cache modes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CSV_COLUMNS = [
    "model_id",
    "gpu_name",
    "dtype",
    "cache_mode",
    "batch_size",
    "seq_len",
    "max_new_tokens",
    "status",
    "error",
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
    "num_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
]


@dataclass
class ModelShape:
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int


DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

DTYPE_BYTES = {
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
}


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def get_attr(config: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return default


def infer_model_shape(config: Any) -> ModelShape:
    num_layers = int(get_attr(config, ["num_hidden_layers", "n_layer", "num_layers"], 0))
    num_attention_heads = int(get_attr(config, ["num_attention_heads", "n_head", "num_heads"], 0))
    num_key_value_heads = int(
        get_attr(config, ["num_key_value_heads", "num_kv_heads", "n_kv_heads"], num_attention_heads)
    )

    head_dim = get_attr(config, ["head_dim", "attention_head_dim"], None)
    if head_dim is None:
        hidden_size = get_attr(config, ["hidden_size", "n_embd", "d_model"], None)
        if hidden_size is None or not num_attention_heads:
            raise ValueError("Cannot infer head_dim from model config")
        head_dim = int(hidden_size) // num_attention_heads
    else:
        head_dim = int(head_dim)

    if not (num_layers and num_attention_heads and num_key_value_heads and head_dim):
        raise ValueError(f"Incomplete model shape: {config}")
    return ModelShape(num_layers, num_attention_heads, num_key_value_heads, head_dim)


def theoretical_kv_bytes(shape: ModelShape, batch_size: int, seq_len: int, dtype_name: str) -> int:
    bytes_per_element = DTYPE_BYTES[dtype_name]
    return int(
        2
        * shape.num_layers
        * batch_size
        * seq_len
        * shape.num_key_value_heads
        * shape.head_dim
        * bytes_per_element
    )


def tensor_bytes(obj: Any) -> int:
    """Recursively sum tensor storage bytes in HF past_key_values/cache objects."""
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())

    total = 0
    # Newer transformers cache classes commonly expose key_cache/value_cache.
    for attr in ("key_cache", "value_cache"):
        if hasattr(obj, attr):
            total += tensor_bytes(getattr(obj, attr))
    if total:
        return total

    # Conservative fallback for cache-like containers.
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            return sum(tensor_bytes(x) for x in obj)
        except TypeError:
            pass
    return 0


def make_inputs(tokenizer: Any, batch_size: int, seq_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    token_id = tokenizer.bos_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        token_id = tokenizer.pad_token_id
    if token_id is None:
        token_id = 0
    input_ids = torch.full((batch_size, seq_len), int(token_id), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def cuda_mem_info() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes), int(total_bytes)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def build_generate_kwargs(cache_mode: str) -> dict[str, Any]:
    if cache_mode == "no_cache":
        return {"use_cache": False}
    if cache_mode == "dynamic":
        return {"use_cache": True, "cache_implementation": "dynamic"}
    if cache_mode == "offloaded":
        return {"use_cache": True, "cache_implementation": "offloaded"}
    if cache_mode == "quantized":
        return {
            "use_cache": True,
            "cache_implementation": "quantized",
            "cache_config": {"nbits": 4, "backend": "quanto"},
        }
    raise ValueError(f"Unsupported cache mode: {cache_mode}")


def prefill_kv_bytes(model: Any, inputs: dict[str, torch.Tensor]) -> int:
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    pkv = getattr(outputs, "past_key_values", None)
    return int(tensor_bytes(pkv))


def bench_one(
    *,
    model: Any,
    tokenizer: Any,
    model_id: str,
    dtype_name: str,
    cache_mode: str,
    batch_size: int,
    seq_len: int,
    max_new_tokens: int,
    shape: ModelShape,
    device: torch.device,
    warmup: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_id": model_id,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dtype": dtype_name,
        "cache_mode": cache_mode,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "max_new_tokens": max_new_tokens,
        "status": "ok",
        "error": "",
        "latency_ms": math.nan,
        "tokens_per_sec": math.nan,
        "generated_tokens_total": 0,
        "theoretical_kv_bytes": theoretical_kv_bytes(shape, batch_size, seq_len, dtype_name),
        "actual_prefill_kv_bytes": math.nan,
        "kv_actual_over_theory": math.nan,
        "base_allocated_bytes": int(torch.cuda.memory_allocated(device) if device.type == "cuda" else 0),
        "peak_allocated_bytes": math.nan,
        "peak_delta_bytes": math.nan,
        "peak_reserved_bytes": math.nan,
        "free_before_bytes": math.nan,
        "free_after_bytes": math.nan,
        "total_vram_bytes": math.nan,
        **asdict(shape),
    }

    inputs: dict[str, torch.Tensor] | None = None
    try:
        inputs = make_inputs(tokenizer, batch_size, seq_len, device)
        actual = prefill_kv_bytes(model, inputs)
        row["actual_prefill_kv_bytes"] = actual
        row["kv_actual_over_theory"] = actual / row["theoretical_kv_bytes"] if row["theoretical_kv_bytes"] else math.nan

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            synchronize_if_cuda(device)

        if warmup:
            warm_inputs = make_inputs(tokenizer, 1, min(seq_len, 32), device)
            with torch.inference_mode():
                _ = model.generate(
                    **warm_inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    **build_generate_kwargs(cache_mode),
                )
            del warm_inputs
            synchronize_if_cuda(device)

        free_before, total_vram = cuda_mem_info()
        row["free_before_bytes"] = free_before
        row["total_vram_bytes"] = total_vram

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            synchronize_if_cuda(device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                **build_generate_kwargs(cache_mode),
            )
        synchronize_if_cuda(device)
        elapsed = time.perf_counter() - start

        generated_tokens_total = int(max(0, generated.shape[-1] - seq_len) * batch_size)
        row["generated_tokens_total"] = generated_tokens_total
        row["latency_ms"] = elapsed * 1000.0
        row["tokens_per_sec"] = generated_tokens_total / elapsed if elapsed > 0 else math.nan
        row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
        row["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0)
        row["peak_delta_bytes"] = int(row["peak_allocated_bytes"] - row["base_allocated_bytes"])
        free_after, _ = cuda_mem_info()
        row["free_after_bytes"] = free_after
        del generated
    except BaseException as exc:  # noqa: BLE001 - this is a benchmark harness; record failures.
        row["status"] = "oom" if is_oom(exc) else "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        if os.environ.get("KV_BENCH_DEBUG"):
            row["error"] += "\n" + traceback.format_exc()
        if device.type == "cuda":
            row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            row["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
            row["peak_delta_bytes"] = int(row["peak_allocated_bytes"] - row["base_allocated_bytes"])
            free_after, total_vram = cuda_mem_info()
            row["free_after_bytes"] = free_after
            row["total_vram_bytes"] = total_vram
            torch.cuda.empty_cache()
    finally:
        del inputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def write_row(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seq-lens", required=True, help="Comma-separated sequence lengths")
    parser.add_argument("--batch-sizes", required=True, help="Comma-separated batch sizes")
    parser.add_argument("--cache-modes", default="dynamic,quantized,offloaded,no_cache")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="fp16")
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    if len(tokenizer) > getattr(model.get_input_embeddings(), "num_embeddings", len(tokenizer)):
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    shape = infer_model_shape(model.config)
    print(f"Loaded {args.model_id} on {device}; shape={shape}", flush=True)

    seq_lens = parse_csv_ints(args.seq_lens)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    cache_modes = parse_csv_strings(args.cache_modes)

    for cache_mode in cache_modes:
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                print(f"case cache={cache_mode} batch={batch_size} seq={seq_len}", flush=True)
                row = bench_one(
                    model=model,
                    tokenizer=tokenizer,
                    model_id=args.model_id,
                    dtype_name=args.dtype,
                    cache_mode=cache_mode,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    max_new_tokens=args.max_new_tokens,
                    shape=shape,
                    device=device,
                    warmup=args.warmup,
                )
                print(f"  -> {row['status']} tps={row['tokens_per_sec']} err={str(row['error'])[:120]}", flush=True)
                write_row(args.out, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
