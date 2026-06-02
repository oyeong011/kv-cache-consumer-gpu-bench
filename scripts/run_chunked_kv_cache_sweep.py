#!/usr/bin/env python3
"""Sweep KV-cache memory boundaries for causal LM inference."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, DynamicCache, QuantizedCache


CSV_COLUMNS = [
    "timestamp",
    "gpu_name",
    "detected_gpu_name",
    "model",
    "resolved_model",
    "dtype",
    "cache_mode",
    "batch_size",
    "seq_len",
    "chunk_size",
    "status",
    "success",
    "error_type",
    "error_message",
    "elapsed_sec",
    "tokens_processed",
    "throughput_tokens_per_sec",
    "peak_memory_allocated_mib",
    "peak_memory_reserved_mib",
    "peak_vram_used_mib",
    "final_vram_used_mib",
    "final_cache_seq_len",
    "oom",
    "position_valid",
    "max_position_embeddings",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "kv_formula_type",
    "kv_theory_mb",
    "kv_actual_mb",
    "kv_actual_over_theory",
    "peak_memory_mb",
    "theoretical_dynamic_kv_mib",
    "cache_tensor_total_mib",
    "cache_tensor_cuda_mib",
    "quant_backend",
    "quant_bits",
    "measurement",
]


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def mib(value: int | float | None) -> str:
    if value is None:
        return ""
    return f"{float(value) / (1024 ** 2):.6f}"


def detected_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cuda-unavailable"
    return torch.cuda.get_device_name(0)


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise SystemExit(f"Unsupported dtype: {name}") from exc


def dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def resolve_model(model: str, local_model_root: Path) -> tuple[str, bool]:
    path = Path(model)
    if path.exists():
        return path.as_posix(), True
    if "/" in model:
        local = local_model_root / model.rsplit("/", 1)[-1]
        if local.exists():
            return local.as_posix(), True
    return model, False


def extend_opt_positions(model: torch.nn.Module, target_seq_len: int) -> bool:
    decoder = getattr(getattr(model, "model", None), "decoder", None)
    embed_positions = getattr(decoder, "embed_positions", None)
    if embed_positions is None or not hasattr(embed_positions, "offset"):
        return False

    offset = int(embed_positions.offset)
    required_embeddings = target_seq_len + offset
    old_weight = embed_positions.weight.data
    if old_weight.shape[0] >= required_embeddings:
        if getattr(model.config, "max_position_embeddings", 0) < target_seq_len:
            model.config.max_position_embeddings = target_seq_len
        return False

    base = old_weight[offset:]
    repeats = math.ceil((required_embeddings - offset) / base.shape[0])
    expanded_body = base.repeat((repeats, 1))[: required_embeddings - offset]
    new_weight = torch.cat([old_weight[:offset], expanded_body], dim=0)

    embed_positions.num_embeddings = required_embeddings
    embed_positions.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    if hasattr(model.config, "max_position_embeddings"):
        model.config.max_position_embeddings = max(model.config.max_position_embeddings, target_seq_len)
    return True


def run_text(command: list[str], timeout: int | None = None) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def query_vram() -> int | None:
    try:
        output = run_text(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if not output:
        return None
    try:
        return int(output.splitlines()[0].strip())
    except ValueError:
        return None


def monitor_vram(stop: threading.Event, samples: list[int], interval_s: float) -> None:
    while not stop.is_set():
        value = query_vram()
        if value is not None:
            samples.append(value)
        stop.wait(interval_s)


def theoretical_dynamic_kv_bytes(config: Any, batch_size: int, seq_len: int, dtype: torch.dtype) -> int:
    shape = kv_shape(config)
    return (
        2
        * shape["num_hidden_layers"]
        * batch_size
        * seq_len
        * shape["num_key_value_heads"]
        * shape["head_dim"]
        * dtype_bytes(dtype)
    )


def text_config(config: Any) -> Any:
    if hasattr(config, "get_text_config"):
        return config.get_text_config(decoder=True)
    return config


def kv_shape(config: Any) -> dict[str, int | str]:
    cfg = text_config(config)
    layers = int(getattr(cfg, "num_hidden_layers"))
    hidden = int(getattr(cfg, "hidden_size"))
    attention_heads = int(getattr(cfg, "num_attention_heads"))
    kv_heads = int(getattr(cfg, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden // attention_heads))
    return {
        "num_hidden_layers": layers,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "kv_formula_type": "gqa_mqa" if kv_heads != attention_heads else "mha",
    }


def position_limit(config: Any) -> int | None:
    value = getattr(config, "_kv_sweep_original_max_position_embeddings", None)
    if value is None:
        value = getattr(text_config(config), "max_position_embeddings", None)
    if value is None:
        return None
    return int(value)


def ratio_string(numerator: int | None, denominator: int | None) -> str:
    if numerator is None or denominator in (None, 0):
        return ""
    return f"{numerator / denominator:.6f}"


def cache_for_mode(mode: str, config: Any, args: argparse.Namespace) -> Any:
    if mode == "dynamic":
        return DynamicCache(config=config)
    if mode == "offloaded":
        return DynamicCache(config=config, offloading=True)
    if mode == "quantized":
        return QuantizedCache(
            backend=args.quant_backend,
            config=config,
            nbits=args.quant_bits,
            residual_length=args.quant_residual_length,
        )
    raise ValueError(f"cache mode does not use a cache object: {mode}")


def tensor_bytes_by_device(obj: Any) -> tuple[int, int]:
    seen: set[int] = set()
    total = 0
    cuda = 0

    def visit(value: Any) -> None:
        nonlocal total, cuda
        obj_id = id(value)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if torch.is_tensor(value):
            size = value.numel() * value.element_size()
            total += size
            if value.is_cuda:
                cuda += size
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
            return
        if hasattr(value, "__dict__") and value.__class__.__module__.startswith("transformers"):
            visit(vars(value))

    visit(obj)
    return total, cuda


def error_type_from_exception(exc: BaseException) -> str:
    message = str(exc).lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "CUDA_OOM"
    if "device-side assert" in message:
        return "CUDA_DEVICE_ASSERT"
    return type(exc).__name__


def run_case(
    model: torch.nn.Module,
    original_model: str,
    resolved_model: str,
    mode: str,
    batch_size: int,
    seq_len: int,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    status = "OK"
    success = False
    error_type = ""
    error_message = ""
    elapsed = 0.0
    tokens_processed = 0
    final_cache_seq_len = 0
    cache_total_bytes = None
    cache_cuda_bytes = None
    cache = None

    samples: list[int] = []
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_vram, args=(stop, samples, args.vram_sample_interval), daemon=True)

    try:
        monitor.start()
        start = time.monotonic()

        if mode == "no_cache":
            for offset in range(0, seq_len, args.chunk_size):
                cur_len = min(args.chunk_size, seq_len - offset)
                input_ids = torch.randint(
                    low=0,
                    high=int(model.config.vocab_size),
                    size=(batch_size, cur_len),
                    device="cuda",
                    dtype=torch.long,
                )
                attention_mask = torch.ones((batch_size, cur_len), device="cuda", dtype=torch.long)
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                tokens_processed += batch_size * cur_len
                del output, input_ids, attention_mask
        else:
            cache = cache_for_mode(mode, model.config, args)
            current_total = 0
            for offset in range(0, seq_len, args.chunk_size):
                cur_len = min(args.chunk_size, seq_len - offset)
                current_total += cur_len
                input_ids = torch.randint(
                    low=0,
                    high=int(model.config.vocab_size),
                    size=(batch_size, cur_len),
                    device="cuda",
                    dtype=torch.long,
                )
                attention_mask = torch.ones((batch_size, current_total), device="cuda", dtype=torch.long)
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                cache = output.past_key_values
                tokens_processed += batch_size * cur_len
                del output, input_ids, attention_mask

            if cache is not None:
                final_cache_seq_len = int(cache.get_seq_length())
                cache_total_bytes, cache_cuda_bytes = tensor_bytes_by_device(cache)

        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        success = True
    except BaseException as exc:
        elapsed = elapsed or 0.0
        error_type = error_type_from_exception(exc)
        status = "OOM" if error_type == "CUDA_OOM" else "ERROR"
        error_message = " ".join(str(exc).split())[:700]
        if args.tracebacks:
            error_message = (error_message + " | " + " ".join(traceback.format_exc().split()))[:1200]
        try:
            if cache is not None and hasattr(cache, "get_seq_length"):
                final_cache_seq_len = int(cache.get_seq_length())
        except Exception:
            final_cache_seq_len = 0
    finally:
        stop.set()
        monitor.join(timeout=5)
        del cache
        gc.collect()
        torch.cuda.empty_cache()

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final_vram = query_vram()
    peak_vram = max(samples) if samples else final_vram
    throughput = (tokens_processed / elapsed) if success and elapsed > 0 else 0.0
    theory_bytes = theoretical_dynamic_kv_bytes(model.config, batch_size, seq_len, dtype)
    shape = kv_shape(model.config)
    max_positions = position_limit(model.config)
    position_valid = "" if max_positions is None else seq_len <= max_positions

    return {
        "timestamp": now_iso(),
        "gpu_name": args.gpu_name,
        "detected_gpu_name": detected_gpu_name(),
        "model": original_model,
        "resolved_model": resolved_model,
        "dtype": args.dtype,
        "cache_mode": mode,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "chunk_size": args.chunk_size,
        "status": status,
        "success": str(success),
        "error_type": error_type,
        "error_message": error_message,
        "elapsed_sec": f"{elapsed:.6f}",
        "tokens_processed": tokens_processed,
        "throughput_tokens_per_sec": f"{throughput:.6f}" if throughput else "",
        "peak_memory_allocated_mib": mib(peak_allocated),
        "peak_memory_reserved_mib": mib(peak_reserved),
        "peak_vram_used_mib": f"{peak_vram:.6f}" if peak_vram is not None else "",
        "final_vram_used_mib": f"{final_vram:.6f}" if final_vram is not None else "",
        "final_cache_seq_len": final_cache_seq_len,
        "oom": status == "OOM",
        "position_valid": position_valid,
        "max_position_embeddings": max_positions if max_positions is not None else "",
        "num_hidden_layers": shape["num_hidden_layers"],
        "num_attention_heads": shape["num_attention_heads"],
        "num_key_value_heads": shape["num_key_value_heads"],
        "head_dim": shape["head_dim"],
        "kv_formula_type": shape["kv_formula_type"],
        "kv_theory_mb": mib(theory_bytes),
        "kv_actual_mb": mib(cache_total_bytes),
        "kv_actual_over_theory": ratio_string(cache_total_bytes, theory_bytes),
        "peak_memory_mb": mib(peak_allocated),
        "theoretical_dynamic_kv_mib": mib(theory_bytes),
        "cache_tensor_total_mib": mib(cache_total_bytes),
        "cache_tensor_cuda_mib": mib(cache_cuda_bytes),
        "quant_backend": args.quant_backend if mode == "quantized" else "",
        "quant_bits": args.quant_bits if mode == "quantized" else "",
        "measurement": "chunked_cache_growth" if mode != "no_cache" else "chunked_no_cache_lower_bound",
    }


def parse_pairs(pairs: list[str] | None) -> set[tuple[int, int]] | None:
    if not pairs:
        return None
    parsed: set[tuple[int, int]] = set()
    for pair in pairs:
        if ":" not in pair:
            raise SystemExit(f"Invalid --pairs entry {pair!r}; expected BATCH:SEQ_LEN")
        batch, seq = pair.split(":", 1)
        parsed.add((int(batch), int(seq)))
    return parsed


def write_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep KV-cache memory boundary cases.")
    parser.add_argument("--model", required=True, help="HF model id or local model directory.")
    parser.add_argument("--gpu-name", required=True, help="GPU label to store in the CSV.")
    parser.add_argument("--dtype", default="fp16", help="fp16, bf16, or fp32.")
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--seq-lens", nargs="+", type=int, required=True)
    parser.add_argument("--cache-modes", nargs="+", choices=["dynamic", "quantized", "offloaded", "no_cache"], required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--pairs", nargs="*", help="Optional exact BATCH:SEQ_LEN pairs to run.")
    parser.add_argument("--local-model-root", type=Path, default=Path("/home/ssu/models"))
    parser.add_argument("--quant-backend", choices=["hqq", "quanto"], default="hqq")
    parser.add_argument("--quant-bits", type=int, default=4)
    parser.add_argument("--quant-residual-length", type=int, default=128)
    parser.add_argument("--vram-sample-interval", type=float, default=0.2)
    parser.add_argument("--tracebacks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this sweep.")

    dtype = dtype_from_name(args.dtype)
    resolved_model, local_files_only = resolve_model(args.model, args.local_model_root)
    max_seq_len = max(args.seq_lens)

    print(f"loading model={args.model} resolved={resolved_model} dtype={args.dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        dtype=dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    original_max_positions = getattr(text_config(model.config), "max_position_embeddings", None)
    if original_max_positions is not None:
        model.config._kv_sweep_original_max_position_embeddings = int(original_max_positions)
    extended = extend_opt_positions(model, max_seq_len)
    if extended:
        print(f"extended OPT positional embeddings to seq_len={max_seq_len}", flush=True)
    model.to("cuda")
    model.eval()

    pairs = parse_pairs(args.pairs)
    write_header(args.output)

    for mode in args.cache_modes:
        for batch_size in args.batch_sizes:
            for seq_len in args.seq_lens:
                if pairs is not None and (batch_size, seq_len) not in pairs:
                    continue
                print(f"run mode={mode} batch={batch_size} seq_len={seq_len}", flush=True)
                row = run_case(model, args.model, resolved_model, mode, batch_size, seq_len, args, dtype)
                append_row(args.output, row)
                print(
                    "result "
                    f"mode={mode} batch={batch_size} seq_len={seq_len} status={row['status']} "
                    f"peak_alloc_mib={row['peak_memory_allocated_mib']} "
                    f"throughput={row['throughput_tokens_per_sec']}",
                    flush=True,
                )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
