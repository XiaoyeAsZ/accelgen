#!/usr/bin/env python3
"""Benchmark one existing PyTorch layer workload on an Ascend NPU."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable


def load_builder(model_name: str) -> tuple[Callable[..., Any], Path]:
    model_root = Path(__file__).resolve().parent.parent / "model"
    if model_name == "llama3-8b":
        from benchmark.model.llama3_8b import build_model

        return build_model, model_root / "llama3_8b"
    if model_name == "qwen3-8b":
        from benchmark.model.qwen3_8b import build_model

        return build_model, model_root / "qwen3_8b"
    if model_name == "gemma-7b":
        from benchmark.model.gemma_7b import build_model

        return build_model, model_root / "gemma_7b"
    if model_name == "qwen3-moe":
        from benchmark.model.qwen3_moe import build_model

        return build_model, model_root / "qwen3_moe" / "config.json"
    raise ValueError(f"unsupported model: {model_name}")


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=("llama3-8b", "qwen3-8b", "gemma-7b", "qwen3-moe"),
    )
    parser.add_argument("--action", required=True, choices=("prefill", "decode"))
    parser.add_argument("--layer", required=True, choices=("attention", "ffn"))
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--config", choices=("edge", "server"))
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warm-up", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="append one JSON object to this file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.length <= 0 or args.iters <= 0 or args.warm_up < 0:
        raise SystemExit("batch, length, and iters must be positive; warm-up cannot be negative")
    try:
        import torch
        import torch_npu  # noqa: F401: registers the Ascend backend with torch
    except ImportError as error:
        raise SystemExit(
            "This benchmark requires a matching PyTorch/torch_npu/CANN environment"
        ) from error

    if not args.device.startswith("npu"):
        raise SystemExit(f"--device must be an Ascend device such as npu:0, got {args.device}")
    if not torch.npu.is_available():
        raise SystemExit("torch_npu is installed, but no Ascend NPU is available")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    build_model, config_path = load_builder(args.model)
    model, example_inputs = build_model(
        args.batch,
        args.length,
        args.action,
        args.block,
        args.layer,
        device=device,
        local_path=config_path,
    )
    model.eval()

    timings_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warm_up):
            model(*example_inputs)
        torch.npu.synchronize()
        for _ in range(args.iters):
            start = time.perf_counter_ns()
            model(*example_inputs)
            torch.npu.synchronize()
            end = time.perf_counter_ns()
            timings_ms.append((end - start) / 1_000_000)

    mean_ms = statistics.mean(timings_ms)
    result = {
        "backend": "ascend",
        "model": args.model,
        "action": args.action,
        "layer": args.layer,
        "block": args.block,
        "batch": args.batch,
        "length": args.length,
        "config": args.config,
        "device": str(device),
        "warm_up": args.warm_up,
        "iters": args.iters,
        "latency_mean_ms": mean_ms,
        "latency_std_ms": statistics.pstdev(timings_ms),
        "latency_p50_ms": percentile(timings_ms, 50),
        "latency_p90_ms": percentile(timings_ms, 90),
        "latency_p99_ms": percentile(timings_ms, 99),
        "throughput_per_second": 1000 * args.batch / mean_ms,
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
    }
    if hasattr(torch.npu, "memory_allocated"):
        result["npu_memory_allocated_bytes"] = torch.npu.memory_allocated(device)
    if hasattr(torch.npu, "memory_reserved"):
        result["npu_memory_reserved_bytes"] = torch.npu.memory_reserved(device)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    del model, example_inputs
    gc.collect()
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
