#!/usr/bin/env python3
"""Compile and benchmark one PyTorch layer workload on AWS Neuron."""

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


def artifact_name(args: argparse.Namespace) -> str:
    return (
        f"{args.model}-block{args.block}-{args.layer}-{args.action}-"
        f"b{args.batch}s{args.length}.pt"
    )


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
    parser.add_argument("--warm-up", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("benchmark/neuron/artifacts")
    )
    parser.add_argument(
        "--compiler-workdir", type=Path, default=Path("benchmark/neuron/compiler")
    )
    parser.add_argument("--output", type=Path, help="append one JSON object to this file")
    parser.add_argument("--recompile", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.length <= 0 or args.iters <= 0 or args.warm_up < 0:
        raise SystemExit("batch, length, and iters must be positive; warm-up cannot be negative")
    try:
        import torch
        import torch_neuronx
    except ImportError as error:
        raise SystemExit(
            "This benchmark requires the PyTorch Neuron environment on an Inf2 instance"
        ) from error

    torch.manual_seed(args.seed)
    build_model, config_path = load_builder(args.model)
    model, example_inputs = build_model(
        args.batch,
        args.length,
        args.action,
        args.block,
        args.layer,
        device=None,
        local_path=config_path,
    )
    model.eval()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.compiler_workdir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.artifact_dir / artifact_name(args)
    compiler_workdir = args.compiler_workdir / artifact_path.stem
    cache_hit = artifact_path.exists() and not args.recompile
    compile_seconds = 0.0
    if cache_hit:
        compiled = torch.jit.load(str(artifact_path))
    else:
        compiler_workdir.mkdir(parents=True, exist_ok=True)
        compile_start = time.perf_counter()
        compiled = torch_neuronx.trace(
            model,
            example_inputs,
            compiler_workdir=str(compiler_workdir),
        )
        compile_seconds = time.perf_counter() - compile_start
        torch.jit.save(compiled, str(artifact_path))

    del model
    gc.collect()
    timings_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warm_up):
            compiled(*example_inputs)
        for _ in range(args.iters):
            start = time.perf_counter_ns()
            compiled(*example_inputs)
            end = time.perf_counter_ns()
            timings_ms.append((end - start) / 1_000_000)

    result = {
        "model": args.model,
        "action": args.action,
        "layer": args.layer,
        "block": args.block,
        "batch": args.batch,
        "length": args.length,
        "config": args.config,
        "warm_up": args.warm_up,
        "iters": args.iters,
        "cache_hit": cache_hit,
        "compile_seconds": compile_seconds,
        "latency_mean_ms": statistics.mean(timings_ms),
        "latency_std_ms": statistics.pstdev(timings_ms),
        "latency_p50_ms": percentile(timings_ms, 50),
        "latency_p90_ms": percentile(timings_ms, 90),
        "latency_p99_ms": percentile(timings_ms, 99),
        "throughput_per_second": 1000 * args.batch / statistics.mean(timings_ms),
        "artifact": str(artifact_path),
        "torch_version": torch.__version__,
        "torch_neuronx_version": getattr(torch_neuronx, "__version__", "unknown"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
