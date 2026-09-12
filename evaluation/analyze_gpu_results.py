#!/usr/bin/env python3
"""Combine A100 and Orin latency logs with AccelGen FLOP counts."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


GPU_RUN_RE = re.compile(
    r"^Running: model=(?P<model>\S+) action=(?P<action>\S+) "
    r"block=(?P<block>\d+) layer=(?P<layer>attention|ffn) "
    r"batch=(?P<batch>\d+) length=(?P<length>\d+)$"
)
GPU_LATENCY_RE = re.compile(
    r"^Average latency over \d+ runs:\s*(?P<value>[-+0-9.eE]+) ms$"
)
PERF_RUN_RE = re.compile(
    r"^Running: model=(?P<model>\S+) action=(?P<action>\S+) "
    r"block=(?P<block>\d+) layer=(?P<layer>attention|ffn) "
    r"batch=(?P<batch>\d+) length=(?P<length>\d+) config=(?P<config>\S+)$"
)
FLOPS_RE = re.compile(r"^Flops:\s*(?P<value>[-+0-9.eE]+)$")

LayerKey = tuple[str, int, str, str, int, int, str]
WorkloadKey = tuple[str, str, int, str, int, int, str]


@dataclass(frozen=True)
class LayerResult:
    latency_ms: float
    flops: float


@dataclass(frozen=True)
class CombinedResult:
    hardware: str
    model: str
    block: int
    action: str
    batch: int
    length: int
    config: str
    attention_latency_ms: float
    ffn_latency_ms: float
    flops: float
    power_w: float

    @property
    def latency_ms(self) -> float:
        return self.attention_latency_ms + self.ffn_latency_ms

    @property
    def energy_j(self) -> float:
        return self.power_w * self.latency_ms / 1000

    @property
    def throughput_gflops(self) -> float:
        return self.flops / (self.latency_ms * 1e6)

    @property
    def efficiency_gflops_per_j(self) -> float:
        return self.throughput_gflops / self.power_w


def config_for_batch(batch: int) -> str:
    if batch == 1:
        return "edge"
    if batch == 8:
        return "server"
    raise ValueError(f"cannot infer edge/server config for batch {batch}")


def read_flops(paths: list[Path]) -> dict[LayerKey, float]:
    result: dict[LayerKey, float] = {}
    for path in paths:
        current: LayerKey | None = None
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            run = PERF_RUN_RE.match(line)
            if run:
                current = (
                    run["model"],
                    int(run["block"]),
                    run["action"],
                    run["layer"],
                    int(run["batch"]),
                    int(run["length"]),
                    run["config"],
                )
                continue
            flops = FLOPS_RE.match(line)
            if flops and current is not None:
                value = float(flops["value"])
                if current in result and result[current] != value:
                    raise ValueError(
                        f"conflicting FLOP record for {current} at {path}:{line_number}"
                    )
                result[current] = value
                current = None
    return result


def read_gpu_log(
    path: Path, hardware: str, flops: dict[LayerKey, float]
) -> dict[WorkloadKey, dict[str, LayerResult]]:
    grouped: dict[WorkloadKey, dict[str, LayerResult]] = {}
    current: dict[str, str] | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        run = GPU_RUN_RE.match(line)
        if run:
            if current is not None:
                raise ValueError(f"missing latency before {path}:{line_number}")
            current = run.groupdict()
            continue
        latency = GPU_LATENCY_RE.match(line)
        if latency and current is not None:
            batch = int(current["batch"])
            config = config_for_batch(batch)
            layer_key: LayerKey = (
                current["model"],
                int(current["block"]),
                current["action"],
                current["layer"],
                batch,
                int(current["length"]),
                config,
            )
            if layer_key not in flops:
                raise ValueError(f"no FLOP record for {layer_key} at {path}:{line_number}")
            workload_key: WorkloadKey = (
                hardware,
                current["model"],
                int(current["block"]),
                current["action"],
                batch,
                int(current["length"]),
                config,
            )
            layers = grouped.setdefault(workload_key, {})
            layer = current["layer"]
            if layer in layers:
                raise ValueError(f"duplicate {layer} record for {workload_key}")
            layers[layer] = LayerResult(float(latency["value"]), flops[layer_key])
            current = None
    if current is not None:
        raise ValueError(f"unterminated entry at end of {path}")
    return grouped


def combine(
    grouped: dict[WorkloadKey, dict[str, LayerResult]], powers: dict[str, float]
) -> list[CombinedResult]:
    rows: list[CombinedResult] = []
    for key, layers in grouped.items():
        if set(layers) != {"attention", "ffn"}:
            raise ValueError(f"missing attention or FFN result for {key}: {set(layers)}")
        hardware, model, block, action, batch, length, config = key
        attention = layers["attention"]
        ffn = layers["ffn"]
        rows.append(
            CombinedResult(
                hardware=hardware,
                model=model,
                block=block,
                action=action,
                batch=batch,
                length=length,
                config=config,
                attention_latency_ms=attention.latency_ms,
                ffn_latency_ms=ffn.latency_ms,
                flops=attention.flops + ffn.flops,
                power_w=powers[hardware],
            )
        )
    action_order = {"prefill": 0, "decode": 1}
    return sorted(
        rows,
        key=lambda row: (
            row.hardware,
            row.model,
            action_order.get(row.action, 99),
            row.length,
        ),
    )


def write_csv(rows: list[CombinedResult], path: Path) -> None:
    fields = [
        "hardware", "model", "block", "action", "config", "batch", "length",
        "attention_latency_ms", "ffn_latency_ms", "total_latency_ms", "total_flops",
        "assumed_power_w", "estimated_energy_j", "throughput_GFLOPS",
        "energy_efficiency_GFLOPS_per_J",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "hardware": row.hardware,
                    "model": row.model,
                    "block": row.block,
                    "action": row.action,
                    "config": row.config,
                    "batch": row.batch,
                    "length": row.length,
                    "attention_latency_ms": row.attention_latency_ms,
                    "ffn_latency_ms": row.ffn_latency_ms,
                    "total_latency_ms": row.latency_ms,
                    "total_flops": row.flops,
                    "assumed_power_w": row.power_w,
                    "estimated_energy_j": row.energy_j,
                    "throughput_GFLOPS": row.throughput_gflops,
                    "energy_efficiency_GFLOPS_per_J": row.efficiency_gflops_per_j,
                }
            )


def write_report(
    rows: list[CombinedResult],
    path: Path,
    a100_source: Path,
    orin_source: Path,
    performance_logs: list[Path],
) -> None:
    lines = [
        "# GPU baseline performance",
        "",
        f"A100 latency source: `{a100_source}`",
        "",
        f"Orin latency source: `{orin_source}`",
        "",
        "FLOPs sources: " + ", ".join(f"`{source}`" for source in performance_logs),
        "",
        "Attention and FFN mean latency and FLOPs are added before computing throughput. "
        "Energy uses a fixed-power estimate and is not a measured value.",
        "",
    ]
    for hardware in ("A100", "Orin"):
        hardware_rows = [row for row in rows if row.hardware == hardware]
        if not hardware_rows:
            continue
        lines.extend(
            [
                f"## {hardware}",
                "",
                f"Power assumption: `{hardware_rows[0].power_w:g} W`",
                "",
            ]
        )
        for model in sorted({row.model for row in hardware_rows}):
            lines.extend(
                [
                    f"### {model}",
                    "",
                    "| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |",
                    "|:--|:--|--:|--:|--:|--:|--:|--:|--:|",
                ]
            )
            for row in (item for item in hardware_rows if item.model == model):
                lines.append(
                    f"| {row.action} | {row.config} | {row.batch} | {row.length} | "
                    f"{row.latency_ms:.4f} | {row.flops / 1e9:.4f} | "
                    f"{row.throughput_gflops:.4f} | {row.energy_j:.6f} | "
                    f"{row.efficiency_gflops_per_j:.4f} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Formulas",
            "",
            "```text",
            "combined_latency = attention_mean_latency + ffn_mean_latency",
            "throughput_GFLOPS = total_FLOPs / (combined_latency_ms * 1e6)",
            "estimated_energy_J = assumed_power_W * combined_latency_ms / 1000",
            "energy_efficiency_GFLOPS_per_J = throughput_GFLOPS / assumed_power_W",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a100-log", type=Path, default=Path("test/perf_a100.txt"))
    parser.add_argument("--orin-log", type=Path, default=Path("test/perf_orin.txt"))
    parser.add_argument("--performance-log", type=Path, default=Path("test/performance.log"))
    parser.add_argument(
        "--llama70b-performance-log",
        type=Path,
        default=Path("test/llama_70b.log"),
    )
    parser.add_argument("--a100-power-w", type=float, default=250.0)
    parser.add_argument("--orin-power-w", type=float, default=9.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/results/gpu-baselines")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.a100_power_w <= 0 or args.orin_power_w <= 0:
        raise SystemExit("power assumptions must be positive")
    performance_logs = [args.performance_log, args.llama70b_performance_log]
    flops = read_flops(performance_logs)
    grouped = read_gpu_log(args.a100_log, "A100", flops)
    for key, layers in read_gpu_log(args.orin_log, "Orin", flops).items():
        if key in grouped:
            raise ValueError(f"duplicate workload across GPU logs: {key}")
        grouped[key] = layers
    rows = combine(grouped, {"A100": args.a100_power_w, "Orin": args.orin_power_w})
    if not rows:
        raise SystemExit("no complete GPU records found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "gpu_baselines.csv")
    write_report(
        rows,
        args.output_dir / "README.md",
        args.a100_log,
        args.orin_log,
        performance_logs,
    )
    print(f"Wrote {len(rows)} combined rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
