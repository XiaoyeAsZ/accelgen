#!/usr/bin/env python3
"""Combine Ascend attention/FFN latency with AccelGen FLOP counts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


RUN_RE = re.compile(
    r"^Running: model=(?P<model>\S+) action=(?P<action>\S+) "
    r"block=(?P<block>\d+) layer=(?P<layer>attention|ffn) "
    r"batch=(?P<batch>\d+) length=(?P<length>\d+) config=(?P<config>\S+)$"
)
FLOPS_RE = re.compile(r"^Flops:\s*(?P<value>[-+0-9.eE]+)$")

LayerKey = tuple[str, int, str, str, int, int, str]
WorkloadKey = tuple[str, int, str, int, int, str]


@dataclass(frozen=True)
class LayerResult:
    latency_ms: float
    flops: float


@dataclass(frozen=True)
class CombinedResult:
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


def read_flops(path: Path) -> dict[LayerKey, float]:
    result: dict[LayerKey, float] = {}
    current: LayerKey | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        run = RUN_RE.match(line)
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
            if current in result:
                raise ValueError(f"duplicate FLOP record for {current} at {path}:{line_number}")
            result[current] = float(flops["value"])
            current = None
    return result


def read_ascend(path: Path, flops: dict[LayerKey, float]) -> dict[WorkloadKey, dict[str, LayerResult]]:
    grouped: dict[WorkloadKey, dict[str, LayerResult]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("backend") != "ascend":
                continue
            layer_key: LayerKey = (
                record["model"],
                int(record["block"]),
                record["action"],
                record["layer"],
                int(record["batch"]),
                int(record["length"]),
                record["config"],
            )
            if layer_key not in flops:
                raise ValueError(f"no FLOP record for {layer_key} at {path}:{line_number}")
            workload_key: WorkloadKey = (
                record["model"],
                int(record["block"]),
                record["action"],
                int(record["batch"]),
                int(record["length"]),
                record["config"],
            )
            layers = grouped.setdefault(workload_key, {})
            if record["layer"] in layers:
                raise ValueError(
                    f"duplicate {record['layer']} record for {workload_key} at {path}:{line_number}"
                )
            layers[record["layer"]] = LayerResult(
                latency_ms=float(record["latency_mean_ms"]),
                flops=flops[layer_key],
            )
    return grouped


def combine(
    grouped: dict[WorkloadKey, dict[str, LayerResult]], power_w: float
) -> list[CombinedResult]:
    combined: list[CombinedResult] = []
    for key, layers in grouped.items():
        if set(layers) != {"attention", "ffn"}:
            raise ValueError(f"missing attention or FFN result for {key}: {set(layers)}")
        model, block, action, batch, length, config = key
        combined.append(
            CombinedResult(
                model=model,
                block=block,
                action=action,
                batch=batch,
                length=length,
                config=config,
                attention_latency_ms=layers["attention"].latency_ms,
                ffn_latency_ms=layers["ffn"].latency_ms,
                flops=layers["attention"].flops + layers["ffn"].flops,
                power_w=power_w,
            )
        )
    return sorted(
        combined,
        key=lambda row: (row.model, row.config, row.action != "prefill", row.length),
    )


def write_csv(rows: list[CombinedResult], path: Path) -> None:
    fields = [
        "hardware", "model", "block", "action", "config", "batch", "length",
        "attention_latency_ms", "ffn_latency_ms", "total_latency_ms", "total_flops",
        "power_w", "estimated_energy_j", "throughput_GFLOPS",
        "energy_efficiency_GFLOPS_per_J",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "hardware": "Ascend 910B3",
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
                    "power_w": row.power_w,
                    "estimated_energy_j": row.energy_j,
                    "throughput_GFLOPS": row.throughput_gflops,
                    "energy_efficiency_GFLOPS_per_J": row.efficiency_gflops_per_j,
                }
            )


def write_report(
    rows: list[CombinedResult], path: Path, source: Path, performance_log: Path
) -> None:
    lines = [
        "# Ascend 910B3 performance",
        "",
        f"Latency source: `{source}` (mean latency)",
        "",
        f"FLOPs source: `{performance_log}`",
        "",
        f"Power assumption: `{rows[0].power_w:g} W` constant board power",
        "",
        "Attention and FFN latency and FLOPs are added before computing throughput. "
        "Energy is estimated as constant power multiplied by combined latency; it is not a measured energy value.",
        "",
    ]
    for model in sorted({row.model for row in rows}):
        selected = [row for row in rows if row.model == model]
        lines.extend(
            [
                f"## {model}",
                "",
                "| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |",
                "|:--|:--|--:|--:|--:|--:|--:|--:|--:|",
            ]
        )
        for row in selected:
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
            "estimated_energy_J = 300 W * combined_latency_ms / 1000",
            "energy_efficiency_GFLOPS_per_J = throughput_GFLOPS / 300 W",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("test/results_all.jsonl"))
    parser.add_argument("--performance-log", type=Path, default=Path("test/performance.log"))
    parser.add_argument("--power-w", type=float, default=300.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/results/ascend-910b3")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.power_w <= 0:
        raise SystemExit("--power-w must be positive")
    rows = combine(read_ascend(args.results, read_flops(args.performance_log)), args.power_w)
    if not rows:
        raise SystemExit(f"no Ascend results found in {args.results}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "ascend_910b3_combined.csv")
    write_report(rows, args.output_dir / "README.md", args.results, args.performance_log)
    print(f"Wrote {len(rows)} combined rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
