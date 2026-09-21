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


def read_flops(paths: list[Path]) -> dict[LayerKey, float]:
    result: dict[LayerKey, float] = {}
    for path in paths:
        current: LayerKey | None = None
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            run = RUN_RE.match(line)
            if run:
                current = (
                    run["model"], int(run["block"]), run["action"], run["layer"],
                    int(run["batch"]), int(run["length"]), run["config"],
                )
                continue
            flops = FLOPS_RE.match(line)
            if flops and current is not None:
                value = float(flops["value"])
                if current in result and result[current] != value:
                    raise ValueError(f"conflicting FLOP record for {current} at {path}:{line_number}")
                result[current] = value
                current = None
    return result


def read_json_records(path: Path) -> list[dict[str, object]]:
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    records: list[dict[str, object]] = []
    offset = 0
    while True:
        start = text.find("{", offset)
        if start < 0:
            break
        try:
            record, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(record, dict) and "model" in record and "layer" in record:
            records.append(record)
        offset = start + end
    return records


def read_ascend(path: Path, flops: dict[LayerKey, float]) -> dict[WorkloadKey, dict[str, LayerResult]]:
    grouped: dict[WorkloadKey, dict[str, LayerResult]] = {}
    for record in read_json_records(path):
        if record.get("backend") not in (None, "ascend"):
            continue
        batch = int(record["batch"])
        config = record.get("config") or ("edge" if batch == 1 else "server")
        layer_key: LayerKey = (
            str(record["model"]), int(record["block"]), str(record["action"]),
            str(record["layer"]), batch, int(record["length"]), str(config),
        )
        if layer_key not in flops:
            raise ValueError(f"no FLOP record for {layer_key} at {path}")
        workload_key: WorkloadKey = (
            str(record["model"]), int(record["block"]), str(record["action"]),
            batch, int(record["length"]), str(config),
        )
        layers = grouped.setdefault(workload_key, {})
        layer = str(record["layer"])
        if layer in layers:
            raise ValueError(f"duplicate {layer} record for {workload_key} at {path}")
        layers[layer] = LayerResult(
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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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
    rows: list[CombinedResult], path: Path, sources: list[Path], performance_logs: list[Path]
) -> None:
    lines = [
        "# Ascend 910B3 performance",
        "",
        "Latency sources: " + ", ".join(f"`{source}`" for source in sources) + " (mean latency)",
        "",
        "FLOPs sources: " + ", ".join(f"`{source}`" for source in performance_logs),
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
    parser.add_argument(
        "--llama70b-results", type=Path, default=Path("test/perf_910b_llama70b.log")
    )
    parser.add_argument(
        "--llama70b-performance-log", type=Path, default=Path("test/llama_70b.log")
    )
    parser.add_argument("--power-w", type=float, default=300.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/results/ascend-910b3")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.power_w <= 0:
        raise SystemExit("--power-w must be positive")
    result_sources = [args.results, args.llama70b_results]
    performance_logs = [args.performance_log, args.llama70b_performance_log]
    flops = read_flops(performance_logs)
    grouped = read_ascend(args.results, flops)
    for key, layers in read_ascend(args.llama70b_results, flops).items():
        if key in grouped:
            raise ValueError(f"duplicate Ascend workload across result logs: {key}")
        grouped[key] = layers
    rows = combine(grouped, args.power_w)
    if not rows:
        raise SystemExit(f"no Ascend results found in {args.results}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "ascend_910b3_combined.csv")
    write_report(rows, args.output_dir / "README.md", result_sources, performance_logs)
    print(f"Wrote {len(rows)} combined rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
