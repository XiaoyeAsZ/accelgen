#!/usr/bin/env python3
"""Summarize combined attention and FFN metrics from performance.log."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


RUN_RE = re.compile(
    r"^Running: model=(?P<model>\S+) action=(?P<action>\S+) "
    r"block=(?P<block>\d+) layer=(?P<layer>attention|ffn) "
    r"batch=(?P<batch>\d+) length=(?P<length>\d+) config=(?P<config>\S+)$"
)
LATENCY_RE = re.compile(r"^Latency:\s*(?P<value>[-+0-9.eE]+)$")
ENERGY_RE = re.compile(r"^Energy:\s*(?P<value>[-+0-9.eE]+)$")
FLOPS_RE = re.compile(r"^Flops:\s*(?P<value>[-+0-9.eE]+)$")


@dataclass(frozen=True)
class LayerRecord:
    model: str
    block: int
    action: str
    layer: str
    batch: int
    length: int
    config: str
    latency_ms: float
    energy_j: float
    flops: float


@dataclass(frozen=True)
class CombinedRecord:
    model: str
    block: int
    action: str
    batch: int
    length: int
    config: str
    attention_latency_ms: float
    ffn_latency_ms: float
    attention_energy_j: float
    ffn_energy_j: float
    flops: float

    @property
    def latency_ms(self) -> float:
        return self.attention_latency_ms + self.ffn_latency_ms

    @property
    def energy_j(self) -> float:
        return self.attention_energy_j + self.ffn_energy_j

    @property
    def throughput_gflops(self) -> float:
        return self.flops / (self.latency_ms * 1e6)

    @property
    def efficiency_gflops_per_j(self) -> float:
        return self.flops / (self.energy_j * 1e9)


def parse_log(path: Path) -> list[LayerRecord]:
    records: list[LayerRecord] = []
    current: dict[str, str] | None = None
    latency_ms = energy_j = flops = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        run = RUN_RE.match(line)
        if run:
            if current is not None:
                raise ValueError(f"incomplete entry before {path}:{line_number}")
            current = run.groupdict()
            latency_ms = energy_j = flops = None
            continue
        if current is None:
            continue
        latency_match = LATENCY_RE.match(line)
        if latency_match:
            latency_ms = float(latency_match["value"])
            continue
        energy_match = ENERGY_RE.match(line)
        if energy_match:
            energy_j = float(energy_match["value"])
            continue
        flops_match = FLOPS_RE.match(line)
        if flops_match:
            flops = float(flops_match["value"])
            continue
        if line.startswith("============================="):
            if None in (latency_ms, energy_j, flops):
                raise ValueError(f"missing metric at {path}:{line_number}")
            records.append(
                LayerRecord(
                    model=current["model"],
                    block=int(current["block"]),
                    action=current["action"],
                    layer=current["layer"],
                    batch=int(current["batch"]),
                    length=int(current["length"]),
                    config=current["config"],
                    latency_ms=latency_ms,
                    energy_j=energy_j,
                    flops=flops,
                )
            )
            current = None
    if current is not None:
        raise ValueError(f"unterminated entry at end of {path}")
    return records


def combine(records: list[LayerRecord]) -> list[CombinedRecord]:
    grouped: dict[
        tuple[str, int, str, int, int, str], dict[str, LayerRecord]
    ] = {}
    for record in records:
        key = (
            record.model,
            record.block,
            record.action,
            record.batch,
            record.length,
            record.config,
        )
        layers = grouped.setdefault(key, {})
        if record.layer in layers:
            raise ValueError(f"duplicate {record.layer} record for {key}")
        layers[record.layer] = record

    result: list[CombinedRecord] = []
    for key, layers in grouped.items():
        if set(layers) != {"attention", "ffn"}:
            raise ValueError(f"missing attention or FFN record for {key}: {set(layers)}")
        model, block, action, batch, length, config = key
        attention = layers["attention"]
        ffn = layers["ffn"]
        result.append(
            CombinedRecord(
                model=model,
                block=block,
                action=action,
                batch=batch,
                length=length,
                config=config,
                attention_latency_ms=attention.latency_ms,
                ffn_latency_ms=ffn.latency_ms,
                attention_energy_j=attention.energy_j,
                ffn_energy_j=ffn.energy_j,
                flops=attention.flops + ffn.flops,
            )
        )
    config_order = {"edge": 0, "server": 1}
    action_order = {"prefill": 0, "decode": 1}
    return sorted(
        result,
        key=lambda row: (
            row.model,
            action_order.get(row.action, 99),
            config_order.get(row.config, 99),
            row.length,
        ),
    )


def write_csv(rows: list[CombinedRecord], path: Path) -> None:
    fields = [
        "model", "block", "action", "config", "batch", "length",
        "attention_latency_ms", "ffn_latency_ms", "total_latency_ms",
        "attention_energy_j", "ffn_energy_j", "total_energy_j", "total_flops",
        "throughput_GFLOPS", "energy_efficiency_GFLOPS_per_J",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.model,
                    "block": row.block,
                    "action": row.action,
                    "config": row.config,
                    "batch": row.batch,
                    "length": row.length,
                    "attention_latency_ms": row.attention_latency_ms,
                    "ffn_latency_ms": row.ffn_latency_ms,
                    "total_latency_ms": row.latency_ms,
                    "attention_energy_j": row.attention_energy_j,
                    "ffn_energy_j": row.ffn_energy_j,
                    "total_energy_j": row.energy_j,
                    "total_flops": row.flops,
                    "throughput_GFLOPS": row.throughput_gflops,
                    "energy_efficiency_GFLOPS_per_J": row.efficiency_gflops_per_j,
                }
            )


def write_report(rows: list[CombinedRecord], path: Path, sources: list[Path]) -> None:
    lines = [
        "# AccelGen model performance",
        "",
        "Sources: " + ", ".join(f"`{source}`" for source in sources),
        "",
        "Attention and FFN latency, energy, and FLOPs are added before computing "
        "throughput and energy efficiency.",
        "",
    ]
    for model in sorted({row.model for row in rows}):
        lines.extend(
            [
                f"## {model}",
                "",
                "| Action | Config | Batch | Length | Latency (ms) | Energy (J) | FLOPs (G) | Throughput (GFLOPS) | Efficiency (GFLOPS/J) |",
                "|:--|:--|--:|--:|--:|--:|--:|--:|--:|",
            ]
        )
        for row in (item for item in rows if item.model == model):
            lines.append(
                f"| {row.action} | {row.config} | {row.batch} | {row.length} | "
                f"{row.latency_ms:.6f} | {row.energy_j:.6f} | {row.flops / 1e9:.6f} | "
                f"{row.throughput_gflops:.4f} | {row.efficiency_gflops_per_j:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Formulas",
            "",
            "```text",
            "total_latency_ms = attention_latency_ms + ffn_latency_ms",
            "total_energy_J = attention_energy_J + ffn_energy_J",
            "total_FLOPs = attention_FLOPs + ffn_FLOPs",
            "throughput_GFLOPS = total_FLOPs / (total_latency_ms * 1e6)",
            "energy_efficiency_GFLOPS_per_J = total_FLOPs / (total_energy_J * 1e9)",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("test/performance.log"))
    parser.add_argument(
        "--llama70b-log", type=Path, default=Path("test/llama_70b.log")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/results/model-performance")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = [args.log, args.llama70b_log]
    records: list[LayerRecord] = []
    seen: set[tuple[str, int, str, str, int, int, str]] = set()
    for source in sources:
        for record in parse_log(source):
            key = (
                record.model,
                record.block,
                record.action,
                record.layer,
                record.batch,
                record.length,
                record.config,
            )
            if key in seen:
                raise ValueError(f"duplicate workload across performance logs: {key}")
            seen.add(key)
            records.append(record)
    rows = combine(records)
    if not rows:
        raise SystemExit(f"no complete records found in {args.log}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "model_performance.csv")
    write_report(rows, args.output_dir / "README.md", sources)
    print(f"Wrote {len(rows)} combined rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
