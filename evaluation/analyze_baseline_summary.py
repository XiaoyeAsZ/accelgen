#!/usr/bin/env python3
"""Combine attention and FFN baseline rows into throughput/efficiency tables."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


STEM_RE = re.compile(
    r"^(?P<model>.+)-block(?P<block>\d+)-(?P<layer>attention|ffn)-"
    r"(?P<action>prefill|decode)-b(?P<batch>\d+)s(?P<length>\d+)$"
)
PERFORMANCE_RUN_RE = re.compile(
    r"^Running: model=(?P<model>\S+) action=(?P<action>\S+) "
    r"block=(?P<block>\d+) layer=(?P<layer>attention|ffn) "
    r"batch=(?P<batch>\d+) length=(?P<length>\d+) config=(?P<config>\S+)$"
)
FLOPS_RE = re.compile(r"^Flops:\s*(?P<flops>[-+0-9.eE]+)$")
PERF_LATENCY_RE = re.compile(r"^Latency:\s*(?P<value>[-+0-9.eE]+)$")
PERF_ENERGY_RE = re.compile(r"^Energy:\s*(?P<value>[-+0-9.eE]+)$")


@dataclass(frozen=True)
class SourceRow:
    model: str
    block: int
    layer: str
    action: str
    batch: int
    length: int
    config: str
    arch: str
    latency_cycles: float
    energy_uJ: float
    flops: float
    succeeded: int
    failed: int
    exit_code: int


@dataclass(frozen=True)
class CombinedRow:
    model: str
    block: int
    action: str
    batch: int
    length: int
    config: str
    arch: str
    latency_cycles: float
    energy_uJ: float
    flops: float

    @property
    def throughput_gflops(self) -> float:
        return self.flops / self.latency_cycles

    @property
    def energy_eff_gflops_per_j(self) -> float:
        # 1 uJ = 1e-6 J and 1 GFLOP = 1e9 FLOP.
        return self.flops / (self.energy_uJ * 1000)


@dataclass(frozen=True)
class PerformanceMetric:
    latency_ms: float
    energy_j: float
    flops: float


def ffn_expert_group_scale(action: str, config: str) -> int:
    """Correct the baseline pass's forced batch dimension for Qwen3-MoE FFN."""
    if action != "prefill":
        return 1
    if config == "edge":
        return 16
    if config == "server":
        return 2
    raise ValueError(f"unsupported config for FFN scaling: {config}")


def read_source(path: Path) -> list[SourceRow]:
    result: list[SourceRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match = STEM_RE.match(row["mlir_stem"])
            if not match:
                raise ValueError(f"cannot parse mlir_stem: {row['mlir_stem']}")
            config_match = re.search(r"_(edge|server)$", row["linear_arch"])
            if not config_match:
                raise ValueError(f"cannot parse edge/server config: {row['linear_arch']}")
            config = config_match.group(1)
            result.append(
                SourceRow(
                    model=match["model"],
                    block=int(match["block"]),
                    layer=match["layer"],
                    action=match["action"],
                    batch=int(match["batch"]),
                    length=int(match["length"]),
                    config=config,
                    arch=row["linear_arch"].removesuffix("_" + config),
                    latency_cycles=float(row["latency_cycles"]),
                    energy_uJ=float(row["energy_uJ"]),
                    flops=float(row["FLOPs"]),
                    succeeded=int(row["succeeded"]),
                    failed=int(row["failed"]),
                    exit_code=int(row["exit_code"]),
                )
            )
    return result


def read_performance_metrics(
    path: Path,
) -> dict[tuple[str, int, str, str, int, int, str], PerformanceMetric]:
    result: dict[tuple[str, int, str, str, int, int, str], PerformanceMetric] = {}
    current: tuple[str, int, str, str, int, int, str] | None = None
    latency_ms = energy_j = flops_value = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        run = PERFORMANCE_RUN_RE.match(line)
        if run:
            current = (
                run["model"],
                int(run["block"]),
                run["layer"],
                run["action"],
                int(run["batch"]),
                int(run["length"]),
                run["config"],
            )
            latency_ms = energy_j = flops_value = None
            continue
        latency = PERF_LATENCY_RE.match(line)
        if latency and current is not None:
            latency_ms = float(latency["value"])
            continue
        energy = PERF_ENERGY_RE.match(line)
        if energy and current is not None:
            energy_j = float(energy["value"])
            continue
        flops = FLOPS_RE.match(line)
        if flops and current is not None:
            flops_value = float(flops["flops"])
            continue
        if line.startswith("=============================") and current is not None:
            if None in (latency_ms, energy_j, flops_value):
                raise ValueError(f"incomplete performance entry for {current} at {path}:{line_number}")
            if current in result:
                raise ValueError(f"duplicate performance entry for {current} at {path}:{line_number}")
            result[current] = PerformanceMetric(latency_ms, energy_j, flops_value)
            current = None
    return result


def combine(
    rows: list[SourceRow],
    performance_metrics: dict[
        tuple[str, int, str, str, int, int, str], PerformanceMetric
    ],
) -> list[CombinedRow]:
    grouped: dict[tuple[str, int, str, int, int, str, str], dict[str, SourceRow]] = defaultdict(dict)
    for row in rows:
        if row.failed or row.exit_code != 0 or row.succeeded == 0:
            continue
        key = (row.model, row.block, row.action, row.batch, row.length, row.config, row.arch)
        if row.layer in grouped[key]:
            raise ValueError(f"duplicate {row.layer} row for {key}")
        grouped[key][row.layer] = row

    combined: list[CombinedRow] = []
    for key, layers in grouped.items():
        if set(layers) != {"attention", "ffn"}:
            raise ValueError(f"missing attention or ffn row for {key}: {set(layers)}")
        model, block, action, batch, length, config, arch = key
        attention = layers["attention"]
        ffn = layers["ffn"]
        ffn_scale = ffn_expert_group_scale(action, config)
        attention_key = (model, block, "attention", action, batch, length, config)
        ffn_key = (model, block, "ffn", action, batch, length, config)
        if attention_key not in performance_metrics or ffn_key not in performance_metrics:
            continue
        combined.append(
            CombinedRow(
                model=model,
                block=block,
                action=action,
                batch=batch,
                length=length,
                config=config,
                arch=arch,
                latency_cycles=attention.latency_cycles
                + ffn.latency_cycles * ffn_scale,
                energy_uJ=attention.energy_uJ + ffn.energy_uJ * ffn_scale,
                flops=performance_metrics[attention_key].flops
                + performance_metrics[ffn_key].flops,
            )
        )

    baseline_workloads = {
        (row.model, row.block, row.action, row.batch, row.length, row.config)
        for row in combined
    }
    for model, block, action, batch, length, config in sorted(baseline_workloads):
        attention_key = (model, block, "attention", action, batch, length, config)
        ffn_key = (model, block, "ffn", action, batch, length, config)
        if attention_key not in performance_metrics or ffn_key not in performance_metrics:
            continue
        attention = performance_metrics[attention_key]
        ffn = performance_metrics[ffn_key]
        combined.append(
            CombinedRow(
                model=model,
                block=block,
                action=action,
                batch=batch,
                length=length,
                config=config,
                arch="ours",
                # ModelPerformance reports milliseconds and joules. Convert
                # to the baseline table's cycle/uJ representation at 1 GHz.
                latency_cycles=(attention.latency_ms + ffn.latency_ms) * 1e6,
                energy_uJ=(attention.energy_j + ffn.energy_j) * 1e6,
                flops=attention.flops + ffn.flops,
            )
        )
    return sorted(combined, key=lambda row: (row.action, row.config, row.length, row.arch))


def write_outputs(
    rows: list[CombinedRow], output_dir: Path, source: Path, performance_log: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "block", "action", "config", "batch", "length", "arch",
        "total_latency_cycles", "total_energy_uJ", "total_flops",
        "throughput_GFLOPS", "energy_eff_GFLOPS_per_J",
    ]
    with (output_dir / "qwen3_moe_combined.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
                    "arch": row.arch,
                    "total_latency_cycles": row.latency_cycles,
                    "total_energy_uJ": row.energy_uJ,
                    "total_flops": row.flops,
                    "throughput_GFLOPS": row.throughput_gflops,
                    "energy_eff_GFLOPS_per_J": row.energy_eff_gflops_per_j,
                }
            )

    architecture_order = ("ours", "gemmini_os", "gemmini_ws", "lego")
    architectures = [arch for arch in architecture_order if any(row.arch == arch for row in rows)]
    lines = [
        "# Qwen3-MoE combined baseline",
        "",
        f"Latency and energy source: `{source}`",
        "",
        f"FLOPs source: `{performance_log}`",
        "",
        "Attention and FFN are summed before deriving metrics. `ours` uses latency, energy, and FLOPs from `performance.log`. Baseline architectures use cycles/energy from `summary.csv` and the same FLOPs from `performance.log`. To approximate the full Qwen3-MoE FFN workload, baseline FFN latency and energy are scaled by 16 for edge prefill and 2 for server prefill; decode needs no correction. Throughput is reported in GFLOPS and energy efficiency in GFLOPS/J.",
        "",
    ]
    for action, config in (("prefill", "edge"), ("prefill", "server"), ("decode", "edge"), ("decode", "server")):
        selected = [row for row in rows if row.action == action and row.config == config]
        if not selected:
            continue
        batch = selected[0].batch
        lines.extend(
            [
                f"## {action.capitalize()} / {config} (batch {batch})",
                "",
                "| Length | "
                + " | ".join(f"{arch} throughput (GFLOPS) | {arch} efficiency (GFLOPS/J)" for arch in architectures)
                + " |",
                "|---:|" + "---:|---:|" * len(architectures),
            ]
        )
        by_length = {row.length: row for row in selected}
        for length in sorted(by_length):
            values = []
            for arch in architectures:
                row = next(row for row in selected if row.length == length and row.arch == arch)
                values.extend([f"{row.throughput_gflops:.4f}", f"{row.energy_eff_gflops_per_j:.4f}"])
            lines.append(f"| {length} | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- `gemmini_os`, `gemmini_ws`, and `lego` are kept as separate architecture columns.",
            "- `ours` converts logged milliseconds to cycles at the framework's 1 GHz assumption and logged joules to uJ before applying the shared formulas.",
            "- Baseline FFN prefill scaling approximates the 16 expert groups that the baseline pass modeled as 1 group on edge and 8 groups on server. FLOPs are not scaled again because `performance.log` already contains the full workload.",
            "- Attention is not scaled; its internal head dimensions require per-operation correction rather than one global factor.",
            "- Edge/server rows use the batches encoded by the source files: batch 1 for edge and batch 8 for server.",
            "- Length 2048 is omitted because the selected performance log has no matching FLOPs records.",
            "- The summary contains successful rows only; failed rows are excluded and would cause an error if either attention or FFN were missing.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("baseline_test_qwen3_moe/summary.csv"))
    parser.add_argument("--performance-log", type=Path, default=Path("test/performance.log"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/qwen3-moe-baseline"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    performance_metrics = read_performance_metrics(args.performance_log)
    rows = combine(read_source(args.summary), performance_metrics)
    if not rows:
        raise SystemExit(f"no successful attention+ffn pairs found in {args.summary}")
    write_outputs(rows, args.output_dir, args.summary, args.performance_log)
    print(f"Wrote {len(rows)} combined rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
