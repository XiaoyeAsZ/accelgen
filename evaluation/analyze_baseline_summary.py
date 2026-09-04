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
BASELINE_DRAM_WORD_BITS = 16


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
    linear_cycles: float
    energy_uJ: float
    dram_accesses: float
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
                    linear_cycles=float(row["linear_cycles"]),
                    energy_uJ=float(row["energy_uJ"]),
                    dram_accesses=float(row["dram_access"]),
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
    *,
    approximate_ffn_batch: bool = True,
    flops_source: str = "performance",
    include_ours: bool = True,
    ffn_latency_scale: float = 1.0,
    attention_memory_latency_scale: float = 1.0,
    baseline_dram_energy_pj_per_bit: float | None = None,
    baseline_source_dram_energy_pj_per_bit: float = 8.0,
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
        attention_memory_cycles = attention.latency_cycles - attention.linear_cycles
        if attention_memory_cycles < 0:
            raise ValueError(f"attention linear cycles exceed total cycles for {key}")
        ffn_scale = ffn_expert_group_scale(action, config) if approximate_ffn_batch else 1
        attention_key = (model, block, "attention", action, batch, length, config)
        ffn_key = (model, block, "ffn", action, batch, length, config)
        if flops_source == "performance" and (
            attention_key not in performance_metrics or ffn_key not in performance_metrics
        ):
            continue
        if flops_source == "performance":
            flops = performance_metrics[attention_key].flops + performance_metrics[ffn_key].flops
        elif flops_source == "summary":
            flops = attention.flops + ffn.flops
        else:
            raise ValueError(f"unsupported FLOPs source: {flops_source}")
        energy_uJ = attention.energy_uJ + ffn.energy_uJ * ffn_scale
        if baseline_dram_energy_pj_per_bit is not None:
            energy_delta = (
                baseline_dram_energy_pj_per_bit
                - baseline_source_dram_energy_pj_per_bit
            )
            scaled_dram_accesses = attention.dram_accesses + ffn.dram_accesses * ffn_scale
            energy_uJ += (
                scaled_dram_accesses
                * energy_delta
                * BASELINE_DRAM_WORD_BITS
                * 1e-6
            )
        combined.append(
            CombinedRow(
                model=model,
                block=block,
                action=action,
                batch=batch,
                length=length,
                config=config,
                arch=arch,
                latency_cycles=attention.linear_cycles
                + attention_memory_cycles * attention_memory_latency_scale
                + ffn.latency_cycles * ffn_scale * ffn_latency_scale,
                energy_uJ=energy_uJ,
                flops=flops,
            )
        )

    if not include_ours:
        return sorted(combined, key=lambda row: (row.action, row.config, row.length, row.arch))

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
    rows: list[CombinedRow],
    output_dir: Path,
    source: Path,
    performance_log: Path,
    *,
    approximate_ffn_batch: bool = True,
    flops_source: str = "performance",
    include_ours: bool = True,
    ffn_latency_scale: float = 1.0,
    attention_memory_latency_scale: float = 1.0,
    baseline_dram_energy_pj_per_bit: float | None = None,
    baseline_source_dram_energy_pj_per_bit: float = 8.0,
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
    flops_description = (
        f"FLOPs source: `{performance_log}`"
        if flops_source == "performance"
        else f"FLOPs source: `{source}`"
    )
    methodology = (
        "Attention and FFN are summed before deriving metrics. Baseline architectures "
        "use cycles and energy from the summary. "
    )
    if flops_source == "performance":
        methodology += "Baseline FLOPs use matching records from `performance.log`. "
    else:
        methodology += "Baseline FLOPs use the summary's per-layer FLOP counts. "
    if approximate_ffn_batch:
        methodology += (
            "Baseline FFN latency and energy are scaled by 16 for edge prefill and 2 "
            "for server prefill to approximate the full expert-group workload. "
        )
    else:
        methodology += "No expert-group FFN scaling is applied because the source preserves batch dimensions. "
    if ffn_latency_scale != 1.0:
        methodology += (
            f"Baseline FFN latency is additionally multiplied by {ffn_latency_scale:g} "
            "to align aggregate bandwidth assumptions; energy and FLOPs are unchanged. "
        )
    if attention_memory_latency_scale != 1.0:
        methodology += (
            "Baseline attention elementwise and data-movement latency is multiplied by "
            f"{attention_memory_latency_scale:g}; attention linear latency, energy, and "
            "FLOPs are unchanged. "
        )
    if baseline_dram_energy_pj_per_bit is not None:
        methodology += (
            "Baseline DRAM energy is normalized from "
            f"{baseline_source_dram_energy_pj_per_bit:g} to "
            f"{baseline_dram_energy_pj_per_bit:g} pJ/bit using recorded 16-bit "
            "scalar DRAM accesses. "
        )
    if include_ours:
        methodology += "`ours` uses latency, energy, and FLOPs from `performance.log`. "
    methodology += "Throughput is reported in GFLOPS and energy efficiency in GFLOPS/J."

    lines = [
        "# Qwen3-MoE combined baseline",
        "",
        f"Latency and energy source: `{source}`",
        "",
        flops_description,
        "",
        methodology,
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
    lines.extend(["## Notes", ""])
    lines.append("- `gemmini_os`, `gemmini_ws`, and `lego` are kept as separate architecture columns.")
    if include_ours:
        lines.append("- `ours` converts logged milliseconds to cycles at the framework's 1 GHz assumption and logged joules to uJ before applying the shared formulas.")
    if approximate_ffn_batch:
        lines.extend(
            [
                "- Baseline FFN prefill scaling approximates the 16 expert groups that the baseline pass modeled as 1 group on edge and 8 groups on server. FLOPs are not scaled again because `performance.log` already contains the full workload.",
                "- Attention is not scaled; its internal head dimensions require per-operation correction rather than one global factor.",
            ]
        )
    else:
        lines.append("- No expert-group workload scaling is applied to the preserved-batch source.")
    if ffn_latency_scale != 1.0:
        lines.append(
            f"- FFN latency uses a `{ffn_latency_scale:g}x` bandwidth correction. "
            "The latency correction itself does not scale energy; the DRAM normalization "
            "below is applied separately."
        )
    if attention_memory_latency_scale != 1.0:
        lines.append(
            "- Attention memory latency is defined as `total_cycles - linear_cycles`, "
            f"and uses a `{attention_memory_latency_scale:g}x` correction. This includes "
            "softmax elementwise work and data movement."
        )
    if baseline_dram_energy_pj_per_bit is not None:
        lines.append(
            "- DRAM energy correction: "
            f"`accesses * ({baseline_dram_energy_pj_per_bit:g} - "
            f"{baseline_source_dram_energy_pj_per_bit:g}) pJ/bit * "
            f"{BASELINE_DRAM_WORD_BITS} bits`, converted to uJ."
        )
    lines.extend(
        [
            "- Edge/server rows use the batches encoded by the source files: batch 1 for edge and batch 8 for server.",
            f"- Included lengths: {', '.join(map(str, sorted({row.length for row in rows})))}.",
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
    parser.add_argument(
        "--ffn-batch-mode",
        choices=("approximate", "preserved"),
        default="approximate",
        help="approximate old forced-batch runs or use already-preserved batch dimensions",
    )
    parser.add_argument(
        "--flops-source", choices=("performance", "summary"), default="performance"
    )
    parser.add_argument(
        "--include-ours", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ffn-latency-scale",
        type=float,
        default=1.0,
        help="multiply baseline FFN cycles without changing energy or FLOPs",
    )
    parser.add_argument(
        "--attention-memory-latency-scale",
        type=float,
        default=1.0,
        help="multiply attention non-linear/data-movement cycles without changing energy",
    )
    parser.add_argument(
        "--baseline-dram-energy-pj-per-bit",
        type=float,
        help="normalize baseline DRAM energy to this target pJ/bit",
    )
    parser.add_argument(
        "--baseline-source-dram-energy-pj-per-bit",
        type=float,
        default=8.0,
        help="DRAM pJ/bit already included in the baseline summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ffn_latency_scale <= 0 or args.attention_memory_latency_scale <= 0:
        raise SystemExit("latency scales must be positive")
    if args.baseline_dram_energy_pj_per_bit is not None and args.baseline_dram_energy_pj_per_bit <= 0:
        raise SystemExit("--baseline-dram-energy-pj-per-bit must be positive")
    if args.baseline_source_dram_energy_pj_per_bit <= 0:
        raise SystemExit("--baseline-source-dram-energy-pj-per-bit must be positive")
    performance_metrics = read_performance_metrics(args.performance_log)
    approximate_ffn_batch = args.ffn_batch_mode == "approximate"
    rows = combine(
        read_source(args.summary),
        performance_metrics,
        approximate_ffn_batch=approximate_ffn_batch,
        flops_source=args.flops_source,
        include_ours=args.include_ours,
        ffn_latency_scale=args.ffn_latency_scale,
        attention_memory_latency_scale=args.attention_memory_latency_scale,
        baseline_dram_energy_pj_per_bit=args.baseline_dram_energy_pj_per_bit,
        baseline_source_dram_energy_pj_per_bit=args.baseline_source_dram_energy_pj_per_bit,
    )
    if not rows:
        raise SystemExit(f"no successful attention+ffn pairs found in {args.summary}")
    write_outputs(
        rows,
        args.output_dir,
        args.summary,
        args.performance_log,
        approximate_ffn_batch=approximate_ffn_batch,
        flops_source=args.flops_source,
        include_ours=args.include_ours,
        ffn_latency_scale=args.ffn_latency_scale,
        attention_memory_latency_scale=args.attention_memory_latency_scale,
        baseline_dram_energy_pj_per_bit=args.baseline_dram_energy_pj_per_bit,
        baseline_source_dram_energy_pj_per_bit=args.baseline_source_dram_energy_pj_per_bit,
    )
    print(f"Wrote {len(rows)} combined rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
