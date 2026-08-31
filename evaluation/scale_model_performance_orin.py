#!/usr/bin/env python3
"""Scale edge model-performance results using Orin-config anchor measurements."""

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
METRIC_RES = {
    "latency_ms": re.compile(r"^Latency:\s*(?P<value>[-+0-9.eE]+)$"),
    "energy_j": re.compile(r"^Energy:\s*(?P<value>[-+0-9.eE]+)$"),
    "flops": re.compile(r"^Flops:\s*(?P<value>[-+0-9.eE]+)$"),
}


@dataclass(frozen=True)
class OrinMetric:
    latency_ms: float
    energy_j: float
    flops: float


def read_expected_flops(path: Path) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    current: tuple[str, str, str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        run = RUN_RE.match(line)
        if run:
            current = (run["model"], run["action"], run["layer"])
            continue
        match = METRIC_RES["flops"].match(line)
        if match and current is not None:
            result.setdefault(current, float(match["value"]))
            current = None
    return result


def read_orin_metrics(
    path: Path, expected_flops: dict[tuple[str, str, str], float]
) -> dict[tuple[str, str, str], OrinMetric]:
    """Read valid config=orin records, ignoring stale interleaved records."""
    result: dict[tuple[str, str, str], OrinMetric] = {}
    current: tuple[str, str, str] | None = None
    values: dict[str, float] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        run = RUN_RE.match(line)
        if run:
            if run["config"] != "orin":
                current = None
                continue
            current = (run["model"], run["action"], run["layer"])
            values = {}
            continue
        if current is None:
            continue
        for name, pattern in METRIC_RES.items():
            match = pattern.match(line)
            if match:
                values[name] = float(match["value"])
                break
        if line.startswith("============================="):
            if set(values) != set(METRIC_RES):
                raise ValueError(f"incomplete Orin record for {current} at {path}:{line_number}")
            key = current
            metric = OrinMetric(values["latency_ms"], values["energy_j"], values["flops"])
            expected = expected_flops.get(key)
            if expected is not None and metric.flops == expected:
                result.setdefault(key, metric)
            current = None
    if current is not None:
        raise ValueError(f"unterminated Orin record for {current}")
    return result


def read_edge_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["config"] == "edge"]


def scale_rows(
    rows: list[dict[str, str]],
    metrics: dict[tuple[str, str, str], OrinMetric],
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], tuple[float, float]]]:
    anchors: dict[tuple[str, str, str], tuple[float, float]] = {}
    for anchor_model in ("llama3-8b", "qwen3-moe"):
        for action in ("prefill", "decode"):
            for layer in ("attention", "ffn"):
                key = (anchor_model, action, layer)
                if key not in metrics:
                    raise ValueError(f"missing Orin anchor {key}")
                edge = next(
                    row for row in rows
                    if row["model"] == anchor_model
                    and row["action"] == action
                    and row["length"] == "128"
                )
                anchors[key] = (
                    metrics[key].latency_ms / float(edge[f"{layer}_latency_ms"]),
                    metrics[key].energy_j / float(edge[f"{layer}_energy_j"]),
                )

    output: list[dict[str, object]] = []
    for row in rows:
        anchor_model = "qwen3-moe" if row["model"] == "qwen3-moe" else "llama3-8b"
        action = row["action"]
        attention_latency_scale, attention_energy_scale = anchors[(anchor_model, action, "attention")]
        ffn_latency_scale, ffn_energy_scale = anchors[(anchor_model, action, "ffn")]
        attention_latency = float(row["attention_latency_ms"]) * attention_latency_scale
        ffn_latency = float(row["ffn_latency_ms"]) * ffn_latency_scale
        attention_energy = float(row["attention_energy_j"]) * attention_energy_scale
        ffn_energy = float(row["ffn_energy_j"]) * ffn_energy_scale
        latency = attention_latency + ffn_latency
        energy = attention_energy + ffn_energy
        flops = float(row["total_flops"])
        output.append(
            {
                "hardware": "Orin-scaled ours",
                "model": row["model"],
                "block": row["block"],
                "action": row["action"],
                "config": "orin",
                "batch": row["batch"],
                "length": row["length"],
                "attention_latency_ms": attention_latency,
                "ffn_latency_ms": ffn_latency,
                "total_latency_ms": latency,
                "attention_energy_j": attention_energy,
                "ffn_energy_j": ffn_energy,
                "total_energy_j": energy,
                "total_flops": flops,
                "throughput_GFLOPS": flops / (latency * 1e6),
                "energy_efficiency_GFLOPS_per_J": flops / (energy * 1e9),
                "attention_latency_scale": attention_latency_scale,
                "ffn_latency_scale": ffn_latency_scale,
                "attention_energy_scale": attention_energy_scale,
                "ffn_energy_scale": ffn_energy_scale,
            }
        )
    return output, anchors


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, object]], anchors: dict[tuple[str, str, str], tuple[float, float]], path: Path, source: Path, orin_log: Path) -> None:
    lines = [
        "# Orin-scaled edge model performance",
        "",
        f"Source edge results: `{source}`",
        "",
        f"Orin anchor measurements: `{orin_log}` (config=orin only)",
        "",
        "The edge results are scaled to an Orin configuration using per-layer ratios at length 128. "
        "Llama3-8B ratios are reused for Llama3-8B, Qwen3-8B, and Gemma-7B; Qwen3-MoE ratios are reused for Qwen3-MoE. FLOPs are unchanged.",
        "",
        "## Anchor ratios",
        "",
        "| Anchor | Action | Layer | Latency scale | Energy scale |",
        "|:--|:--|:--|--:|--:|",
    ]
    for anchor_model in ("llama3-8b", "qwen3-moe"):
        for action in ("prefill", "decode"):
            for layer in ("attention", "ffn"):
                latency_scale, energy_scale = anchors[(anchor_model, action, layer)]
                lines.append(f"| {anchor_model} | {action} | {layer} | {latency_scale:.8f} | {energy_scale:.8f} |")
    lines.append("")
    for model in sorted({str(row["model"]) for row in rows}):
        lines.extend(
            [
                f"## {model}",
                "",
                "| Action | Config | Batch | Length | Latency (ms) | Energy (J) | FLOPs (G) | Throughput (GFLOPS) | Efficiency (GFLOPS/J) |",
                "|:--|:--|--:|--:|--:|--:|--:|--:|--:|",
            ]
        )
        for row in (item for item in rows if item["model"] == model):
            lines.append(
                f"| {row['action']} | {row['config']} | {row['batch']} | {row['length']} | "
                f"{float(row['total_latency_ms']):.6f} | {float(row['total_energy_j']):.6f} | "
                f"{float(row['total_flops']) / 1e9:.6f} | {float(row['throughput_GFLOPS']):.4f} | "
                f"{float(row['energy_efficiency_GFLOPS_per_J']):.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Formulas",
            "",
            "```text",
            "scaled_attention = edge_attention * latency_scale",
            "scaled_ffn = edge_ffn * latency_scale",
            "scaled_energy = edge_energy * energy_scale",
            "throughput_GFLOPS = total_FLOPs / (scaled_latency_ms * 1e6)",
            "energy_efficiency_GFLOPS_per_J = total_FLOPs / (scaled_energy_J * 1e9)",
            "```",
            "",
            "The current `moe_perf.log` contains stale interleaved duplicate records; records with mismatched model/layer FLOPs were ignored.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-csv", type=Path, default=Path("evaluation/results/model-performance/model_performance.csv"))
    parser.add_argument("--orin-log", type=Path, default=Path("test/moe_perf.log"))
    parser.add_argument("--performance-log", type=Path, default=Path("test/performance.log"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/model-performance-orin"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    edge_rows = read_edge_rows(args.edge_csv)
    metrics = read_orin_metrics(args.orin_log, read_expected_flops(args.performance_log))
    rows, anchors = scale_rows(edge_rows, metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "model_performance_orin_scaled.csv")
    write_report(rows, anchors, args.output_dir / "README.md", args.edge_csv, args.orin_log)
    print(f"Wrote {len(rows)} Orin-scaled edge rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
