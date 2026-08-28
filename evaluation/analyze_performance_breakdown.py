#!/usr/bin/env python3
"""Summarize execution time and EDP from an AccelGen performance log."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


RUN_RE = re.compile(
    r"^Running: model=(?P<model>\S+) action=(?P<action>\S+) "
    r"block=(?P<block>\d+) layer=(?P<layer>attention|ffn) "
    r"batch=(?P<batch>\d+) length=(?P<length>\d+) config=(?P<config>\S+)"
    r"(?: size=(?P<size>\d+))?$"
)
EXEC_RE = re.compile(
    r"^Execution time \(ms\)\s*:\s*(?P<ours>[-+0-9.eE]+)\s+"
    r"(?P<original>[-+0-9.eE]+)$"
)
LATENCY_RE = re.compile(r"^Latency:\s*(?P<value>[-+0-9.eE]+)$")
ENERGY_RE = re.compile(r"^Energy:\s*(?P<value>[-+0-9.eE]+)$")


@dataclass(frozen=True)
class Record:
    model: str
    action: str
    block: int
    layer: str
    batch: int
    length: int
    config: str
    size: int
    ours_ms: float
    original_ms: float
    latency: float
    energy: float


def parse_log(path: Path) -> list[Record]:
    records: list[Record] = []
    current: dict[str, str] | None = None
    ours_ms = original_ms = latency = energy = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        run = RUN_RE.match(line)
        if run:
            if current is not None:
                raise ValueError(f"missing metrics before new run at {path}:{line_number}")
            current = run.groupdict()
            if current["size"] is None:
                current["size"] = "1"
            ours_ms = original_ms = latency = energy = None
            continue
        if current is None or not line.strip():
            continue
        match = EXEC_RE.match(line)
        if match:
            ours_ms = float(match["ours"])
            original_ms = float(match["original"])
            continue
        match = LATENCY_RE.match(line)
        if match:
            latency = float(match["value"])
            continue
        match = ENERGY_RE.match(line)
        if match:
            energy = float(match["value"])
            continue
        if line.startswith("============================="):
            if None in (ours_ms, original_ms, latency, energy):
                raise ValueError(f"incomplete metrics at {path}:{line_number}")
            records.append(
                Record(
                    model=current["model"],
                    action=current["action"],
                    block=int(current["block"]),
                    layer=current["layer"],
                    batch=int(current["batch"]),
                    length=int(current["length"]),
                    config=current["config"],
                    size=int(current["size"]),
                    ours_ms=ours_ms,
                    original_ms=original_ms,
                    latency=latency,
                    energy=energy,
                )
            )
            current = None
    if current is not None:
        raise ValueError(f"unterminated run at end of {path}")
    return records


def aggregate(records: list[Record]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str, int, int], dict[str, Record]] = {}
    for record in records:
        key = (
            record.model,
            record.action,
            record.block,
            record.config,
            record.batch,
            record.length,
        )
        by_layer = grouped.setdefault(key + (record.size,), {})
        if record.layer in by_layer:
            raise ValueError(f"duplicate {record.layer} record for {key}, size={record.size}")
        by_layer[record.layer] = record

    rows: list[dict[str, object]] = []
    for (model, action, block, config, batch, length, size), by_layer in sorted(grouped.items()):
        missing = {"attention", "ffn"} - set(by_layer)
        if missing:
            raise ValueError(
                f"missing layers {sorted(missing)} for model={model}, length={length}, size={size}"
            )
        attention = by_layer["attention"]
        ffn = by_layer["ffn"]
        total_latency = attention.latency + ffn.latency
        total_energy = attention.energy + ffn.energy
        rows.append(
            {
                "model": model,
                "action": action,
                "block": block,
                "config": config,
                "batch": batch,
                "length": length,
                "size": size,
                "execution_ours_ms": attention.ours_ms + ffn.ours_ms,
                "execution_original_ms": attention.original_ms + ffn.original_ms,
                "attention_latency": attention.latency,
                "ffn_latency": ffn.latency,
                "total_latency": total_latency,
                "attention_energy": attention.energy,
                "ffn_energy": ffn.energy,
                "total_energy": total_energy,
                "edp": total_latency * total_energy,
            }
        )
    return rows


def format_number(value: object) -> str:
    return f"{float(value):.6g}"


def write_outputs(rows: list[dict[str, object]], output_dir: Path, source: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "performance_breakdown_summary.csv"
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    model_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        model_groups[str(row["model"])].append(row)
    lengths = sorted({int(row["length"]) for row in rows})
    lines = [
        "# Performance breakdown summary",
        "",
        f"Source: `{source}`",
        "",
        "Execution time is the sum of attention and FFN `Execution time (ms)` values. The `original` column is the logged `timeNaive` estimate, not a separately measured hardware baseline. EDP is computed as `(attention latency + FFN latency) * (attention energy + FFN energy)` using the logged Model Performance values.",
        "",
        "The active log contains only the lengths listed below. Missing lengths cannot be reconstructed from this file.",
        "",
        f"Lengths: {', '.join(map(str, lengths))}",
        "",
    ]
    for model, model_rows in model_groups.items():
        first = model_rows[0]
        lines.extend(
            [
                f"## {model}",
                "",
                f"Action: `{first['action']}`, config: `{first['config']}`, batch: `{first['batch']}`, block: `{first['block']}`",
                "",
                "| Length | Size | Ours execution (ms) | Naive estimate (ms) | Total latency (logged) | Total energy (logged) | EDP (logged units) |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(model_rows, key=lambda item: (int(item["length"]), int(item["size"]))):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["length"]),
                        str(row["size"]),
                        format_number(row["execution_ours_ms"]),
                        format_number(row["execution_original_ms"]),
                        format_number(row["total_latency"]),
                        format_number(row["total_energy"]),
                        format_number(row["edp"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    (output_dir / "performance_breakdown_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("test/performance_breakdown.log"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results/performance-breakdown"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = aggregate(parse_log(args.log))
    if not rows:
        raise SystemExit(f"no complete runs found in {args.log}")
    write_outputs(rows, args.output_dir, args.log)
    print(f"Wrote {len(rows)} combined attention+FFN rows to {args.output_dir}")
    print(f"Models: {', '.join(sorted({str(row['model']) for row in rows}))}")
    print(f"Lengths: {', '.join(map(str, sorted({int(row['length']) for row in rows})))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
