#!/usr/bin/env python3
"""Create SRAM/MAC/mux area tables and a paper-ready breakdown figure."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path


SRAM_BANK_AREA_UM2 = 192291.0
BF16_MAC_AREA_UM2 = 3243.1
MUX_2TO1_AREA_UM2 = 13.7894
MUX_FANIN_RE = re.compile(r'"dap\.mux"\(\) <\{fanin = (\d+) : i32\}')
FILE_RE = re.compile(
    r"^(?P<model>.+)-block(?P<block>\d+)-(?P<layer>attention|ffn)-"
    r"(?P<action>prefill|decode)-b(?P<batch>\d+)s(?P<length>\d+)-"
    r"dap-(?P<config>edge|server)\.mlir$"
)


@dataclass
class AreaRow:
    model: str
    config: str
    batch: int
    sram_banks: int
    bf16_mulf: int
    bf16_addf: int
    bf16_macs: int
    mux_ops: int
    mux_2to1: int
    sram_mm2: float
    mac_mm2: float
    mux_mm2: float

    @property
    def total_mm2(self) -> float:
        return self.sram_mm2 + self.mac_mm2 + self.mux_mm2

    def percentage(self, value: float) -> float:
        return 100 * value / self.total_mm2


def parse_dap(path: Path, metadata: re.Match[str]) -> AreaRow:
    sram_banks = bf16_mulf = bf16_addf = mux_ops = mux_2to1 = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"dap.sram"' in line:
                sram_banks += 1
            elif '"dap.mulf"' in line and "-> (bf16, bf16, bf16)" in line:
                bf16_mulf += 1
            elif '"dap.add"' in line and "-> (bf16, bf16, bf16)" in line:
                bf16_addf += 1
            elif '"dap.mux"' in line:
                match = MUX_FANIN_RE.search(line)
                if not match:
                    raise ValueError(f"cannot parse mux fan-in in {path}: {line.strip()}")
                fanin = int(match.group(1))
                mux_ops += 1
                mux_2to1 += fanin.bit_length() - 1
    bf16_macs = max(bf16_mulf, bf16_addf)
    return AreaRow(
        model=metadata["model"],
        config=metadata["config"],
        batch=int(metadata["batch"]),
        sram_banks=sram_banks,
        bf16_mulf=bf16_mulf,
        bf16_addf=bf16_addf,
        bf16_macs=bf16_macs,
        mux_ops=mux_ops,
        mux_2to1=mux_2to1,
        sram_mm2=sram_banks * SRAM_BANK_AREA_UM2 / 1e6,
        mac_mm2=bf16_macs * BF16_MAC_AREA_UM2 / 1e6,
        mux_mm2=mux_2to1 * MUX_2TO1_AREA_UM2 / 1e6,
    )


def discover(args: argparse.Namespace) -> list[AreaRow]:
    rows: list[AreaRow] = []
    for path in args.dap_dir.glob("*.mlir"):
        metadata = FILE_RE.match(path.name)
        if not metadata:
            continue
        if (
            int(metadata["block"]) != args.block
            or metadata["layer"] != args.layer
            or metadata["action"] != args.action
            or int(metadata["length"]) != args.length
        ):
            continue
        rows.append(parse_dap(path, metadata))
    config_order = {"edge": 0, "server": 1}
    model_order = {model: index for index, model in enumerate(args.models)}
    rows = [row for row in rows if row.model in model_order]
    rows.sort(key=lambda row: (model_order[row.model], config_order[row.config]))
    expected = {(model, config) for model in args.models for config in config_order}
    actual = {(row.model, row.config) for row in rows}
    if expected != actual:
        missing = sorted(expected - actual)
        raise FileNotFoundError(f"missing DAP results: {missing}")
    return rows


def write_tables(rows: list[AreaRow], output_dir: Path) -> None:
    fields = [
        "model", "config", "batch", "sram_banks", "bf16_macs", "mux_ops",
        "mux_2to1_equivalents", "sram_area_mm2", "mac_area_mm2", "mux_area_mm2",
        "total_area_mm2", "sram_percent", "mac_percent", "mux_percent",
    ]
    with (output_dir / "area_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.model,
                    "config": row.config,
                    "batch": row.batch,
                    "sram_banks": row.sram_banks,
                    "bf16_macs": row.bf16_macs,
                    "mux_ops": row.mux_ops,
                    "mux_2to1_equivalents": row.mux_2to1,
                    "sram_area_mm2": row.sram_mm2,
                    "mac_area_mm2": row.mac_mm2,
                    "mux_area_mm2": row.mux_mm2,
                    "total_area_mm2": row.total_mm2,
                    "sram_percent": row.percentage(row.sram_mm2),
                    "mac_percent": row.percentage(row.mac_mm2),
                    "mux_percent": row.percentage(row.mux_mm2),
                }
            )

    lines = [
        "# DAP area breakdown",
        "",
        "Only the three characterized primitives are included: SRAM banks, paired BF16 MACs, and 2:1 mux equivalents.",
        "",
        "| Model | Config | SRAM (mm^2) | MAC (mm^2) | Mux (mm^2) | Total (mm^2) | SRAM (%) | MAC (%) | Mux (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.model} | {row.config} | {row.sram_mm2:.6f} | "
            f"{row.mac_mm2:.6f} | {row.mux_mm2:.6f} | {row.total_mm2:.6f} | "
            f"{row.percentage(row.sram_mm2):.3f} | "
            f"{row.percentage(row.mac_mm2):.3f} | "
            f"{row.percentage(row.mux_mm2):.3f} |"
        )
    lines.extend(
        [
            "",
            "Area constants:",
            "",
            f"- SRAM bank: `{SRAM_BANK_AREA_UM2:g} um^2`",
            f"- BF16 MAC: `{BF16_MAC_AREA_UM2:g} um^2`",
            f"- 2:1 mux: `{MUX_2TO1_AREA_UM2:g} um^2`",
            "",
            "![Area breakdown](area_breakdown.svg)",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_svg(rows: list[AreaRow], output_path: Path) -> None:
    colors = {"sram": "#2A9D8F", "mac": "#E76F51", "mux": "#264653"}
    width, height = 1500, 620
    label_x = 150
    absolute_x, absolute_width = 190, 570
    normalized_x, normalized_width = 830, 420
    mux_x, mux_width = 1320, 145
    top, row_height, bar_height = 105, 59, 27
    max_total = max(row.total_mm2 for row in rows)
    max_mux = max(row.mux_mm2 for row in rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#202124;letter-spacing:0}.head{font-size:15px;font-weight:700}.label{font-size:12px;font-weight:600}.small{font-size:10px;fill:#5f6368}.value{font-size:10px;font-weight:600}.axis{stroke:#9aa0a6;stroke-width:1}.grid{stroke:#e2e5e8;stroke-width:1}.bar{stroke:#fff;stroke-width:.7}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="head" x="{absolute_x + absolute_width / 2}" y="28" text-anchor="middle">Absolute area (mm^2)</text>',
        f'<text class="head" x="{normalized_x + normalized_width / 2}" y="28" text-anchor="middle">Normalized composition</text>',
        f'<text class="head" x="{mux_x + mux_width / 2}" y="28" text-anchor="middle">Mux (mm^2)</text>',
    ]
    for tick in (0, 200, 400, 600):
        x = absolute_x + tick / 650 * absolute_width
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="48" x2="{x:.1f}" y2="{top + len(rows) * row_height}"/>')
        parts.append(f'<text class="small" x="{x:.1f}" y="44" text-anchor="middle">{tick}</text>')
    for tick in (0, 25, 50, 75, 100):
        x = normalized_x + tick / 100 * normalized_width
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="48" x2="{x:.1f}" y2="{top + len(rows) * row_height}"/>')
        parts.append(f'<text class="small" x="{x:.1f}" y="44" text-anchor="middle">{tick}%</text>')

    for index, row in enumerate(rows):
        y = top + index * row_height
        label = f"{row.model} / {row.config}"
        parts.append(f'<text class="label" x="{label_x}" y="{y + 18}" text-anchor="end">{html.escape(label)}</text>')
        cursor = absolute_x
        for key, value in (("sram", row.sram_mm2), ("mac", row.mac_mm2), ("mux", row.mux_mm2)):
            segment = value / 650 * absolute_width
            parts.append(f'<rect class="bar" x="{cursor:.2f}" y="{y}" width="{max(segment, .5):.2f}" height="{bar_height}" fill="{colors[key]}"/>')
            cursor += segment
        parts.append(f'<text class="value" x="{cursor + 7:.2f}" y="{y + 18}">{row.total_mm2:.1f}</text>')

        cursor = normalized_x
        for key, value in (("sram", row.sram_mm2), ("mac", row.mac_mm2), ("mux", row.mux_mm2)):
            percent = row.percentage(value)
            segment = percent / 100 * normalized_width
            parts.append(f'<rect class="bar" x="{cursor:.2f}" y="{y}" width="{max(segment, .5):.2f}" height="{bar_height}" fill="{colors[key]}"/>')
            if key != "mux" and segment > 40:
                parts.append(f'<text x="{cursor + segment / 2:.2f}" y="{y + 18}" text-anchor="middle" font-size="10" fill="#ffffff">{percent:.1f}%</text>')
            cursor += segment

        mux_bar = row.mux_mm2 / max_mux * mux_width
        parts.append(f'<rect class="bar" x="{mux_x}" y="{y}" width="{mux_bar:.2f}" height="{bar_height}" fill="{colors["mux"]}"/>')
        parts.append(f'<text class="value" x="{mux_x + mux_bar + 5:.2f}" y="{y + 18}">{row.mux_mm2:.3f}</text>')

    legend_y = top + len(rows) * row_height + 25
    cursor = absolute_x
    for key, name in (("sram", "SRAM"), ("mac", "BF16 MAC"), ("mux", "Mux")):
        parts.append(f'<rect x="{cursor}" y="{legend_y}" width="16" height="12" fill="{colors[key]}"/>')
        parts.append(f'<text class="small" x="{cursor + 22}" y="{legend_y + 11}">{name}</text>')
        cursor += 105
    parts.append('</svg>')
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dap-dir", type=Path, default=Path("eval/dap"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/area-breakdown"))
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--layer", default="attention", choices=("attention", "ffn"))
    parser.add_argument("--action", default="prefill", choices=("prefill", "decode"))
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument(
        "--models", nargs="+",
        default=("llama3-8b", "qwen3-8b", "gemma-7b", "qwen3-moe"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = discover(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(rows, args.output_dir)
    write_svg(rows, args.output_dir / "area_breakdown.svg")
    print(f"Wrote {len(rows)} area rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
