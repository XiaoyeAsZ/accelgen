#!/usr/bin/env python3
"""Collect aggregated Qwen3-MoE baseline logs into baseline_test_70b-style summaries."""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from datetime import datetime

from collect_llama70b_baseline import (
    parse_log,
    write_summary_csv,
)


def parse_aggregated_log(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    starts = list(re.finditer(r"(?m)^\[[^\]]+\] START benchmark/mlir/", text))
    results = []
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        chunk = text[match.start():end]
        result = parse_log_from_text(chunk)
        if result["mlir_stem"] and result["total_cycles"] > 0:
            results.append(result)
    return results


def parse_log_from_text(text):
    # parse_log is intentionally file-oriented; use a temporary in-memory-compatible
    # implementation by writing no files and extracting the same fields from a chunk.
    result = {
        "mlir_stem": "", "linear_arch": "", "ppu_arch": "", "dm_arch": "",
        "exit_code": -1, "linear_cycles": 0, "linear_energy": 0.0,
        "elem_cycles": 0, "elem_energy": 0.0, "dm_cycles": 0,
        "dm_mem_energy": 0.0, "dm_total_energy": 0.0, "dram_access": 0,
        "total_cycles": 0, "total_energy": 0.0, "total_flops": 0,
        "throughput_gflops": 0.0, "energy_eff_gflops_j": 0.0,
        "total_ops": 0, "succeeded": 0, "failed": 0,
        "linear_ops": [], "elem_ops": [], "dm_ops": [], "failed_ops": [],
    }
    m = re.search(r"START benchmark/mlir/([^\s]+\.mlir)", text)
    if m:
        result["mlir_stem"] = os.path.splitext(os.path.basename(m.group(1)))[0]
    for key, pattern in (("linear_arch", r"linear arch:\s+(\S+)"),
                         ("ppu_arch", r"nonlinear arch:\s+(\S+)"),
                         ("dm_arch", r"datamove arch:\s+(\S+)")):
        m = re.search(pattern, text)
        if m:
            result[key] = m.group(1)
    m = re.search(r"\[(?:[^\]]+)\] EXIT_CODE=(\d+)", text)
    if m:
        result["exit_code"] = int(m.group(1))
    else:
        m = re.search(r"\[(?:[^\]]+)\] EXIT\s+(\d+)", text)
        if m:
            result["exit_code"] = int(m.group(1))
    m = re.search(r"Latency:\s+(\d+)\s+cycles", text)
    if m:
        result["total_cycles"] = int(m.group(1))
    m = re.search(r"^\s{4}Energy:\s+([\d.eE+\-]+)\s*uJ", text, re.M)
    if m:
        result["total_energy"] = float(m.group(1))
    for key, pattern in (("total_flops", r"FLOPs:\s+(\d+)"),
                         ("throughput_gflops", r"Throughput:\s+([\d.eE+\-]+)\s*GFLOPS"),
                         ("energy_eff_gflops_j", r"Energy Efficiency:\s+([\d.eE+\-]+)\s*GFLOPS/J"),
                         ("dram_access", r"DRAM Access \(dm\):\s+(\d+)")):
        m = re.search(pattern, text)
        if m:
            result[key] = float(m.group(1)) if "gflops" in key else int(m.group(1))
    m = re.search(r"Linear total: Cycles=(\d+)\s+Energy=([\d.eE+\-]+)", text)
    if m:
        result["linear_cycles"], result["linear_energy"] = int(m.group(1)), float(m.group(2))
    m = re.search(r"Elementwise total: Cycles=(\d+)\s+Energy=([\d.eE+\-]+)", text)
    if m:
        result["elem_cycles"], result["elem_energy"] = int(m.group(1)), float(m.group(2))
    m = re.search(r"Datamove total: Cycles=(\d+)\s+MemEnergy=([\d.eE+\-]+)\s+uJ\s+\(TotalEnergy=([\d.eE+\-]+)", text)
    if m:
        result["dm_cycles"], result["dm_mem_energy"], result["dm_total_energy"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
    corrected = result["linear_energy"] + result["elem_energy"] + result["dm_mem_energy"]
    if corrected:
        result["total_energy"] = corrected
        if result["total_flops"]:
            result["energy_eff_gflops_j"] = result["total_flops"] / (corrected * 1e-6) / 1e9
    m = re.search(r"Operators:\s+(\d+)\s+\(linear=\d+\s+elem=\d+\s+datamove=\d+\)\s+Succeeded=(\d+)\s+Failed=(\d+)", text)
    if m:
        result["total_ops"], result["succeeded"], result["failed"] = map(int, m.groups())
    # Reuse the established operator table parser through a temporary file only when needed.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
        f.write(text)
        temp_path = f.name
    try:
        parsed = parse_log(temp_path)
        for key in ("linear_ops", "elem_ops", "dm_ops", "failed_ops"):
            result[key] = parsed[key]
    finally:
        os.unlink(temp_path)
    return result


def parse_mapper_stats(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    cycles = int(re.search(r"^Cycles:\s+(\d+)", text, re.M).group(1))
    energy = float(re.search(r"^Energy:\s+([\d.]+)\s+uJ", text, re.M).group(1))
    computes = int(re.search(r"^Computes\s*=\s*(\d+)", text, re.M).group(1))
    util = float(re.search(r"^Utilization:\s+([\d.]+)%", text, re.M).group(1))
    dram_text = text.split("=== DRAM ===")[-1]
    dram = int(re.search(r"Total scalar accesses\s*:\s*(\d+)", dram_text).group(1))
    return dict(cycles=cycles, energy=energy, computes=computes, util=util, dram=dram)


def apply_failed_op_overrides(results, override_dir):
    for result in results:
        if not result["failed_ops"]:
            continue
        stem = result["mlir_stem"]
        workload = stem.removeprefix("qwen3-moe-block0-")
        remaining = []
        for failed_op in result["failed_ops"]:
            stats_path = os.path.join(
                override_dir, workload, failed_op["name"], "timeloop-mapper.stats.txt"
            )
            if not os.path.isfile(stats_path):
                remaining.append(failed_op)
                continue
            stats = parse_mapper_stats(stats_path)
            result["elem_ops"].append({
                "ssa": failed_op["ssa"],
                "name": failed_op["name"],
                "cycles": stats["cycles"],
                "energy": stats["energy"],
                "arch": result["ppu_arch"],
                "util": str(stats["util"]),
                "dram_acc": stats["dram"],
            })
            result["elem_cycles"] += stats["cycles"]
            result["elem_energy"] += stats["energy"]
            result["total_cycles"] += stats["cycles"]
            result["total_energy"] += stats["energy"]
            result["total_flops"] += stats["computes"]
            result["succeeded"] += 1
            result["failed"] -= 1
        result["failed_ops"] = remaining
        if result["total_cycles"]:
            result["throughput_gflops"] = result["total_flops"] / result["total_cycles"]
        if result["total_energy"]:
            result["energy_eff_gflops_j"] = result["total_flops"] / (result["total_energy"] * 1e3)


def op_sort_key(op):
    match = re.search(r"_(\d+)$", op["name"])
    return (int(match.group(1)) if match else 1 << 30, op["name"], op["ssa"])


def write_detailed_csv(out_dir, results):
    path = os.path.join(out_dir, "summary_detail.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mlir_stem", "linear_arch", "op_type", "op_name", "ssa",
            "cycles", "energy_uJ", "op_arch", "util_percent",
            "dram_access", "status",
        ])
        for result in results:
            for key, op_type in (("linear_ops", "linear"),
                                 ("elem_ops", "elementwise"),
                                 ("dm_ops", "datamove")):
                for op in sorted(result[key], key=op_sort_key):
                    writer.writerow([
                        result["mlir_stem"], result["linear_arch"], op_type,
                        op["name"], op["ssa"], op["cycles"],
                        f"{op['energy']:.6f}", op.get("arch", ""),
                        op.get("util", ""), op.get("dram_acc", 0), "OK",
                    ])
            for op in result.get("failed_ops", []):
                writer.writerow([
                    result["mlir_stem"], result["linear_arch"], op["type"],
                    op["name"], op["ssa"], "FAILED", "FAILED",
                    result["ppu_arch"] if op["type"] == "elementwise" else "",
                    "", "", "FAILED",
                ])


def write_detailed_summary(out_dir, results, title):
    path = os.path.join(out_dir, "summary.txt")
    by_arch = defaultdict(list)
    for result in results:
        by_arch[result["linear_arch"]].append(result)
    width = 145
    with open(path, "w") as f:
        f.write("=" * width + "\n")
        f.write(f"  {title} Baseline Detailed Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * width + "\n\n")
        grand_cycles = grand_flops = grand_dram = grand_ok = grand_fail = 0
        grand_energy = 0.0
        for arch in sorted(by_arch):
            arch_results = sorted(by_arch[arch], key=lambda r: r["mlir_stem"])
            f.write("#" * width + "\n")
            f.write(f"# Architecture: {arch}\n")
            f.write(f"# PPU: {arch_results[0]['ppu_arch']}  DM: {arch_results[0]['dm_arch']}  Workloads: {len(arch_results)}\n")
            f.write("#" * width + "\n\n")
            arch_cycles = arch_flops = arch_dram = arch_ok = arch_fail = 0
            arch_energy = 0.0
            for result in arch_results:
                f.write(f"  Workload: {result['mlir_stem']}\n")
                f.write(f"  {'-' * (width - 2)}\n")
                f.write(
                    f"    {'Type':<14} {'SSA':<12} {'Operator':<31} {'Arch':<23} "
                    f"{'Util%':>9} {'Cycles':>14} {'Energy(uJ)':>16} {'DRAM_Acc':>16}\n"
                )
                f.write(f"    {'-' * (width - 4)}\n")
                groups = (("linear", result["linear_ops"]),
                          ("elementwise", result["elem_ops"]),
                          ("datamove", result["dm_ops"]))
                for op_type, ops in groups:
                    for op in sorted(ops, key=op_sort_key):
                        util = op.get("util", "")
                        if util != "":
                            try:
                                util = f"{float(util):.1f}%"
                            except ValueError:
                                pass
                        else:
                            util = "-"
                        f.write(
                            f"    {op_type:<14} {op['ssa']:<12} {op['name']:<31} "
                            f"{op.get('arch', ''):<23} {util:>9} {op['cycles']:>14} "
                            f"{op['energy']:>16.4f} {op.get('dram_acc', 0):>16}\n"
                        )
                for op in result.get("failed_ops", []):
                    f.write(
                        f"    {op['type']:<14} {op['ssa']:<12} {op['name']:<31} "
                        f"{result['ppu_arch']:<23} {'-':>9} {'FAILED':>14} "
                        f"{'FAILED':>16} {'-':>16}\n"
                    )
                f.write(f"    {'-' * (width - 4)}\n")
                f.write(
                    f"    Breakdown: linear={result['linear_cycles']} cycles/{result['linear_energy']:.4f} uJ  "
                    f"elementwise={result['elem_cycles']} cycles/{result['elem_energy']:.4f} uJ  "
                    f"datamove={result['dm_cycles']} cycles/{result['dm_mem_energy']:.4f} uJ\n"
                )
                f.write(
                    f"    Summary: Latency={result['total_cycles']} cycles  "
                    f"Energy={result['total_energy']:.4f} uJ  FLOPs={result['total_flops']}  "
                    f"Thpt={result['throughput_gflops']:.4f} GFLOPS  "
                    f"Eff={result['energy_eff_gflops_j']:.4f} GFLOPS/J  "
                    f"DRAM_Acc={result['dram_access']}  OK={result['succeeded']} Fail={result['failed']}\n\n"
                )
                arch_cycles += result["total_cycles"]
                arch_energy += result["total_energy"]
                arch_flops += result["total_flops"]
                arch_dram += result["dram_access"]
                arch_ok += result["succeeded"]
                arch_fail += result["failed"]
            arch_thpt = arch_flops / arch_cycles if arch_cycles else 0.0
            arch_eff = arch_flops / (arch_energy * 1e3) if arch_energy else 0.0
            f.write(f"  {'=' * (width - 2)}\n")
            f.write(
                f"  ARCH TOTAL: {arch}  Workloads={len(arch_results)}  "
                f"Latency={arch_cycles} cycles  Energy={arch_energy:.4f} uJ  "
                f"FLOPs={arch_flops}  Thpt={arch_thpt:.4f} GFLOPS  "
                f"Eff={arch_eff:.4f} GFLOPS/J  DRAM_Acc={arch_dram}  "
                f"OK={arch_ok} Fail={arch_fail}\n\n"
            )
            grand_cycles += arch_cycles
            grand_energy += arch_energy
            grand_flops += arch_flops
            grand_dram += arch_dram
            grand_ok += arch_ok
            grand_fail += arch_fail
        grand_thpt = grand_flops / grand_cycles if grand_cycles else 0.0
        grand_eff = grand_flops / (grand_energy * 1e3) if grand_energy else 0.0
        f.write("=" * width + "\n")
        f.write("GRAND TOTAL\n")
        f.write(
            f"  Workloads={len(results)}  Latency={grand_cycles} cycles  "
            f"Energy={grand_energy:.4f} uJ  FLOPs={grand_flops}  "
            f"Thpt={grand_thpt:.4f} GFLOPS  Eff={grand_eff:.4f} GFLOPS/J  "
            f"DRAM_Acc={grand_dram}  OK={grand_ok} Fail={grand_fail}\n"
        )
        f.write("=" * width + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--override-dir")
    parser.add_argument("--failed-op-dir")
    parser.add_argument("--title", default="Qwen3-MoE")
    args = parser.parse_args()
    results = []
    for path in sorted(glob.glob(os.path.join(args.in_dir, "logs", "*.log"))):
        results.extend(parse_aggregated_log(path))
    if args.override_dir:
        overrides = []
        for path in sorted(glob.glob(os.path.join(args.override_dir, "logs", "*.log"))):
            result = parse_log(path)
            if result["mlir_stem"] and result["total_cycles"] > 0:
                overrides.append(result)
        override_by_key = {
            (result["mlir_stem"], result["linear_arch"]): result
            for result in overrides
        }
        results = [
            override_by_key.get((result["mlir_stem"], result["linear_arch"]), result)
            for result in results
        ]
    if args.failed_op_dir:
        apply_failed_op_overrides(results, args.failed_op_dir)
    if not results:
        raise SystemExit(f"No completed workloads found in {args.in_dir}/logs")
    os.makedirs(args.out_dir, exist_ok=True)
    write_summary_csv(args.out_dir, results)
    write_detailed_csv(args.out_dir, results)
    write_detailed_summary(args.out_dir, results, args.title)
    print(f"Parsed {len(results)} workloads")
    print(f"Saved: {args.out_dir}/summary.csv {args.out_dir}/summary_detail.csv {args.out_dir}/summary.txt")


def append_failed_rows(out_dir, results):
    detail_path = os.path.join(out_dir, "summary_detail.csv")
    failed = []
    for result in results:
        for op in result.get("failed_ops", []):
            failed.append([
                result["mlir_stem"], result["linear_arch"], op["type"],
                op["name"], op["ssa"], "FAILED", "FAILED",
            ])
    if failed:
        with open(detail_path, "a", newline="") as f:
            csv.writer(f).writerows(failed)
        txt_path = os.path.join(out_dir, "summary.txt")
        with open(txt_path, "a") as f:
            f.write("\n\nFAILED OPERATORS (replace after successful rerun):\n")
            for row in failed:
                f.write(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]} ({row[4]})\n")


if __name__ == "__main__":
    main()
