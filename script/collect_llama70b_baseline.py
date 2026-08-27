#!/usr/bin/env python3
"""Collect llama3-70b baseline logs into the same CSV schema as baseline_test0412.

Usage:
  python3 script/collect_llama70b_baseline.py
  python3 script/collect_llama70b_baseline.py --out-dir baseline_test_70b
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from datetime import datetime


def _num(s, default=0):
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except (TypeError, ValueError):
        return default


def parse_log(log_path):
    result = {
        "mlir_stem": "",
        "linear_arch": "",
        "ppu_arch": "",
        "dm_arch": "",
        "exit_code": -1,
        "linear_cycles": 0,
        "linear_energy": 0.0,
        "elem_cycles": 0,
        "elem_energy": 0.0,
        "dm_cycles": 0,
        "dm_mem_energy": 0.0,
        "dm_total_energy": 0.0,
        "dram_access": 0,
        "total_cycles": 0,
        "total_energy": 0.0,
        "total_flops": 0,
        "throughput_gflops": 0.0,
        "energy_eff_gflops_j": 0.0,
        "total_ops": 0,
        "succeeded": 0,
        "failed": 0,
        "linear_ops": [],
        "elem_ops": [],
        "dm_ops": [],
        "failed_ops": [],
    }

    if not os.path.isfile(log_path):
        return result

    with open(log_path) as f:
        content = f.read()

    m = re.search(r"MLIR file:\s+(.+)", content)
    if m:
        result["mlir_stem"] = os.path.splitext(os.path.basename(m.group(1).strip()))[0]

    m = re.search(r"linear arch:\s+(\S+)", content)
    if m:
        result["linear_arch"] = m.group(1)

    m = re.search(r"nonlinear arch:\s+(\S+)", content)
    if m:
        result["ppu_arch"] = m.group(1)

    m = re.search(r"datamove arch:\s+(\S+)", content)
    if m:
        result["dm_arch"] = m.group(1)

    m = re.search(r"EXIT_CODE=(\d+)", content)
    if m:
        result["exit_code"] = int(m.group(1))

    patterns = {
        "total_cycles": r"Latency:\s+(\d+)\s+cycles",
        "total_energy": r"^\s{4}Energy:\s+([\d.eE+\-]+)\s*uJ",
        "total_flops": r"FLOPs:\s+(\d+)",
        "throughput_gflops": r"Throughput:\s+([\d.eE+\-]+)\s*GFLOPS",
        "energy_eff_gflops_j": r"Energy Efficiency:\s+([\d.eE+\-]+)\s*GFLOPS/J",
        "dram_access": r"DRAM Access \(dm\):\s+(\d+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, content, re.MULTILINE)
        if m:
            result[key] = _num(m.group(1), result[key])

    m = re.search(r"Linear total: Cycles=(\d+)\s+Energy=([\d.eE+\-]+)", content)
    if m:
        result["linear_cycles"] = int(m.group(1))
        result["linear_energy"] = float(m.group(2))

    m = re.search(r"Elementwise total: Cycles=(\d+)\s+Energy=([\d.eE+\-]+)", content)
    if m:
        result["elem_cycles"] = int(m.group(1))
        result["elem_energy"] = float(m.group(2))

    m = re.search(
        r"Datamove total: Cycles=(\d+)\s+MemEnergy=([\d.eE+\-]+)\s+uJ\s+\(TotalEnergy=([\d.eE+\-]+)\s+uJ\)",
        content,
    )
    if m:
        result["dm_cycles"] = int(m.group(1))
        result["dm_mem_energy"] = float(m.group(2))
        result["dm_total_energy"] = float(m.group(3))

    corrected_energy = result["linear_energy"] + result["elem_energy"] + result["dm_mem_energy"]
    if corrected_energy > 0:
        result["total_energy"] = corrected_energy
        if result["total_flops"] > 0:
            result["energy_eff_gflops_j"] = result["total_flops"] / (corrected_energy * 1e-6) / 1e9

    m = re.search(
        r"Operators:\s+(\d+)\s+\(linear=(\d+)\s+elem=(\d+)\s+datamove=(\d+)\)"
        r"\s+Succeeded=(\d+)\s+Failed=(\d+)",
        content,
    )
    if m:
        result["total_ops"] = int(m.group(1))
        result["succeeded"] = int(m.group(5))
        result["failed"] = int(m.group(6))

    parse_ops(content, result, "Linear", "linear_ops", "linear", result["linear_arch"])
    parse_ops(content, result, "Elementwise", "elem_ops", "elementwise", result["ppu_arch"])
    parse_datamove_ops(content, result)
    return result


def parse_ops(content, result, title, key, op_type, arch):
    section = re.search(rf"--- {title} Ops.*?\n-+\n(.*?)\n-+\n", content, re.DOTALL)
    if not section:
        return
    for line in section.group(1).strip().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "FAILED":
            result["failed_ops"].append({"type": op_type, "ssa": parts[0], "name": parts[1]})
            continue
        if len(parts) < 4:
            continue
        dram_acc = 0
        if len(parts) >= 7:
            dram_acc = int(_num(parts[-1], 0))
        util = parts[5].rstrip("%") if len(parts) >= 6 else ""
        result[key].append({
            "ssa": parts[0],
            "name": parts[1],
            "cycles": int(_num(parts[2], 0)),
            "energy": float(_num(parts[3], 0.0)),
            "arch": arch,
            "util": util,
            "dram_acc": dram_acc,
        })


def parse_datamove_ops(content, result):
    section = re.search(r"--- Data Movement Ops.*?\n-+\n(.*?)\n-+\n", content, re.DOTALL)
    if not section:
        return
    for line in section.group(1).strip().splitlines():
        parts = line.split()
        if len(parts) == 7:
            ssa, name = parts[0], parts[1]
            cycles, energy = parts[2], parts[4]
        elif len(parts) == 6:
            merged = re.match(r"(%\S+?)(transpose_\d+|broadcast_\d+|type_cast_\d+)$", parts[0])
            ssa, name = (merged.group(1), merged.group(2)) if merged else (parts[0], "?")
            cycles, energy = parts[1], parts[3]
        else:
            continue
        if cycles == "FAILED":
            result["failed_ops"].append({"type": "datamove", "ssa": ssa, "name": name})
            continue
        result["dm_ops"].append({
            "ssa": ssa,
            "name": name,
            "cycles": int(_num(cycles, 0)),
            "energy": float(_num(energy, 0.0)),
            "arch": result["dm_arch"],
            "util": "",
            "dram_acc": int(_num(parts[-1], 0)),
        })


def collect(out_dir, title="Llama3-70B"):
    log_dir = os.path.join(out_dir, "logs")
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    results = [parse_log(path) for path in log_files]
    results = [r for r in results if r["mlir_stem"] and r["total_cycles"] > 0]

    if not results:
        raise SystemExit(f"No completed logs found in {log_dir}")

    os.makedirs(out_dir, exist_ok=True)
    write_summary_csv(out_dir, results)
    write_detail_csv(out_dir, results)
    write_summary_txt(out_dir, results, title)
    print(f"Parsed {len(results)} workloads from {log_dir}")
    print(f"Saved: {out_dir}/summary.csv  {out_dir}/summary_detail.csv  {out_dir}/summary.txt")


def write_summary_csv(out_dir, results):
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mlir_stem", "linear_arch", "ppu_arch", "dm_arch",
            "latency_cycles", "energy_uJ", "FLOPs",
            "throughput_GFLOPS", "energy_eff_GFLOPS_per_J",
            "dram_access", "linear_cycles", "linear_energy_uJ",
            "elem_cycles", "elem_energy_uJ",
            "total_ops", "succeeded", "failed", "exit_code",
        ])
        for r in results:
            writer.writerow([
                r["mlir_stem"], r["linear_arch"], r["ppu_arch"], r["dm_arch"],
                r["total_cycles"], f"{r['total_energy']:.6f}", r["total_flops"],
                f"{r['throughput_gflops']:.4f}", f"{r['energy_eff_gflops_j']:.4f}",
                r["dram_access"], r["linear_cycles"], f"{r['linear_energy']:.6f}",
                r["elem_cycles"], f"{r['elem_energy']:.6f}",
                r["total_ops"], r["succeeded"], r["failed"], r["exit_code"],
            ])


def write_detail_csv(out_dir, results):
    with open(os.path.join(out_dir, "summary_detail.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mlir_stem", "linear_arch", "op_type", "op_name", "ssa", "cycles", "energy_uJ"])
        for r in results:
            for key, op_type in (("linear_ops", "linear"), ("elem_ops", "elementwise"), ("dm_ops", "datamove")):
                for op in r[key]:
                    writer.writerow([
                        r["mlir_stem"], r["linear_arch"], op_type,
                        op["name"], op["ssa"], op["cycles"], f"{op['energy']:.6f}",
                    ])


def write_summary_txt(out_dir, results, title="Llama3-70B"):
    by_arch = defaultdict(list)
    for r in results:
        by_arch[r["linear_arch"]].append(r)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"  {title} Baseline Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        grand_cycles = grand_flops = grand_dram = grand_ok = grand_fail = 0
        grand_energy = 0.0

        for arch, rows in sorted(by_arch.items()):
            rows = sorted(rows, key=lambda r: r["mlir_stem"])
            f.write(f"Architecture: {arch}\n")
            f.write(f"  PPU: {rows[0]['ppu_arch']}  DM: {rows[0]['dm_arch']}\n\n")

            arch_cycles = arch_flops = arch_dram = 0
            arch_energy = 0.0
            for r in rows:
                f.write(
                    f"  {r['mlir_stem']}: "
                    f"Latency={r['total_cycles']} cycles  "
                    f"Energy={r['total_energy']:.4f} uJ  "
                    f"FLOPs={r['total_flops']}  "
                    f"Thpt={r['throughput_gflops']:.4f} GFLOPS  "
                    f"Eff={r['energy_eff_gflops_j']:.4f} GFLOPS/J  "
                    f"DRAM_Acc={r['dram_access']}  "
                    f"OK={r['succeeded']} Fail={r['failed']}\n"
                )
                arch_cycles += r["total_cycles"]
                arch_energy += r["total_energy"]
                arch_flops += r["total_flops"]
                arch_dram += r["dram_access"]
                grand_ok += r["succeeded"]
                grand_fail += r["failed"]

            arch_thpt = arch_flops / arch_cycles if arch_cycles else 0.0
            arch_eff = arch_flops / (arch_energy * 1e-6) / 1e9 if arch_energy else 0.0
            f.write(
                f"\n  TOTAL: Latency={arch_cycles} cycles  "
                f"Energy={arch_energy:.4f} uJ  FLOPs={arch_flops}  "
                f"Thpt={arch_thpt:.4f} GFLOPS  Eff={arch_eff:.4f} GFLOPS/J  "
                f"DRAM_Acc={arch_dram}  Workloads={len(rows)}\n\n"
            )
            grand_cycles += arch_cycles
            grand_energy += arch_energy
            grand_flops += arch_flops
            grand_dram += arch_dram

        grand_thpt = grand_flops / grand_cycles if grand_cycles else 0.0
        grand_eff = grand_flops / (grand_energy * 1e-6) / 1e9 if grand_energy else 0.0
        f.write("=" * 80 + "\n")
        f.write("GRAND TOTAL:\n")
        f.write(f"  Latency: {grand_cycles} cycles\n")
        f.write(f"  Energy: {grand_energy:.4f} uJ\n")
        f.write(f"  FLOPs: {grand_flops}\n")
        f.write(f"  Throughput: {grand_thpt:.4f} GFLOPS\n")
        f.write(f"  Energy Efficiency: {grand_eff:.4f} GFLOPS/J\n")
        f.write(f"  DRAM Access: {grand_dram}\n")
        f.write(f"  Operators: succeeded={grand_ok} failed={grand_fail}\n")
        f.write(f"  Workloads: {len(results)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="baseline_test_70b")
    parser.add_argument("--title", default="Llama3-70B")
    args = parser.parse_args()
    collect(args.out_dir, args.title)


if __name__ == "__main__":
    main()
