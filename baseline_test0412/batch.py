#!/usr/bin/env python3
"""
Baseline Test Batch Orchestrator
=================================
1. Reads testorder.txt (360 commands)
2. Dispatches via tmux panes for parallel execution
3. After all done, parses logs and generates summary CSV/TXT

Usage:
  python3 batch.py                    # Run all 360 tests with 8 parallel panes
  python3 batch.py --parallel 4       # Use 4 parallel panes
  python3 batch.py --summary-only     # Only generate summary from existing logs
  python3 batch.py --group lego_edge  # Only run tests for a specific arch group
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
WORK_DIR = "/home/accelgen"
LOG_DIR = os.path.join(WORK_DIR, "baseline_test0412/logs")
TESTORDER_FILE = os.path.join(WORK_DIR, "baseline_test0412/testorder.txt")
SUMMARY_DIR = os.path.join(WORK_DIR, "baseline_test0412")

ARCH_GROUPS = {
    "gemmini_os_edge":   {"ppu": "PPU_edge",   "dm": "PPU_datamove_edge",   "batch": "b1"},
    "gemmini_ws_edge":   {"ppu": "PPU_edge",   "dm": "PPU_datamove_edge",   "batch": "b1"},
    "lego_edge":         {"ppu": "PPU_edge",   "dm": "PPU_datamove_edge",   "batch": "b1"},
    "gemmini_os_server": {"ppu": "PPU_server", "dm": "PPU_datamove_server", "batch": "b8"},
    "gemmini_ws_server": {"ppu": "PPU_server", "dm": "PPU_datamove_server", "batch": "b8"},
    "lego_server":       {"ppu": "PPU_server", "dm": "PPU_datamove_server", "batch": "b8"},
}


# ============================================================================
# Parse a single log file for summary metrics
# ============================================================================
def parse_log(log_path):
    """Parse a baseline test log file, return dict of metrics."""
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
        "elem_dram_accesses": 0,
        "dm_cycles": 0,
        "dm_mem_energy": 0.0,
        "dm_total_energy": 0.0,
        "dm_dram_accesses": 0,
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

    content = open(log_path).read()

    # Extract config
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

    # Parse exit code
    m = re.search(r"EXIT_CODE=(\d+)", content)
    if m:
        result["exit_code"] = int(m.group(1))

    # Parse GRAND TOTAL metrics (new format)
    m = re.search(r"Latency:\s+(\d+)\s+cycles", content)
    if m:
        result["total_cycles"] = int(m.group(1))

    m = re.search(r"^\s{4}Energy:\s+([\d.eE+\-]+)\s*uJ", content, re.MULTILINE)
    if m:
        result["total_energy"] = float(m.group(1))

    m = re.search(r"FLOPs:\s+(\d+)", content)
    if m:
        result["total_flops"] = int(m.group(1))

    m = re.search(r"Throughput:\s+([\d.eE+\-]+)\s*GFLOPS", content)
    if m:
        result["throughput_gflops"] = float(m.group(1))

    m = re.search(r"Energy Efficiency:\s+([\d.eE+\-]+)\s*GFLOPS/J", content)
    if m:
        result["energy_eff_gflops_j"] = float(m.group(1))

    m = re.search(r"DRAM Access \(dm\):\s+(\d+)", content)
    if m:
        result["dm_dram_accesses"] = int(m.group(1))

    # Backwards compatibility with old format
    if result["total_cycles"] == 0:
        m = re.search(r"GRAND TOTAL: Cycles=(\d+)\s+Energy=([\d.eE+\-]+)\s*uJ", content)
        if m:
            result["total_cycles"] = int(m.group(1))
            result["total_energy"] = float(m.group(2))

    # Parse per-section totals
    m = re.search(r"Linear total: Cycles=(\d+)\s+Energy=([\d.eE+\-]+)", content)
    if m:
        result["linear_cycles"] = int(m.group(1))
        result["linear_energy"] = float(m.group(2))

    m = re.search(r"Elementwise total: Cycles=(\d+)\s+Energy=([\d.eE+\-]+)(?:\s+uJ\s+DRAM_Acc=(\d+))?", content)
    if m:
        result["elem_cycles"] = int(m.group(1))
        result["elem_energy"] = float(m.group(2))
        if m.group(3):
            result["elem_dram_accesses"] = int(m.group(3))

    m = re.search(r"Datamove total: Cycles=(\d+)\s+MemEnergy=([\d.eE+\-]+)\s+uJ\s+\(TotalEnergy=([\d.eE+\-]+)\s+uJ\)", content)
    if m:
        result["dm_cycles"] = int(m.group(1))
        result["dm_mem_energy"] = float(m.group(2))
        result["dm_total_energy"] = float(m.group(3))

    # Fix: C++ GRAND TOTAL Energy omits datamove energy.
    # Recalculate as linear + elementwise + datamove(MemEnergy, excluding MAC).
    corrected_energy = result["linear_energy"] + result["elem_energy"] + result["dm_mem_energy"]
    if corrected_energy > 0:
        result["total_energy"] = corrected_energy
        if result["total_flops"] > 0:
            result["energy_eff_gflops_j"] = (
                result["total_flops"] / (corrected_energy * 1e-6) / 1e9
            )

    # Parse operator counts
    m = re.search(
        r"Operators:\s+(\d+)\s+\(linear=(\d+)\s+elem=(\d+)\s+datamove=(\d+)\)"
        r"\s+Succeeded=(\d+)\s+Failed=(\d+)",
        content,
    )
    if m:
        result["total_ops"] = int(m.group(1))
        result["succeeded"] = int(m.group(5))
        result["failed"] = int(m.group(6))

    # Parse individual linear ops
    lin_header = re.search(r"--- Linear Ops \(([^)]+)\) ---", content)
    lin_arch = lin_header.group(1) if lin_header else result.get("linear_arch", "")
    lin_section = re.search(
        r"--- Linear Ops.*?\n-+\n(.*?)\n-+\n", content, re.DOTALL
    )
    if lin_section:
        for line in lin_section.group(1).strip().split("\n"):
            parts = line.split()
            if len(parts) >= 4 and parts[2] != "FAILED":
                util_str = ""
                if len(parts) >= 6:
                    util_str = parts[5].rstrip("%")
                dram_acc = 0
                try:
                    dram_acc = int(float(parts[-1]))
                except (ValueError, IndexError):
                    dram_acc = 0
                result["linear_ops"].append({
                    "ssa": parts[0],
                    "name": parts[1],
                    "cycles": int(float(parts[2])),
                    "energy": float(parts[3]),
                    "arch": lin_arch,
                    "util": util_str,
                    "dram_acc": dram_acc,
                })
            elif len(parts) >= 3 and parts[2] == "FAILED":
                result["failed_ops"].append({"type": "linear", "ssa": parts[0], "name": parts[1]})

    # Parse individual elementwise ops
    elem_header = re.search(r"--- Elementwise Ops \(([^)]+)\) ---", content)
    elem_arch = elem_header.group(1) if elem_header else result.get("ppu_arch", "")
    elem_section = re.search(
        r"--- Elementwise Ops.*?\n-+\n(.*?)\n-+\n", content, re.DOTALL
    )
    if elem_section:
        for line in elem_section.group(1).strip().split("\n"):
            parts = line.split()
            if len(parts) >= 4 and parts[2] != "FAILED":
                util_str = ""
                if len(parts) >= 6:
                    util_str = parts[5].rstrip("%")
                # New C++ format has 7 parts: SSA OpName Cycles Energy Computes Util% DRAM_Acc
                # Old format has 6 parts: SSA OpName Cycles Energy Computes Util%
                dram_acc = 0
                if len(parts) >= 7:
                    try:
                        dram_acc = int(parts[-1])
                    except (ValueError, IndexError):
                        dram_acc = 0
                result["elem_ops"].append({
                    "ssa": parts[0],
                    "name": parts[1],
                    "cycles": int(float(parts[2])),
                    "energy": float(parts[3]),
                    "arch": elem_arch,
                    "util": util_str,
                    "dram_acc": dram_acc,
                })
            elif len(parts) >= 3 and parts[2] == "FAILED":
                result["failed_ops"].append({"type": "elementwise", "ssa": parts[0], "name": parts[1]})

    # Parse individual datamove ops
    dm_header = re.search(r"--- Data Movement Ops \(([^)]+)\) ---", content)
    dm_arch = dm_header.group(1) if dm_header else result.get("dm_arch", "")
    dm_section = re.search(
        r"--- Data Movement Ops.*?\n-+\n(.*?)\n-+\n", content, re.DOTALL
    )
    if dm_section:
        for line in dm_section.group(1).strip().split("\n"):
            parts = line.split()
            if len(parts) < 4:
                continue
            # Detect SSA+OpName merge: 7 parts = normal, 6 parts = merged
            # Normal:  [SSA, OpName, Cycles, Energy, MemEnergy, MacEnergy, DRAM_Acc]
            # Merged:  [SSA+OpName, Cycles, Energy, MemEnergy, MacEnergy, DRAM_Acc]
            # Use MemEnergy (index 4 / 3) instead of TotalEnergy for display
            if len(parts) == 7:
                ssa = parts[0]
                name = parts[1]
                cycles_str = parts[2]
                energy_str = parts[4]  # MemEnergy (excl. MAC)
            elif len(parts) == 6:
                # SSA and OpName merged — split with regex
                m_ssa = re.match(r'(%\S+?)(transpose_\d+)$', parts[0])
                if m_ssa:
                    ssa = m_ssa.group(1)
                    name = m_ssa.group(2)
                else:
                    ssa = parts[0]
                    name = "?"
                cycles_str = parts[1]
                energy_str = parts[3]  # MemEnergy (excl. MAC)
            else:
                continue
            if cycles_str == "FAILED":
                result["failed_ops"].append({"type": "datamove", "ssa": ssa, "name": name})
                continue
            dram_acc = 0
            try:
                dram_acc = int(float(parts[-1]))
            except (ValueError, IndexError):
                dram_acc = 0
            result["dm_ops"].append({
                "ssa": ssa,
                "name": name,
                "cycles": int(float(cycles_str)),
                "energy": float(energy_str),
                "arch": dm_arch,
                "util": "",
                "dram_acc": dram_acc,
            })

    return result


# ============================================================================
# Generate summary files
# ============================================================================
def generate_summary():
    """Parse all logs and generate summary CSV and TXT."""
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))

    if not log_files:
        print("No log files found in", LOG_DIR)
        return

    print(f"\nParsing {len(log_files)} log files...")

    all_results = []
    for lf in log_files:
        r = parse_log(lf)
        # Only include logs that have completed (have GRAND TOTAL with cycles > 0)
        if r["mlir_stem"] and r["total_cycles"] > 0:
            all_results.append(r)

    if not all_results:
        print("No valid results found.")
        return

    # ---- summary.csv: one row per (mlir_stem, linear_arch) ----
    csv_path = os.path.join(SUMMARY_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mlir_stem", "linear_arch", "ppu_arch", "dm_arch",
            "latency_cycles", "energy_uJ", "FLOPs",
            "throughput_GFLOPS", "energy_eff_GFLOPS_per_J",
            "dram_access",
            "linear_cycles", "linear_energy_uJ",
            "elem_cycles", "elem_energy_uJ",
            "total_ops", "succeeded", "failed", "exit_code",
        ])
        for r in all_results:
            writer.writerow([
                r["mlir_stem"], r["linear_arch"], r["ppu_arch"], r["dm_arch"],
                r["total_cycles"], f'{r["total_energy"]:.6f}', r["total_flops"],
                f'{r["throughput_gflops"]:.4f}', f'{r["energy_eff_gflops_j"]:.4f}',
                r["dm_dram_accesses"],
                r["linear_cycles"], f'{r["linear_energy"]:.6f}',
                r["elem_cycles"], f'{r["elem_energy"]:.6f}',
                r["total_ops"], r["succeeded"], r["failed"], r["exit_code"],
            ])
    print(f"  Written: {csv_path}")

    # ---- summary_detail.csv: one row per operator ----
    detail_path = os.path.join(SUMMARY_DIR, "summary_detail.csv")
    with open(detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mlir_stem", "linear_arch", "op_type", "op_name",
            "ssa", "cycles", "energy_uJ",
        ])
        for r in all_results:
            for op in r["linear_ops"]:
                writer.writerow([
                    r["mlir_stem"], r["linear_arch"], "linear",
                    op["name"], op["ssa"], op["cycles"], f'{op["energy"]:.6f}',
                ])
            for op in r["elem_ops"]:
                writer.writerow([
                    r["mlir_stem"], r["linear_arch"], "elementwise",
                    op["name"], op["ssa"], op["cycles"], f'{op["energy"]:.6f}',
                ])
            for op in r["dm_ops"]:
                writer.writerow([
                    r["mlir_stem"], r["linear_arch"], "datamove",
                    op["name"], op["ssa"], op["cycles"], f'{op["energy"]:.6f}',
                ])
    print(f"  Written: {detail_path}")

    # ---- summary.txt: human-readable grouped by architecture ----
    txt_path = os.path.join(SUMMARY_DIR, "summary.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"  Baseline Test Summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Group by linear arch
        by_arch = defaultdict(list)
        for r in all_results:
            by_arch[r["linear_arch"]].append(r)

        grand_total_cycles = 0
        grand_total_energy = 0.0
        grand_total_flops = 0
        grand_total_dram = 0
        grand_succeeded = 0
        grand_failed = 0

        for arch in sorted(by_arch.keys()):
            results = sorted(by_arch[arch], key=lambda x: x["mlir_stem"])
            f.write(f"\n{'#' * 70}\n")
            f.write(f"# Architecture: {arch}\n")
            f.write(f"#   PPU: {results[0]['ppu_arch']}  DM: {results[0]['dm_arch']}\n")
            f.write(f"#   Workloads: {len(results)}\n")
            f.write(f"{'#' * 70}\n\n")

            arch_cycles = 0
            arch_energy = 0.0
            arch_flops = 0
            arch_dram = 0

            for r in results:
                dram_bytes = r['dm_dram_accesses'] * 2  # 16-bit = 2 bytes per access

                # ── Workload header ──
                f.write(f"  Workload: {r['mlir_stem']}\n")
                f.write(f"  {'─' * 100}\n")

                # ── Per-operator detail table ──
                all_ops = []
                for op in r["linear_ops"]:
                    all_ops.append(("linear", op))
                for op in r["elem_ops"]:
                    all_ops.append(("elementwise", op))
                for op in r["dm_ops"]:
                    all_ops.append(("datamove", op))

                if all_ops:
                    f.write(f"    {'Type':<14} {'SSA':<10} {'Op Name':<30} "
                            f"{'Arch':<22} {'Util%':>8} "
                            f"{'Cycles':>14} {'Energy(uJ)':>14} {'DRAM_Acc':>14}\n")
                    f.write(f"    {'─' * 132}\n")
                    for op_type, op in all_ops:
                        util_disp = op.get('util', '')
                        if util_disp:
                            try:
                                util_disp = f"{float(util_disp):.1f}%"
                            except ValueError:
                                util_disp = util_disp
                        else:
                            util_disp = "-"
                        arch_disp = op.get('arch', '')
                        dram_disp = op.get('dram_acc', '')
                        if dram_disp != '' and dram_disp != 0:
                            dram_disp = f"{dram_disp:>14}"
                        else:
                            dram_disp = f"{'-':>14}"
                        f.write(f"    {op_type:<14} {op['ssa']:<10} {op['name']:<30} "
                                f"{arch_disp:<22} {util_disp:>8} "
                                f"{op['cycles']:>14} {op['energy']:>14.4f} {dram_disp}\n")
                    f.write(f"    {'─' * 132}\n")

                # ── Workload summary ──
                f.write(f"    Summary: Latency={r['total_cycles']} cycles  "
                        f"Energy={r['total_energy']:.4f} uJ  "
                        f"FLOPs={r['total_flops']}  "
                        f"Thpt={r['throughput_gflops']:.4f} GFLOPS  "
                        f"Eff={r['energy_eff_gflops_j']:.4f} GFLOPS/J  "
                        f"DRAM={dram_bytes} B  "
                        f"OK={r['succeeded']} Fail={r['failed']}\n")
                f.write("\n")

                arch_cycles += r["total_cycles"]
                arch_energy += r["total_energy"]
                arch_flops += r["total_flops"]
                arch_dram += r["dm_dram_accesses"]
                grand_succeeded += r["succeeded"]
                grand_failed += r["failed"]

            # ── Architecture total ──
            arch_thpt = arch_flops / (arch_cycles * 1e-9) / 1e9 if arch_cycles > 0 else 0
            arch_eff = arch_flops / (arch_energy * 1e-6) / 1e9 if arch_energy > 0 else 0
            arch_dram_bytes = arch_dram * 2
            f.write(f"  {'━' * 100}\n")
            f.write(f"  {arch} TOTAL: Latency={arch_cycles} cycles  "
                    f"Energy={arch_energy:.4f} uJ  "
                    f"FLOPs={arch_flops}  "
                    f"Thpt={arch_thpt:.4f} GFLOPS  "
                    f"Eff={arch_eff:.4f} GFLOPS/J  "
                    f"DRAM_Acc={arch_dram_bytes} B  "
                    f"Workloads={len(results)}\n\n")
            grand_total_cycles += arch_cycles
            grand_total_energy += arch_energy
            grand_total_flops += arch_flops
            grand_total_dram += arch_dram

        grand_thpt = grand_total_flops / (grand_total_cycles * 1e-9) / 1e9 if grand_total_cycles > 0 else 0
        grand_eff = grand_total_flops / (grand_total_energy * 1e-6) / 1e9 if grand_total_energy > 0 else 0

        f.write("=" * 80 + "\n")
        f.write(f"  GRAND TOTAL across all architectures:\n")
        f.write(f"    Latency:           {grand_total_cycles} cycles\n")
        f.write(f"    Energy:            {grand_total_energy:.4f} uJ\n")
        f.write(f"    FLOPs:             {grand_total_flops}\n")
        f.write(f"    Throughput:         {grand_thpt:.4f} GFLOPS\n")
        f.write(f"    Energy Efficiency:  {grand_eff:.4f} GFLOPS/J\n")
        grand_dram_bytes = grand_total_dram * 2
        f.write(f"    DRAM Access:        {grand_dram_bytes} B\n")
        f.write(f"    Operators: succeeded={grand_succeeded}  failed={grand_failed}\n")
        f.write(f"    Workloads: {len(all_results)}\n")

        # Collect and write all failed operators
        all_failed = []
        for r in all_results:
            for fop in r.get("failed_ops", []):
                all_failed.append({
                    "workload": r["mlir_stem"],
                    "arch": r["linear_arch"],
                    **fop,
                })
        if all_failed:
            f.write(f"\n    Failed operators ({len(all_failed)}):\n")
            for fo in all_failed:
                f.write(f"      - {fo['workload']} @ {fo['arch']}: "
                        f"{fo['type']} {fo['name']} ({fo['ssa']})\n")

        f.write("=" * 80 + "\n")

    print(f"  Written: {txt_path}")
    print(f"\nSummary: {len(all_results)} workloads across "
          f"{len(by_arch)} architectures")
    print(f"  Total Latency: {grand_total_cycles} cycles")
    print(f"  Total Energy:  {grand_total_energy:.4f} uJ")
    print(f"  Total FLOPs:   {grand_total_flops}")
    print(f"  Throughput:    {grand_thpt:.4f} GFLOPS")
    print(f"  Energy Eff:    {grand_eff:.4f} GFLOPS/J")
    print(f"  DRAM Access:   {grand_dram_bytes} B")
    print(f"  Succeeded: {grand_succeeded}  Failed: {grand_failed}")


# ============================================================================
# Tmux-based parallel execution
# ============================================================================
def run_parallel(commands, num_parallel=8):
    """Run commands in parallel using tmux panes."""
    session_name = f"baseline_{datetime.now().strftime('%H%M%S')}"

    # Kill existing session if any
    subprocess.run(["tmux", "kill-session", "-t", session_name],
                   capture_output=True)

    # Split commands into chunks for each pane
    chunks = [[] for _ in range(num_parallel)]
    for i, cmd in enumerate(commands):
        chunks[i % num_parallel].append(cmd)

    # Create wrapper scripts for each pane
    script_dir = os.path.join(LOG_DIR, "pane_scripts")
    os.makedirs(script_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        script_path = os.path.join(script_dir, f"pane_{i}.sh")
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"cd {WORK_DIR}\n")
            f.write(f"echo 'Pane {i}: {len(chunk)} tasks'\n")
            for j, cmd in enumerate(chunk):
                # Extract stem and arch from command for logging
                m = re.search(r"mlir/(\S+)\.mlir.*accelerator-name=(\S+)\s", cmd)
                if m:
                    stem, arch = m.group(1), m.group(2)
                    log_file = os.path.join(LOG_DIR, f"{stem}_{arch}.log")
                else:
                    stem = f"task_{i}_{j}"
                    log_file = os.path.join(LOG_DIR, f"{stem}.log")

                arch_label = arch if m else "unknown"
                f.write(f"\necho '[{j+1}/{len(chunk)}] {stem} on {arch_label}'\n")
                # Quote the -pass-pipeline argument to prevent shell word-splitting
                # on spaces inside {…}
                quoted_cmd = re.sub(
                    r'(-pass-pipeline=\S+\{.*?\})',
                    r"'\1'",
                    cmd,
                )
                f.write(f"{quoted_cmd} 2>&1 | tee {log_file}\n")
                f.write(f"echo 'EXIT_CODE='$? >> {log_file}\n")

            f.write(f"\necho 'Pane {i}: ALL DONE'\n")
            f.write("read -p 'Press Enter to close...'\n")
        os.chmod(script_path, 0o755)

    # Create tmux session
    first_script = os.path.join(script_dir, "pane_0.sh")
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name,
                     f"bash {first_script}"])

    for i in range(1, num_parallel):
        script_path = os.path.join(script_dir, f"pane_{i}.sh")
        if os.path.isfile(script_path) and chunks[i]:
            subprocess.run(["tmux", "split-window", "-t", session_name,
                            f"bash {script_path}"])
            subprocess.run(["tmux", "select-layout", "-t", session_name,
                            "tiled"])

    print(f"\nStarted tmux session: {session_name}")
    print(f"  Parallel panes: {min(num_parallel, len([c for c in chunks if c]))}")
    print(f"  Total commands: {len(commands)}")
    print(f"\n  To attach:  tmux attach -t {session_name}")
    print(f"  To monitor: watch -n5 'ls {LOG_DIR}/*.log | wc -l'")
    print(f"\n  After completion, run:  python3 batch.py --summary-only")


# ============================================================================
# Sequential execution (no tmux dependency)
# ============================================================================
def run_sequential(commands):
    """Run commands one by one (for debugging or small subsets)."""
    total = len(commands)
    failed = 0

    for i, cmd in enumerate(commands):
        m = re.search(r"mlir/(\S+)\.mlir.*accelerator-name=(\S+)\s", cmd)
        if m:
            stem, arch = m.group(1), m.group(2)
            log_file = os.path.join(LOG_DIR, f"{stem}_{arch}.log")
        else:
            stem, arch = f"task_{i}", "unknown"
            log_file = os.path.join(LOG_DIR, f"{stem}.log")

        print(f"\n[{i+1}/{total}] {stem} on {arch}")

        result = subprocess.run(
            f"cd {WORK_DIR} && {cmd}",
            shell=True, capture_output=True, text=True
        )
        output = result.stdout + result.stderr
        with open(log_file, "w") as f:
            f.write(output)
            f.write(f"\nEXIT_CODE={result.returncode}\n")

        if result.returncode != 0:
            print(f"  *** FAILED (exit={result.returncode})")
            failed += 1
        else:
            print(f"  OK")

    print(f"\nDone. Total={total} Failed={failed}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Baseline test orchestrator")
    parser.add_argument("--parallel", type=int, default=8,
                        help="Number of parallel tmux panes (default: 8)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially instead of tmux")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only generate summary from existing logs")
    parser.add_argument("--group", type=str, default="",
                        help="Only run a specific arch group (e.g. lego_edge)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    if args.summary_only:
        generate_summary()
        return

    # Read commands
    if not os.path.isfile(TESTORDER_FILE):
        print(f"ERROR: {TESTORDER_FILE} not found.")
        print("Run: bash baseline_test/generate_testorder.sh > baseline_test/testorder.txt")
        sys.exit(1)

    with open(TESTORDER_FILE) as f:
        commands = [line.strip() for line in f if line.strip()]

    # Filter by group if specified
    if args.group:
        commands = [c for c in commands if f"accelerator-name={args.group}" in c]
        if not commands:
            print(f"No commands found for group: {args.group}")
            sys.exit(1)

    print(f"Commands to run: {len(commands)}")

    if args.dry_run:
        for c in commands:
            print(c)
        return

    if args.sequential:
        run_sequential(commands)
    else:
        run_parallel(commands, args.parallel)

    # Always generate summary after execution (for sequential mode)
    if args.sequential:
        generate_summary()


if __name__ == "__main__":
    main()
