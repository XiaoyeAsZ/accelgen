#!/usr/bin/env python3
"""
Patch the 5 log files that still contain FAILED elementwise ops.
Replace FAILED lines with actual results from timeloop-mapper.stats.txt,
and update Elementwise total, GRAND TOTAL, and Succeeded/Failed counts.
"""
import re, os, sys

BASE = "/home/accelgen/baseline_test0412"

# Map: (log_file_basename, model, [op_names])
PATCHES = [
    # gemma-7b on lego_server: 7 elem ops failed
    ("gemma-7b-block0-ffn-prefill-b8s4096_lego_server.log",
     "lego_server", "gemma-7b",
     ["sqrt_0", "divf_1", "erf_2", "addf_3", "mulf_4", "mulf_5", "mulf_6"]),
    # gemma-7b on gemmini_ws_server: 7 elem ops failed
    ("gemma-7b-block0-ffn-prefill-b8s4096_gemmini_ws_server.log",
     "gemmini_ws_server", "gemma-7b",
     ["sqrt_0", "divf_1", "erf_2", "addf_3", "mulf_4", "mulf_5", "mulf_6"]),
    # llama3-8b on lego_server: 5 elem ops failed
    ("llama3-8b-block0-ffn-prefill-b8s4096_lego_server.log",
     "lego_server", "llama3-8b",
     ["negf_0", "addf_2", "divf_3", "mulf_4", "mulf_5"]),
    # llama3-8b on gemmini_os_server: 5 elem ops failed
    ("llama3-8b-block0-ffn-prefill-b8s4096_gemmini_os_server.log",
     "gemmini_os_server", "llama3-8b",
     ["negf_0", "addf_2", "divf_3", "mulf_4", "mulf_5"]),
    # llama3-8b on gemmini_ws_server: 5 elem ops failed
    ("llama3-8b-block0-ffn-prefill-b8s4096_gemmini_ws_server.log",
     "gemmini_ws_server", "llama3-8b",
     ["negf_0", "addf_2", "divf_3", "mulf_4", "mulf_5"]),
]


def parse_stats(stats_path):
    """Extract Cycles, Energy(uJ), Computes, Utilization%, DRAM_Acc from timeloop stats."""
    with open(stats_path) as f:
        content = f.read()

    cycles = int(re.search(r"^Cycles:\s+(\d+)", content, re.M).group(1))
    energy_uj = float(re.search(r"^Energy:\s+([\d.]+)\s+uJ", content, re.M).group(1))
    computes = int(re.search(r"^Computes\s*=\s*(\d+)", content, re.M).group(1))
    util = float(re.search(r"^Utilization:\s+([\d.]+)%", content, re.M).group(1))

    # DRAM total scalar accesses - last "Total scalar accesses" in the file
    # which belongs to the DRAM level (the last level)
    dram_section = content.split("=== DRAM ===")[-1]
    dram_acc = int(re.search(r"Total scalar accesses\s+:\s+(\d+)", dram_section).group(1))

    return {
        "cycles": cycles,
        "energy": energy_uj,
        "computes": computes,
        "util": util,
        "dram_acc": dram_acc,
    }


def patch_log(log_basename, arch, model, op_names):
    log_path = os.path.join(BASE, "logs", log_basename)
    print(f"\nPatching: {log_basename}")
    print(f"  Arch={arch}, Model={model}, Ops={op_names}")

    # Collect stats for each op
    op_stats = {}
    for op in op_names:
        stats_path = os.path.join(BASE, arch, model, "ffn", "prefill", "b8s4096", op, "timeloop-mapper.stats.txt")
        if not os.path.exists(stats_path):
            print(f"  WARNING: {stats_path} not found!")
            return False
        op_stats[op] = parse_stats(stats_path)
        s = op_stats[op]
        print(f"  {op}: Cycles={s['cycles']}  Energy={s['energy']:.6e}  Computes={s['computes']}  Util={s['util']:.6e}%  DRAM={s['dram_acc']}")

    with open(log_path) as f:
        lines = f.readlines()

    new_lines = []
    elem_total_cycles = 0
    elem_total_energy = 0.0
    elem_total_dram = 0
    elem_total_computes = 0
    num_fixed = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Replace FAILED lines
        # Format: "%6         sqrt_0                         FAILED\n"
        failed_match = re.match(r'^(%\S+)\s+(\S+)\s+FAILED\s*$', line)
        if failed_match:
            ssa = failed_match.group(1)
            op_name = failed_match.group(2)
            if op_name in op_stats:
                s = op_stats[op_name]
                # Format like successful ops:
                # SSA(10) OpName(31) Cycles(14) Energy(16) Computes(12) Util%(16) DRAM_Acc
                new_line = f"%-10s %-30s %-14d%-16.6e%-12d%.6e%%   %d\n" % (
                    ssa, op_name, s['cycles'], s['energy'], s['computes'], s['util'], s['dram_acc'])
                new_lines.append(new_line)
                elem_total_cycles += s['cycles']
                elem_total_energy += s['energy']
                elem_total_dram += s['dram_acc']
                elem_total_computes += s['computes']
                num_fixed += 1
                i += 1
                continue

        # Replace "Elementwise total" line
        if line.strip().startswith("Elementwise total:"):
            # Parse existing values (they should be 0 for failed case)
            # Replace with our totals
            new_line = f"  Elementwise total: Cycles={elem_total_cycles}  Energy={elem_total_energy:.6e} uJ  DRAM_Acc={elem_total_dram}\n"
            new_lines.append(new_line)
            i += 1
            continue

        # Update GRAND TOTAL section
        # "    Latency:            441499648 cycles ..."
        lat_match = re.match(r'^(\s+Latency:\s+)(\d+)(\s+cycles\s+\()(.+)', line)
        if lat_match and num_fixed > 0:
            old_lat = int(lat_match.group(2))
            new_lat = old_lat + elem_total_cycles
            us = new_lat / 1000.0
            ms = us / 1000.0
            new_line = f"    Latency:            {new_lat} cycles ({new_lat:.6e} us, {ms:.6e} ms)\n"
            new_lines.append(new_line)
            i += 1
            continue

        # "    Energy:             7.029973e+07 uJ"
        eng_match = re.match(r'^(\s+Energy:\s+)([\d.e+\-]+)(\s+uJ)', line)
        if eng_match and num_fixed > 0:
            old_eng = float(eng_match.group(2))
            new_eng = old_eng + elem_total_energy
            new_line = f"    Energy:             {new_eng:.6e} uJ\n"
            new_lines.append(new_line)
            i += 1
            continue

        # "    FLOPs:              14843406974976 (linear_MACs=7421703487488, elem_ops=0)"
        flops_match = re.match(r'^(\s+FLOPs:\s+)(\d+)\s+\(linear_MACs=(\d+),\s*elem_ops=(\d+)\)', line)
        if flops_match and num_fixed > 0:
            old_flops = int(flops_match.group(2))
            linear_macs = int(flops_match.group(3))
            old_elem = int(flops_match.group(4))
            # elem_ops = total computes (each elementwise op's computes)
            new_elem = old_elem + elem_total_computes
            new_flops = linear_macs * 2 + new_elem
            new_line = f"    FLOPs:              {new_flops} (linear_MACs={linear_macs}, elem_ops={new_elem})\n"
            new_lines.append(new_line)
            i += 1
            continue

        # "    Throughput:         3.362043e+04 GFLOPS"
        thpt_match = re.match(r'^(\s+Throughput:\s+)', line)
        if thpt_match and num_fixed > 0:
            # We'll recalculate after knowing new latency and flops
            # Skip for now, we need the new values
            # Actually let's just recalculate from what we have
            # Re-read from new_lines to get new latency and flops
            new_lat_val = None
            new_flops_val = None
            for nl in reversed(new_lines):
                if "Latency:" in nl and new_lat_val is None:
                    m = re.search(r'Latency:\s+(\d+)\s+cycles', nl)
                    if m:
                        new_lat_val = int(m.group(1))
                if "FLOPs:" in nl and new_flops_val is None:
                    m = re.search(r'FLOPs:\s+(\d+)', nl)
                    if m:
                        new_flops_val = int(m.group(1))
            if new_lat_val and new_flops_val:
                thpt = new_flops_val / new_lat_val  # GFLOPS @ 1GHz
                new_line = f"    Throughput:         {thpt:.4e} GFLOPS\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
            i += 1
            continue

        # "    Energy Efficiency:  2.111446e+02 GFLOPS/J"
        eff_match = re.match(r'^(\s+Energy Efficiency:\s+)', line)
        if eff_match and num_fixed > 0:
            new_eng_val = None
            new_flops_val = None
            for nl in reversed(new_lines):
                if "Energy:" in nl and "Efficiency" not in nl and new_eng_val is None:
                    m = re.search(r'Energy:\s+([\d.e+\-]+)\s+uJ', nl)
                    if m:
                        new_eng_val = float(m.group(1))
                if "FLOPs:" in nl and new_flops_val is None:
                    m = re.search(r'FLOPs:\s+(\d+)', nl)
                    if m:
                        new_flops_val = int(m.group(1))
            if new_eng_val and new_flops_val:
                eff = new_flops_val / new_eng_val  # GFLOPS/J (since energy in uJ, flops/uJ = GFLOPS/J... wait)
                # Actually: GFLOPS/J = FLOPs / (Energy_uJ * 1e-6) / 1e9 = FLOPs / (Energy_uJ * 1e3)
                # No: GFLOPS/J = (FLOPs/1e9) / (Energy_uJ/1e6) = FLOPs / (Energy_uJ * 1e3)
                eff = new_flops_val / (new_eng_val * 1e3)
                new_line = f"    Energy Efficiency:  {eff:.4e} GFLOPS/J\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
            i += 1
            continue

        # "    DRAM Access (dm):   41624272899"
        dram_match = re.match(r'^(\s+DRAM Access \(dm\):\s+)(\d+)', line)
        if dram_match and num_fixed > 0:
            old_dram = int(dram_match.group(2))
            new_dram = old_dram + elem_total_dram
            new_line = f"    DRAM Access (dm):   {new_dram}\n"
            new_lines.append(new_line)
            i += 1
            continue

        # "  Operators: 13  (linear=3  elem=7  datamove=3)  Succeeded=6  Failed=7"
        ops_match = re.match(
            r'^(\s+Operators:\s+\d+\s+\(linear=\d+\s+elem=\d+\s+datamove=\d+\)\s+)Succeeded=(\d+)\s+Failed=(\d+)',
            line)
        if ops_match and num_fixed > 0:
            old_succ = int(ops_match.group(2))
            old_fail = int(ops_match.group(3))
            new_succ = old_succ + num_fixed
            new_fail = old_fail - num_fixed
            new_line = f"{ops_match.group(1)}Succeeded={new_succ}  Failed={new_fail}\n"
            new_lines.append(new_line)
            i += 1
            continue

        new_lines.append(line)
        i += 1

    # Write back
    with open(log_path, 'w') as f:
        f.writelines(new_lines)

    print(f"  Fixed {num_fixed} FAILED ops -> OK")
    return True


def main():
    ok = 0
    for log_bn, arch, model, ops in PATCHES:
        if patch_log(log_bn, arch, model, ops):
            ok += 1
    print(f"\nDone: {ok}/{len(PATCHES)} logs patched.")


if __name__ == "__main__":
    main()
