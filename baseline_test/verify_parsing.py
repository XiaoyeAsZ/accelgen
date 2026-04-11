#!/usr/bin/env python3
"""Verify batch.py parsing against raw C++ log output for all operators."""
import os, re, glob

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
errors = []
total_ops = 0
checked_files = 0

def parse_raw_tab_fields(line):
    """Parse a C++ output line using tab as primary separator.
    Format: <padded SSA+OpName block> Cycles\tEnergy\t...
    The SSA+OpName block is space-padded (42 chars total), followed by tab-separated fields.
    """
    # Split on tabs first
    tab_parts = line.split('\t')
    if len(tab_parts) < 2:
        return None
    # First tab_part contains the space-padded SSA + OpName + Cycles
    # The Cycles value is the last whitespace-separated token before the first tab
    prefix_tokens = tab_parts[0].split()
    if len(prefix_tokens) < 2:
        return None
    return prefix_tokens, tab_parts

for logfile in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
    basename = os.path.basename(logfile)
    with open(logfile) as f:
        content = f.read()

    checked_files += 1

    # ── Verify Linear Ops ──
    lin_section = re.search(
        r"--- Linear Ops.*?\n-+\n(.*?)\n-+\n", content, re.DOTALL
    )
    if lin_section:
        for line in lin_section.group(1).strip().split("\n"):
            tab_parts = line.split('\t')
            if len(tab_parts) < 5:
                continue
            prefix_tokens = tab_parts[0].split()
            if len(prefix_tokens) < 3:
                continue
            # C++ format: SSA OpName Cycles\tEnergy\tComputes\tUtil%\tDRAM_Acc
            if prefix_tokens[-1] == "FAILED":
                continue

            raw_ssa = prefix_tokens[0]
            raw_name = prefix_tokens[1]
            raw_cycles = int(float(prefix_tokens[2]))  # last token before tab
            raw_energy = float(tab_parts[1])
            raw_dram = int(float(tab_parts[4])) if len(tab_parts) >= 5 else 0

            # Now simulate batch.py parsing: parts = line.split()
            parts = line.split()
            if len(parts) < 4 or parts[2] == "FAILED":
                continue
            py_ssa = parts[0]
            py_name = parts[1]
            py_cycles = int(float(parts[2]))
            py_energy = float(parts[3])
            py_dram = int(float(parts[-1])) if len(parts) >= 7 else 0

            total_ops += 1
            if py_cycles != raw_cycles:
                errors.append(f"[LINEAR] {basename} {raw_ssa} {raw_name}: Cycles mismatch: parsed={py_cycles} raw={raw_cycles}")
            if abs(py_energy - raw_energy) > 0.01:
                errors.append(f"[LINEAR] {basename} {raw_ssa} {raw_name}: Energy mismatch: parsed={py_energy} raw={raw_energy}")
            if py_dram != raw_dram:
                errors.append(f"[LINEAR] {basename} {raw_ssa} {raw_name}: DRAM mismatch: parsed={py_dram} raw={raw_dram}")

    # ── Verify Elementwise Ops ──
    elem_section = re.search(
        r"--- Elementwise Ops.*?\n-+\n(.*?)\n-+\n", content, re.DOTALL
    )
    if elem_section:
        for line in elem_section.group(1).strip().split("\n"):
            tab_parts = line.split('\t')
            if len(tab_parts) < 4:
                continue
            prefix_tokens = tab_parts[0].split()
            if len(prefix_tokens) < 3:
                continue
            if prefix_tokens[-1] == "FAILED":
                continue

            raw_ssa = prefix_tokens[0]
            raw_name = prefix_tokens[1]
            raw_cycles = int(float(prefix_tokens[2]))
            raw_energy = float(tab_parts[1])

            parts = line.split()
            if len(parts) < 4 or parts[2] == "FAILED":
                continue
            py_cycles = int(float(parts[2]))
            py_energy = float(parts[3])

            total_ops += 1
            if py_cycles != raw_cycles:
                errors.append(f"[ELEM] {basename} {raw_ssa} {raw_name}: Cycles mismatch: parsed={py_cycles} raw={raw_cycles}")
            if abs(py_energy - raw_energy) > 0.01:
                errors.append(f"[ELEM] {basename} {raw_ssa} {raw_name}: Energy mismatch: parsed={py_energy} raw={raw_energy}")

    # ── Verify Data Movement Ops ──
    dm_section = re.search(
        r"--- Data Movement Ops.*?\n-+\n(.*?)\n-+\n", content, re.DOTALL
    )
    if dm_section:
        for line in dm_section.group(1).strip().split("\n"):
            tab_parts = line.split('\t')
            if len(tab_parts) < 5:
                continue
            prefix_tokens = tab_parts[0].split()
            if len(prefix_tokens) < 2:
                continue
            if prefix_tokens[-1] == "FAILED":
                continue

            # C++ format: SSA(pad11) OpName(pad31) Cycles\tEnergy\tMemEnergy\tMacEnergy\tDRAM_Acc
            # prefix_tokens[-1] is always Cycles (last space-separated token before first tab)
            raw_cycles = int(float(prefix_tokens[-1]))
            raw_energy = float(tab_parts[1])
            raw_dram = int(float(tab_parts[4])) if len(tab_parts) >= 5 else 0

            # Determine raw SSA and OpName from tab-split prefix
            # When not merged: prefix has [SSA, OpName, Cycles] (3+ tokens)
            # When merged: prefix has [SSA+OpName, Cycles] (2 tokens)
            if len(prefix_tokens) >= 3:
                raw_ssa = prefix_tokens[0]
                raw_name = prefix_tokens[1]
            else:
                raw_ssa = prefix_tokens[0]
                raw_name = "(merged)"

            # Now simulate batch.py's FIXED parsing
            parts = line.split()
            if len(parts) < 4:
                continue
            if len(parts) == 7:
                py_cycles = int(float(parts[2]))
                py_energy = float(parts[3])
            elif len(parts) == 6:
                py_cycles = int(float(parts[1]))
                py_energy = float(parts[2])
            else:
                continue
            py_dram = int(float(parts[-1]))

            total_ops += 1
            if py_cycles != raw_cycles:
                errors.append(f"[DM] {basename} {raw_ssa} {raw_name}: Cycles mismatch: parsed={py_cycles} raw={raw_cycles}")
            if abs(py_energy - raw_energy) > 0.01:
                errors.append(f"[DM] {basename} {raw_ssa} {raw_name}: Energy mismatch: parsed={py_energy} raw={raw_energy}")
            if py_dram != raw_dram:
                errors.append(f"[DM] {basename} {raw_ssa} {raw_name}: DRAM mismatch: parsed={py_dram} raw={raw_dram}")

print(f"Checked {checked_files} log files, {total_ops} operators total.")
if errors:
    print(f"\n❌ Found {len(errors)} mismatches:")
    for e in errors:
        print(f"  {e}")
else:
    print("✅ All operators match perfectly!")
