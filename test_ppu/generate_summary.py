#!/usr/bin/env python3
"""
Generate summary CSV for PPU test results.
Columns: Arch, Config, Model, Layer, Phase,
         Lin_Cycles, Lin_Energy(uJ),
         Elem_Cycles, Elem_Energy(uJ),
         DM_Cycles, DM_Energy(uJ), DM_DRAM_Acc, DM_SRAM_Acc,
         Total_Cycles, Total_Energy(uJ)
"""

import os
import re
import csv
import sys

BASE_DIR = "/home/accelgen/test_ppu"
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Op name patterns to classify
ELEM_PREFIXES = [
    "mulf_", "addf_", "subf_", "divf_", "negf_", "exp_", "maximumf_",
    "addf_reduction_", "maximumf_reduction_", "minimumf_reduction_",
    "sqrt_", "rsqrt_", "erf_", "fpowi_", "absf_", "ceilf_",
    "floorf_", "roundf_", "truncf_", "log_", "tanh_",
]
DM_PREFIXES = [
    "transpose_", "broadcast_", "collapse_shape_", "expand_shape_",
    "fill_", "extract_slice_", "concat_", "type_cast_",
]
LIN_PREFIXES = [
    "batch_matmul_",
]


def classify_op(op_name):
    """Classify op as 'elem', 'dm', or 'lin'."""
    for p in LIN_PREFIXES:
        if op_name.startswith(p):
            return "lin"
    for p in ELEM_PREFIXES:
        if op_name.startswith(p):
            return "elem"
    for p in DM_PREFIXES:
        if op_name.startswith(p):
            return "dm"
    # fallback: try to guess from name
    return "dm"  # default to datamove for unknown


def parse_stats(stats_file):
    """Parse a timeloop-mapper.stats.txt file.
    Returns dict with cycles, energy_uj, dram_acc, sram_acc.
    """
    result = {
        "cycles": 0,
        "energy_uj": 0.0,
        "dram_acc": 0,
        "sram_acc": 0,
    }
    if not os.path.exists(stats_file):
        return None

    with open(stats_file, "r") as f:
        content = f.read()

    # Extract from Summary Stats section
    # Cycles: 524288
    m = re.search(r"Summary Stats.*?Cycles:\s+(\d+)", content, re.DOTALL)
    if m:
        result["cycles"] = int(m.group(1))

    # Energy: 9760.01 uJ
    m = re.search(r"Summary Stats.*?Energy:\s+([\d.eE+\-]+)\s+uJ", content, re.DOTALL)
    if m:
        result["energy_uj"] = float(m.group(1))

    # DRAM total scalar accesses - get from Operational Intensity section
    # === DRAM ===
    #     Total scalar accesses  : 150994952
    dram_section = content.rfind("=== DRAM ===")
    if dram_section >= 0:
        after = content[dram_section:]
        m = re.search(r"Total scalar accesses\s*:\s*(\d+)", after)
        if m:
            result["dram_acc"] = int(m.group(1))

    # GlobalBuffer (SRAM) total scalar accesses - get from Operational Intensity section
    gb_section = content.rfind("=== GlobalBuffer ===")
    if gb_section >= 0:
        after = content[gb_section:]
        m = re.search(r"Total scalar accesses\s*:\s*(\d+)", after)
        if m:
            result["sram_acc"] = int(m.group(1))

    return result


def parse_model_name(model_dir_name):
    """Parse model directory name into (model, layer, phase).
    e.g. 'llama3-8b-block0-attention-prefill-b8s1024'
    -> model='llama3-8b', layer='attention', phase='prefill-b8s1024'
    """
    # Pattern: <model>-block<N>-<layer>-<phase>
    m = re.match(r"^(.+?)-block\d+-(.+?)-(prefill|decode)-(.+)$", model_dir_name)
    if m:
        model = m.group(1)
        layer = m.group(2)
        phase = m.group(3)
        return model, layer, phase
    return model_dir_name, "", ""


def process_config(arch, config):
    """Process one config (edge or server) and return rows."""
    config_dir = os.path.join(BASE_DIR, config)
    if not os.path.isdir(config_dir):
        return []

    rows = []
    for model_dir in sorted(os.listdir(config_dir)):
        model_path = os.path.join(config_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        model, layer, phase = parse_model_name(model_dir)

        lin_cycles = 0
        lin_energy = 0.0
        elem_cycles = 0
        elem_energy = 0.0
        dm_cycles = 0
        dm_energy = 0.0
        dm_dram_acc = 0
        dm_sram_acc = 0

        for op_dir in sorted(os.listdir(model_path)):
            op_path = os.path.join(model_path, op_dir)
            if not os.path.isdir(op_path):
                continue

            stats_file = os.path.join(op_path, "timeloop-mapper.stats.txt")
            stats = parse_stats(stats_file)
            if stats is None:
                continue

            op_type = classify_op(op_dir)

            if op_type == "lin":
                lin_cycles += stats["cycles"]
                lin_energy += stats["energy_uj"]
            elif op_type == "elem":
                elem_cycles += stats["cycles"]
                elem_energy += stats["energy_uj"]
            else:  # dm
                dm_cycles += stats["cycles"]
                dm_energy += stats["energy_uj"]
                dm_dram_acc += stats["dram_acc"]
                dm_sram_acc += stats["sram_acc"]

        total_cycles = lin_cycles + elem_cycles + dm_cycles
        total_energy = lin_energy + elem_energy + dm_energy

        rows.append({
            "Arch": arch,
            "Config": config,
            "Model": model,
            "Layer": layer,
            "Phase": phase,
            "Lin_Cycles": lin_cycles,
            "Lin_Energy(uJ)": f"{lin_energy:.4e}" if lin_energy > 0 else "0",
            "Elem_Cycles": elem_cycles,
            "Elem_Energy(uJ)": f"{elem_energy:.4e}" if elem_energy > 0 else "0",
            "DM_Cycles": dm_cycles,
            "DM_Energy(uJ)": f"{dm_energy:.4e}" if dm_energy > 0 else "0",
            "DM_DRAM_Acc": dm_dram_acc,
            "DM_SRAM_Acc": dm_sram_acc,
            "Total_Cycles": total_cycles,
            "Total_Energy(uJ)": f"{total_energy:.4e}" if total_energy > 0 else "0",
        })

    return rows


def main():
    all_rows = []

    # PPU_edge
    all_rows.extend(process_config("PPU", "edge"))
    # PPU_server
    all_rows.extend(process_config("PPU", "server"))

    out_file = os.path.join(BASE_DIR, "ppu_summary.csv")
    fields = [
        "Arch", "Config", "Model", "Layer", "Phase",
        "Lin_Cycles", "Lin_Energy(uJ)",
        "Elem_Cycles", "Elem_Energy(uJ)",
        "DM_Cycles", "DM_Energy(uJ)", "DM_DRAM_Acc", "DM_SRAM_Acc",
        "Total_Cycles", "Total_Energy(uJ)",
    ]

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written {len(all_rows)} rows to {out_file}")

    # Also print to stdout
    with open(out_file, "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
