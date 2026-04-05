#!/usr/bin/env python3
"""
Generate per-operator detail CSV for PPU test results.
Includes MLIR element counts from logs and YAML problem dimensions.
"""

import os
import re
import csv
import yaml

BASE_DIR = "/home/accelgen/test_ppu"
LOGS_DIR = os.path.join(BASE_DIR, "logs")

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
    for p in LIN_PREFIXES:
        if op_name.startswith(p):
            return "linear"
    for p in ELEM_PREFIXES:
        if op_name.startswith(p):
            return "elementwise"
    for p in DM_PREFIXES:
        if op_name.startswith(p):
            return "datamove"
    return "datamove"


def parse_log_ops(log_file):
    """Parse log file to get per-op MLIR info (elems, etc.)."""
    ops = {}
    if not os.path.exists(log_file):
        return ops
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            # [ModelBaseline] Found transpose_0 (datamove) elems=16777216
            # [ModelBaseline] Found broadcast_2 (datamove) inputElems=32768 outputElems=32768 broadcastFactor(M)=1
            # [ModelBaseline] Found mulf_0 (arith.mulf, C=1)
            # [ModelBaseline] Found maximumf_reduction_10 (reduction, 3 ops/step)
            m = re.match(r'\[ModelBaseline\] Found (\S+) \((.+?)\)(.*)', line)
            if m:
                op_name = m.group(1)
                op_desc = m.group(2)
                extra = m.group(3).strip()
                info = {"mlir_desc": op_desc}

                # Parse elems
                em = re.search(r'elems=(\d+)', extra)
                if em:
                    info["mlir_elems"] = int(em.group(1))

                # Parse inputElems / outputElems / broadcastFactor
                em = re.search(r'inputElems=(\d+)', extra)
                if em:
                    info["mlir_input_elems"] = int(em.group(1))
                em = re.search(r'outputElems=(\d+)', extra)
                if em:
                    info["mlir_output_elems"] = int(em.group(1))
                em = re.search(r'broadcastFactor\(M\)=(\d+)', extra)
                if em:
                    info["mlir_broadcast_factor"] = int(em.group(1))

                ops[op_name] = info
    return ops


def parse_problem_yaml(yaml_file):
    """Parse problem.yaml to get dimensions N,C,M,R,S,P,Q,G."""
    dims = {}
    if not os.path.exists(yaml_file):
        return dims
    with open(yaml_file, "r") as f:
        content = f.read()

    # Extract dimensions from instance section
    in_instance = False
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('instance:'):
            in_instance = True
            continue
        if in_instance:
            m2 = re.match(r'\s+([A-Z]):\s*(\d+)', line)
            if m2:
                dims[m2.group(1)] = int(m2.group(2))
            elif stripped and not stripped.startswith('#'):
                # End of instance block if we hit a non-indented, non-empty line
                if not line.startswith('    '):
                    break
    return dims


def parse_stats(stats_file):
    """Parse timeloop-mapper.stats.txt."""
    result = {
        "cycles": 0,
        "energy_uj": 0.0,
        "computes": 0,
        "utilization": 0.0,
        "dram_acc": 0,
        "sram_acc": 0,
    }
    if not os.path.exists(stats_file):
        return None

    with open(stats_file, "r") as f:
        content = f.read()

    m = re.search(r"Summary Stats.*?Cycles:\s+(\d+)", content, re.DOTALL)
    if m:
        result["cycles"] = int(m.group(1))

    m = re.search(r"Summary Stats.*?Energy:\s+([\d.eE+\-]+)\s+uJ", content, re.DOTALL)
    if m:
        result["energy_uj"] = float(m.group(1))

    m = re.search(r"Summary Stats.*?Utilization:\s+([\d.]+)%", content, re.DOTALL)
    if m:
        result["utilization"] = float(m.group(1))

    m = re.search(r"Computes\s*=\s*(\d+)", content)
    if m:
        result["computes"] = int(m.group(1))

    dram_section = content.rfind("=== DRAM ===")
    if dram_section >= 0:
        after = content[dram_section:]
        m = re.search(r"Total scalar accesses\s*:\s*(\d+)", after)
        if m:
            result["dram_acc"] = int(m.group(1))

    gb_section = content.rfind("=== GlobalBuffer ===")
    if gb_section >= 0:
        after = content[gb_section:]
        m = re.search(r"Total scalar accesses\s*:\s*(\d+)", after)
        if m:
            result["sram_acc"] = int(m.group(1))

    return result


def parse_model_name(model_dir_name):
    m = re.match(r"^(.+?)-block\d+-(.+?)-(prefill|decode)-(.+)$", model_dir_name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return model_dir_name, "", ""


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def build_mlir_elems_str(op_name, log_info):
    """Build MLIR dimension string from log info."""
    if not log_info:
        return ""
    parts = []
    if "mlir_elems" in log_info:
        parts.append(str(log_info["mlir_elems"]))
    elif "mlir_input_elems" in log_info and "mlir_output_elems" in log_info:
        parts.append(f"in={log_info['mlir_input_elems']}")
        parts.append(f"out={log_info['mlir_output_elems']}")
        if "mlir_broadcast_factor" in log_info:
            parts.append(f"bcast={log_info['mlir_broadcast_factor']}")
    return " ".join(parts)


def main():
    all_rows = []

    for config in ["edge", "server"]:
        config_dir = os.path.join(BASE_DIR, config)
        if not os.path.isdir(config_dir):
            continue

        for model_dir in sorted(os.listdir(config_dir)):
            model_path = os.path.join(config_dir, model_dir)
            if not os.path.isdir(model_path):
                continue

            model, layer, phase = parse_model_name(model_dir)

            # Determine config suffix for log file
            log_file = os.path.join(LOGS_DIR, f"{model_dir}_{config}.log")
            log_ops = parse_log_ops(log_file)

            for op_dir in sorted(os.listdir(model_path), key=natural_sort_key):
                op_path = os.path.join(model_path, op_dir)
                if not os.path.isdir(op_path):
                    continue

                stats_file = os.path.join(op_path, "timeloop-mapper.stats.txt")
                stats = parse_stats(stats_file)
                if stats is None:
                    continue

                op_type = classify_op(op_dir)

                # MLIR info from log
                log_info = log_ops.get(op_dir, {})
                mlir_desc = log_info.get("mlir_desc", "")
                mlir_elems = build_mlir_elems_str(op_dir, log_info)

                # YAML problem dims
                yaml_file = os.path.join(op_path, "problem.yaml")
                dims = parse_problem_yaml(yaml_file)
                yaml_N = dims.get("N", "")
                yaml_C = dims.get("C", "")
                yaml_M = dims.get("M", "")
                yaml_R = dims.get("R", "")
                yaml_S = dims.get("S", "")
                yaml_P = dims.get("P", "")
                yaml_Q = dims.get("Q", "")
                yaml_G = dims.get("G", "")

                all_rows.append({
                    "Arch": "PPU",
                    "Config": config,
                    "Model": model,
                    "Layer": layer,
                    "Phase": phase,
                    "OpName": op_dir,
                    "OpType": op_type,
                    "MLIR_Desc": mlir_desc,
                    "MLIR_Elems": mlir_elems,
                    "YAML_N": yaml_N,
                    "YAML_C": yaml_C,
                    "YAML_M": yaml_M,
                    "YAML_R": yaml_R,
                    "YAML_S": yaml_S,
                    "YAML_P": yaml_P,
                    "YAML_Q": yaml_Q,
                    "YAML_G": yaml_G,
                    "Cycles": stats["cycles"],
                    "Energy(uJ)": f"{stats['energy_uj']:.4e}",
                    "Computes": stats["computes"],
                    "Utilization(%)": f"{stats['utilization']:.2f}",
                    "DRAM_Acc": stats["dram_acc"],
                    "SRAM_Acc": stats["sram_acc"],
                })

    out_file = os.path.join(BASE_DIR, "ppu_detail.csv")
    fields = [
        "Arch", "Config", "Model", "Layer", "Phase",
        "OpName", "OpType", "MLIR_Desc", "MLIR_Elems",
        "YAML_N", "YAML_C", "YAML_M", "YAML_R", "YAML_S", "YAML_P", "YAML_Q", "YAML_G",
        "Cycles", "Energy(uJ)", "Computes", "Utilization(%)",
        "DRAM_Acc", "SRAM_Acc",
    ]

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written {len(all_rows)} rows to {out_file}")

    # Print a few sample rows
    print("\nSample rows:")
    for r in all_rows[:5]:
        print(f"  {r['Config']:6} {r['Model']:10} {r['Layer']:10} {r['Phase']:8} "
              f"{r['OpName']:30} {r['OpType']:12} "
              f"MLIR={r['MLIR_Elems']:20} "
              f"YAML=N{r['YAML_N']}C{r['YAML_C']}M{r['YAML_M']}P{r['YAML_P']}Q{r['YAML_Q']}G{r['YAML_G']} "
              f"Cyc={r['Cycles']:>10}")


if __name__ == "__main__":
    main()
