#!/usr/bin/env python3
"""
Generate per-operator detail CSV for PPU test results.
Each operator gets its own row, with MLIR and YAML dimension info at the end.
"""

import os
import re
import csv

BASE_DIR = "/home/accelgen/test_ppu"
LOGS_DIR = os.path.join(BASE_DIR, "logs")

ELEM_PREFIXES = [
    "addf_reduction_", "maximumf_reduction_", "minimumf_reduction_",
    "mulf_", "addf_", "subf_", "divf_", "negf_", "exp_", "maximumf_",
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


def parse_stats(stats_file):
    result = {
        "cycles": 0, "energy_uj": 0.0, "computes": 0,
        "utilization": 0.0, "dram_acc": 0, "sram_acc": 0,
    }
    if not os.path.exists(stats_file):
        return None
    with open(stats_file, "r") as f:
        content = f.read()
    m = re.search(r"Summary Stats.*?Cycles:\s+(\d+)", content, re.DOTALL)
    if m: result["cycles"] = int(m.group(1))
    m = re.search(r"Summary Stats.*?Energy:\s+([\d.eE+\-]+)\s+uJ", content, re.DOTALL)
    if m: result["energy_uj"] = float(m.group(1))
    m = re.search(r"Summary Stats.*?Utilization:\s+([\d.]+)%", content, re.DOTALL)
    if m: result["utilization"] = float(m.group(1))
    m = re.search(r"Computes\s*=\s*(\d+)", content)
    if m: result["computes"] = int(m.group(1))
    dram_section = content.rfind("=== DRAM ===")
    if dram_section >= 0:
        after = content[dram_section:]
        m = re.search(r"Total scalar accesses\s*:\s*(\d+)", after)
        if m: result["dram_acc"] = int(m.group(1))
    gb_section = content.rfind("=== GlobalBuffer ===")
    if gb_section >= 0:
        after = content[gb_section:]
        m = re.search(r"Total scalar accesses\s*:\s*(\d+)", after)
        if m: result["sram_acc"] = int(m.group(1))
    return result


def parse_problem_yaml(yaml_file):
    dims = {}
    if not os.path.exists(yaml_file):
        return dims
    with open(yaml_file, "r") as f:
        for line in f:
            m = re.match(r"\s+([NCMRSPQG]):\s*(\d+)", line)
            if m:
                dims[m.group(1)] = int(m.group(2))
    return dims


def parse_log_mlir_info(log_file):
    op_info = {}
    if not os.path.exists(log_file):
        return op_info
    with open(log_file, "r") as f:
        raw = f.read()
    # Find all [ModelBaseline] Found lines, handling line wraps
    for m in re.finditer(r"\[ModelBaseline\] Found (\S+) (.+?)(?=\n\[|\n===|\n$|\Z)", raw, re.DOTALL):
        op_name = m.group(1)
        rest = m.group(2).replace("\n", " ").strip()
        # Extract elems
        elems = ""
        em = re.search(r"(?:outputElems|elems)=(\d+)", rest)
        if em:
            elems = em.group(1)
        op_info[op_name] = {"mlir_desc": rest, "mlir_elems": elems}
    return op_info


def parse_model_name(name):
    m = re.match(r"^(.+?)-block\d+-(.+?)-(prefill|decode)-(.+)$", name)
    if m: return m.group(1), m.group(2), m.group(3)
    return name, "", ""


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


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
            log_file = os.path.join(LOGS_DIR, f"{model_dir}_{config}.log")
            mlir_info = parse_log_mlir_info(log_file)

            for op_dir in sorted(os.listdir(model_path), key=natural_sort_key):
                op_path = os.path.join(model_path, op_dir)
                if not os.path.isdir(op_path):
                    continue
                stats_file = os.path.join(op_path, "timeloop-mapper.stats.txt")
                stats = parse_stats(stats_file)
                if stats is None:
                    continue
                op_type = classify_op(op_dir)
                mi = mlir_info.get(op_dir, {})
                mlir_desc = mi.get("mlir_desc", "")
                mlir_elems = mi.get("mlir_elems", "")
                yaml_file = os.path.join(op_path, "problem.yaml")
                dims = parse_problem_yaml(yaml_file)

                all_rows.append({
                    "Arch": "PPU",
                    "Config": config,
                    "Model": model,
                    "Layer": layer,
                    "Phase": phase,
                    "OpName": op_dir,
                    "OpType": op_type,
                    "Cycles": stats["cycles"],
                    "Energy(uJ)": f"{stats['energy_uj']:.4e}",
                    "Computes": stats["computes"],
                    "Utilization(%)": f"{stats['utilization']:.2f}",
                    "DRAM_Acc": stats["dram_acc"],
                    "SRAM_Acc": stats["sram_acc"],
                    "MLIR_Desc": mlir_desc,
                    "MLIR_Elems": mlir_elems,
                    "YAML_N": dims.get("N", ""),
                    "YAML_C": dims.get("C", ""),
                    "YAML_M": dims.get("M", ""),
                    "YAML_R": dims.get("R", ""),
                    "YAML_S": dims.get("S", ""),
                    "YAML_P": dims.get("P", ""),
                    "YAML_Q": dims.get("Q", ""),
                    "YAML_G": dims.get("G", ""),
                })

    out_file = os.path.join(BASE_DIR, "ppu_detail.csv")
    fields = [
        "Arch", "Config", "Model", "Layer", "Phase",
        "OpName", "OpType",
        "Cycles", "Energy(uJ)", "Computes", "Utilization(%)",
        "DRAM_Acc", "SRAM_Acc",
        "MLIR_Desc", "MLIR_Elems",
        "YAML_N", "YAML_C", "YAML_M", "YAML_R", "YAML_S",
        "YAML_P", "YAML_Q", "YAML_G",
    ]
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written {len(all_rows)} rows to {out_file}")
    has_yaml = sum(1 for r in all_rows if r["YAML_N"] != "")
    has_mlir = sum(1 for r in all_rows if r["MLIR_Desc"] != "")
    has_elems = sum(1 for r in all_rows if r["MLIR_Elems"] != "")
    print(f"  YAML dims: {has_yaml}/{len(all_rows)}")
    print(f"  MLIR desc: {has_mlir}/{len(all_rows)}")
    print(f"  MLIR elems: {has_elems}/{len(all_rows)}")

if __name__ == "__main__":
    main()
