#!/usr/bin/env python3
"""
Create a 3-sheet Excel:
  Sheet1: "Edge_Complete" - combined full workload per model on edge
          (PPU elem+datamove + gemmini linear + lego linear)
  Sheet2: "PPU_Detail" - raw PPU detail (edge only)
  Sheet3: "Gemmini_Lego_Edge" - raw gemmini+lego edge results
"""

import csv
import re
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

PPU_CSV = "/home/accelgen/test_ppu/ppu_detail.csv"
EDGE_CSV = "/home/accelgen/edge_results.csv"
OUT_XLSX = "/home/accelgen/test_ppu/edge_workload_summary.xlsx"

HEADER_FONT = Font(bold=True)
HEADER_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
THIN_BORDER = Border(
    left=Side(style='thin'), right=Side(style='thin'),
    top=Side(style='thin'), bottom=Side(style='thin')
)

def read_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames

def style_header(ws, ncols):
    for col in range(1, ncols+1):
        cell = ws.cell(row=1, column=col)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal='center')

def auto_width(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 3, 40)

def write_sheet_from_rows(ws, headers, rows):
    # Write header
    for c, h in enumerate(headers, 1):
        ws.cell(row=1, column=c, value=h)
    # Write data
    for r, row in enumerate(rows, 2):
        for c, h in enumerate(headers, 1):
            val = row.get(h, "")
            # Try numeric conversion
            try:
                if '.' in str(val) or 'e' in str(val).lower():
                    val = float(val)
                elif str(val).isdigit():
                    val = int(val)
            except (ValueError, TypeError):
                pass
            ws.cell(row=r, column=c, value=val)
    style_header(ws, len(headers))
    auto_width(ws)

def normalize_model(name):
    """Normalize model names: gemma_7b -> gemma-7b etc."""
    return name.replace("_", "-")

def normalize_layer(layer):
    """Normalize layer: attn_decode_0 -> attention, attn_prefill_0 -> attention, etc."""
    if "attn" in layer:
        return "attention"
    if "ffn" in layer:
        return "ffn"
    return layer

def build_combined_sheet(ppu_rows, edge_rows):
    """
    Build combined rows: for each (model, layer, phase) on edge,
    list all ops: linear from gemmini/lego + elementwise/datamove from PPU.
    """
    combined = []

    # Group PPU edge rows by (model, layer, phase)
    ppu_edge = [r for r in ppu_rows if r.get("Config") == "edge"]

    # Group edge_results by (arch, model, layer, phase)
    # edge_results has Layer like 'attn_decode_0', need to extract phase and layer
    gemmini_rows = []
    lego_rows = []
    for r in edge_rows:
        arch = r.get("Arch", "")
        layer_raw = r.get("Layer", "")
        # Parse layer: "attn_decode_0" -> layer=attention
        # Phase is already in Phase column
        if arch == "gemmini":
            gemmini_rows.append(r)
        elif arch == "lego":
            lego_rows.append(r)

    # Build combined: for each model×phase×layer combo,
    # gather PPU ops + gemmini ops + lego ops
    # Key: (model_normalized, layer, phase)

    # Collect all unique (model, layer, phase) from PPU edge
    model_combos = []
    seen = set()
    for r in ppu_edge:
        key = (r["Model"], r["Layer"], r["Phase"])
        if key not in seen:
            seen.add(key)
            model_combos.append(key)

    for model, layer, phase in sorted(model_combos):
        model_norm = normalize_model(model)

        # Add PPU elementwise+datamove ops
        for r in ppu_edge:
            if r["Model"] == model and r["Layer"] == layer and r["Phase"] == phase:
                row = dict(r)
                row["Source"] = "PPU_edge"
                combined.append(row)

        # Find matching gemmini ops
        for r in gemmini_rows:
            r_model = normalize_model(r.get("Model", ""))
            r_layer_raw = r.get("Layer", "")
            r_phase = r.get("Phase", "")
            r_layer = normalize_layer(r_layer_raw)
            if r_model == model_norm and r_layer == layer and r_phase == phase:
                row = dict(r)
                row["Source"] = "gemmini_edge"
                row["Model"] = model  # normalize
                row["Layer"] = layer
                row["OpType"] = "linear"
                combined.append(row)

        # Find matching lego ops
        for r in lego_rows:
            r_model = normalize_model(r.get("Model", ""))
            r_layer_raw = r.get("Layer", "")
            r_phase = r.get("Phase", "")
            r_layer = normalize_layer(r_layer_raw)
            if r_model == model_norm and r_layer == layer and r_phase == phase:
                row = dict(r)
                row["Source"] = "lego_edge"
                row["Model"] = model
                row["Layer"] = layer
                row["OpType"] = "linear"
                combined.append(row)

    return combined


def main():
    ppu_rows, ppu_headers = read_csv(PPU_CSV)
    edge_rows, edge_headers = read_csv(EDGE_CSV)

    wb = Workbook()

    # Sheet1: Edge Complete
    ws1 = wb.active
    ws1.title = "Edge_Complete"
    combined = build_combined_sheet(ppu_rows, edge_rows)
    combined_headers = ["Source", "Config", "Model", "Layer", "Phase",
                        "OpName", "OpType",
                        "Cycles", "Energy(uJ)", "Computes", "Utilization(%)",
                        "DRAM_Acc", "SRAM_Acc",
                        "MLIR_Desc", "MLIR_Elems",
                        "YAML_N", "YAML_C", "YAML_M", "YAML_R", "YAML_S",
                        "YAML_P", "YAML_Q", "YAML_G"]
    write_sheet_from_rows(ws1, combined_headers, combined)

    # Sheet2: PPU_Detail (edge only)
    ws2 = wb.create_sheet("PPU_Edge_Detail")
    ppu_edge_rows = [r for r in ppu_rows if r.get("Config") == "edge"]
    write_sheet_from_rows(ws2, ppu_headers, ppu_edge_rows)

    # Sheet3: Gemmini_Lego_Edge
    ws3 = wb.create_sheet("Gemmini_Lego_Edge")
    write_sheet_from_rows(ws3, edge_headers, edge_rows)

    wb.save(OUT_XLSX)
    print(f"Saved to {OUT_XLSX}")
    print(f"  Sheet1 'Edge_Complete': {len(combined)} rows")
    print(f"  Sheet2 'PPU_Edge_Detail': {len(ppu_edge_rows)} rows")
    print(f"  Sheet3 'Gemmini_Lego_Edge': {len(edge_rows)} rows")

if __name__ == "__main__":
    main()
