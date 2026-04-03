#!/usr/bin/env python3
"""Collect and summarize all baseline test results.
Usage: python3 script/collect_results.py
"""
import os, re, csv

LOG_DIR  = "baseline_test/logs"
OUT_DIR  = "baseline_test"

MODELS = [
    "llama3-8b-block0-attention-prefill-b8s1024",
    "llama3-8b-block0-attention-decode-b8s1024",
    "llama3-8b-block0-ffn-prefill-b8s1024",
    "llama3-8b-block0-ffn-decode-b8s1024",
    "gemma-7b-block0-attention-prefill-b8s1024",
    "gemma-7b-block0-attention-decode-b8s1024",
    "gemma-7b-block0-ffn-prefill-b8s1024",
    "gemma-7b-block0-ffn-decode-b8s1024",
    "qwen3-8b-block0-attention-prefill-b8s1024",
    "qwen3-8b-block0-attention-decode-b8s1024",
    "qwen3-8b-block0-ffn-prefill-b8s1024",
    "qwen3-8b-block0-ffn-decode-b8s1024",
]
ARCHS = ["simba_edge", "simba_server",
         "gemmini_edge", "gemmini_server",
         "lego_edge", "lego_server"]

os.chdir("/home/accelgen")

# ── parse one log ────────────────────────────────────────────────────────────
def parse_log(path):
    """Return (status, total_ops, succ, fail, [op_dicts])"""
    if not os.path.isfile(path):
        return ("NOT_RUN", 0, 0, 0, [])
    text = open(path).read()
    m = re.search(r"Total operators:\s*(\d+)\s+Succeeded:\s*(\d+)\s+Failed:\s*(\d+)", text)
    if not m:
        return ("INCOMPLETE", 0, 0, 0, [])

    total_ops, succ, fail = int(m[1]), int(m[2]), int(m[3])
    status = "OK" if fail == 0 else "PARTIAL_FAIL"

    ops = []
    lines = text.split("\n")
    in_table = False
    for line in lines:
        if re.match(r"^-{20,}", line):
            in_table = not in_table
            continue
        if not in_table:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        if parts[1] == "TOTAL":
            continue
        if "FAILED" in parts:
            # Handle concatenated arch+type for FAILED lines too
            arch_field = parts[2]
            typ_field = parts[3] if len(parts) > 3 else "unknown"
            for suffix in ("linear", "elementwise"):
                if arch_field.endswith(suffix) and len(arch_field) > len(suffix):
                    typ_field = suffix
                    arch_field = arch_field[:-len(suffix)]
                    break
            ops.append(dict(ssa=parts[0], name=parts[1], arch=arch_field,
                            typ=typ_field, cyc=0, eng=0.0, comp=0, util=0.0,
                            ok=False))
            continue
        # Handle two formats:
        # (A) separated:  %5  batch_matmul_0  simba_edge  linear  33,554,432  3.698e+06  137,438,953,472  100.00%
        # (B) concatenated: %5  batch_matmul_0  simba_edge_nonoclinear  134217728  1.523e+06  137438953472  2.5e+01%
        arch_field = parts[2]
        concat = False
        for suffix in ("linear", "elementwise"):
            if arch_field.endswith(suffix) and len(arch_field) > len(suffix):
                typ_field = suffix
                arch_field = arch_field[:-len(suffix)]
                concat = True
                break
        if concat:
            # Format (B): parts = [ssa, name, arch+type, cyc, eng, comp, util%, dram, sram]
            if len(parts) >= 7:
                try:
                    cyc  = int(parts[3].replace(",", ""))
                    eng  = float(parts[4])
                    comp = int(parts[5].replace(",", ""))
                    util = float(parts[6].rstrip("%"))
                    ops.append(dict(ssa=parts[0], name=parts[1], arch=arch_field,
                                    typ=typ_field, cyc=cyc, eng=eng, comp=comp,
                                    util=util, ok=True))
                except ValueError:
                    pass
        else:
            # Format (A): parts = [ssa, name, arch, type, cyc, eng, comp, util%, dram, sram]
            typ_field = parts[3]
            if len(parts) >= 8:
                try:
                    cyc  = int(parts[4].replace(",", ""))
                    eng  = float(parts[5])
                    comp = int(parts[6].replace(",", ""))
                    util = float(parts[7].rstrip("%"))
                    ops.append(dict(ssa=parts[0], name=parts[1], arch=arch_field,
                                    typ=typ_field, cyc=cyc, eng=eng, comp=comp,
                                    util=util, ok=True))
                except ValueError:
                    pass
    return (status, total_ops, succ, fail, ops)

# ── collect ──────────────────────────────────────────────────────────────────
rows = []        # summary rows
detail_rows = [] # per-op rows

for arch in ARCHS:
    for model in MODELS:
        logf = os.path.join(LOG_DIR, f"{model}_{arch}.log")
        status, tops, succ, fail, ops = parse_log(logf)

        tc = te = tcomp = wu = 0     # total
        lc = le = lcomp = lw = 0     # linear
        ec = ee = ecomp = ew = 0     # elementwise

        for o in ops:
            detail_rows.append((model, arch, o))
            if not o["ok"]:
                continue
            c, e, comp, u = o["cyc"], o["eng"], o["comp"], o["util"]
            tc += c;  te += e;  tcomp += comp;  wu += c * u
            if o["typ"] == "linear":
                lc += c; le += e; lcomp += comp; lw += c * u
            else:
                ec += c; ee += e; ecomp += comp; ew += c * u

        avg  = wu / tc if tc > 0 else 0.0
        lutil = lw / lc if lc > 0 else 0.0
        eutil = ew / ec if ec > 0 else 0.0

        rows.append(dict(model=model, arch=arch, status=status,
                         tops=tops, succ=succ, fail=fail,
                         tc=tc, te=te, tcomp=tcomp, avg=avg,
                         lc=lc, le=le, lcomp=lcomp, lu=lutil,
                         ec=ec, ee=ee, ecomp=ecomp, eu=eutil))

# ── write summary CSV ────────────────────────────────────────────────────────
with open(f"{OUT_DIR}/summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model","Arch","Status","Ops","Succ","Fail",
                "TotalCycles","TotalEnergy(uJ)","TotalComputes","AvgUtil(%)",
                "LinCycles","LinEnergy(uJ)","LinComputes","LinUtil(%)",
                "ElemCycles","ElemEnergy(uJ)","ElemComputes","ElemUtil(%)"])
    for r in rows:
        w.writerow([r["model"], r["arch"], r["status"], r["tops"], r["succ"], r["fail"],
                    r["tc"], f'{r["te"]:.4e}', r["tcomp"], f'{r["avg"]:.2f}',
                    r["lc"], f'{r["le"]:.4e}', r["lcomp"], f'{r["lu"]:.2f}',
                    r["ec"], f'{r["ee"]:.4e}', r["ecomp"], f'{r["eu"]:.2f}'])

# ── write detail CSV ─────────────────────────────────────────────────────────
with open(f"{OUT_DIR}/summary_detail.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model","Arch","SSA","OpName","OpArch","OpType",
                "Cycles","Energy(uJ)","Computes","Util(%)"])
    for model, arch, o in detail_rows:
        if o["ok"]:
            w.writerow([model, arch, o["ssa"], o["name"], o["arch"], o["typ"],
                        o["cyc"], f'{o["eng"]:.4e}', o["comp"], f'{o["util"]:.2f}'])
        else:
            w.writerow([model, arch, o["ssa"], o["name"], o["arch"], o["typ"],
                        "FAILED","FAILED","FAILED","FAILED"])

# ── print detailed tables ────────────────────────────────────────────────────
W = 145  # table width matching C++ output
out = []
def p(s=""): out.append(s); print(s)

# ── Part 1: Per-model detailed tables (same format as C++ output) ────────────
for arch in ARCHS:
    for r in rows:
        if r["arch"] != arch:
            continue
        model = r["model"]
        status = r["status"]

        if status in ("NOT_RUN", "INCOMPLETE"):
            p(f"\n{'='*W}")
            p(f"  {model}  |  {arch}  —  {status}")
            p(f"{'='*W}")
            continue

        # Find ops for this model+arch
        ops = [o for m, a, o in detail_rows if m == model and a == arch]

        p(f"\n{'='*W}")
        p(f"  ModelBaselineAccelerator — Timeloop Results Summary")
        p(f"  MLIR file:    benchmark/mlir/{model}.mlir")
        p(f"  Linear arch:  {arch}")
        p(f"{'='*W}")

        p(f"{'SSA':<11}{'Operator':<31}{'Arch':<12}{'Type':<13}"
          f"{'Cycles':>14}{'Energy(uJ)':>14}{'Computes':>16}{'Util%':>16}")
        p("-" * W)

        for o in ops:
            if o["ok"]:
                p(f"{o['ssa']:<11}{o['name']:<31}{o['arch']:<12}{o['typ']:<13}"
                  f"{o['cyc']:>14,}{o['eng']:>14.6e}{o['comp']:>16,}"
                  f"{o['util']:>15.2f}%")
            else:
                p(f"{o['ssa']:<11}{o['name']:<31}{o['arch']:<12}{o['typ']:<13}"
                  f"{'FAILED':>14}{'FAILED':>14}{'FAILED':>16}{'N/A':>16}")

        # Compute DRAM/SRAM totals for this model
        p("-" * W)
        p(f"{'':11}{'TOTAL':<31}{'':12}{'':13}"
          f"{r['tc']:>14,}{r['te']:>14.6e}{r['tcomp']:>16,}")
        p(f"{'='*W}")
        p(f"  Total operators: {r['tops']}  Succeeded: {r['succ']}  Failed: {r['fail']}"
          f"  |  AvgUtil: {r['avg']:.1f}%  LinUtil: {r['lu']:.1f}%  ElemUtil: {r['eu']:.1f}%")
        p(f"{'='*W}")

# ── Part 2: Grand summary table ─────────────────────────────────────────────
SEP = "─" * 190
p(f"\n\n{'='*190}")
p("  GRAND SUMMARY — All Models × All Architectures")
p(f"{'='*190}\n")
p(f"{'Model':<52} {'Arch':<20} {'St':>7} {'Ops':>4} {'Fail':>4} "
  f"{'TotalCycles':>14} {'Energy(uJ)':>14} {'Computes':>16} | "
  f"{'AvgUtl':>7} {'LinUtl':>7} {'ElmUtl':>7}")
p(SEP)

for arch in ARCHS:
    ar = [r for r in rows if r["arch"] == arch]
    for r in ar:
        p(f"{r['model']:<52} {r['arch']:<20} {r['status']:>7} {r['tops']:>4} {r['fail']:>4} "
          f"{r['tc']:>14,} {r['te']:>14.4e} {r['tcomp']:>16,} | "
          f"{r['avg']:>6.1f}% {r['lu']:>6.1f}% {r['eu']:>6.1f}%")

    ac = sum(r["tc"] for r in ar)
    ae = sum(r["te"] for r in ar)
    acomp = sum(r["tcomp"] for r in ar)
    awu = sum(r["tc"]*r["avg"] for r in ar)
    alc = sum(r["lc"] for r in ar)
    alw = sum(r["lc"]*r["lu"] for r in ar)
    aec = sum(r["ec"] for r in ar)
    aew = sum(r["ec"]*r["eu"] for r in ar)
    aa = awu/ac if ac else 0; al = alw/alc if alc else 0; ae2 = aew/aec if aec else 0

    p(f"{'  >>> ARCH TOTAL':<52} {arch:<20} {'':>7} {'':>4} {'':>4} "
      f"{ac:>14,} {ae:>14.4e} {acomp:>16,} | "
      f"{aa:>6.1f}% {al:>6.1f}% {ae2:>6.1f}%")
    p(SEP)

p(f"\n{'='*190}")

with open(f"{OUT_DIR}/summary.txt", "w") as f:
    f.write("\n".join(out) + "\n")

print(f"\nSaved: {OUT_DIR}/summary.csv  {OUT_DIR}/summary_detail.csv  {OUT_DIR}/summary.txt")
