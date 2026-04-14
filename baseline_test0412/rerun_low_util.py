#!/usr/bin/env python3
"""
Re-run low-utilization PPU elementwise ops with:
1. exp_1 (llama3): P-fold P:4096→512, Q:14336→114688
2. sqrt_0 (gemma): re-run with victory_condition=500
3. erf_2 (gemma): re-run with victory_condition=500

All 3 share same PPU_server, so run once and copy results to 3 archs.
"""

import os, re, yaml, subprocess, shutil

BASE = "/home/accelgen/baseline_test0412"
ARCHS = ["lego_server", "gemmini_os_server", "gemmini_ws_server"]

# Define ops to rerun: (model, op, new_P, new_Q) or None for P/Q if no change
OPS_TO_RERUN = [
    ("llama3-8b", "exp_1", 512, 114688),    # P-fold: 4096→512, Q: 14336→114688
    ("gemma-7b", "sqrt_0", None, None),      # just re-run with higher victory_condition
    ("gemma-7b", "erf_2", None, None),       # just re-run with higher victory_condition
]

def modify_and_run(model, op, new_p, new_q):
    """Modify parsed-processed-input.yaml and run timeloop-mapper."""
    # Use lego_server as the reference directory
    ref_dir = os.path.join(BASE, "lego_server", model, "ffn/prefill/b8s4096", op)
    input_yaml = os.path.join(ref_dir, "parsed-processed-input.yaml")
    
    if not os.path.isfile(input_yaml):
        print(f"ERROR: {input_yaml} not found")
        return False
    
    # Read the full config as TEXT (not YAML parse, to preserve quoting)
    with open(input_yaml) as f:
        content = f.read()
    
    # Extract current P and Q via regex
    m_p = re.search(r'(instance:\s*\n(?:.*\n)*?\s+P:\s*)(\d+)', content)
    m_q = re.search(r'(instance:\s*\n(?:.*\n)*?\s+Q:\s*)(\d+)', content)
    old_p = int(m_p.group(2)) if m_p else 0
    old_q = int(m_q.group(2)) if m_q else 0
    
    # Modify problem dimensions if specified (text replacement)
    if new_p is not None:
        content = re.sub(r'(\s+P:\s*)\d+', f'\\g<1>{new_p}', content, count=1)
        content = re.sub(r'(\s+Q:\s*)\d+', f'\\g<1>{new_q}', content, count=1)
        print(f"  P-fold: P {old_p}→{new_p}, Q {old_q}→{new_q}")
    
    # Modify mapper victory_condition to 500 (text replacement)
    content = re.sub(r'(victory_condition:\s*)\d+', r'\g<1>500', content)
    print(f"  victory_condition → 500")
    
    # Create temp run directory
    run_dir = os.path.join(BASE, f"_rerun_tmp/{model}_{op}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Write modified config (preserves original YAML quoting)
    modified_yaml = os.path.join(run_dir, "input.yaml")
    with open(modified_yaml, 'w') as f:
        f.write(content)
    
    # Run timeloop-mapper
    print(f"  Running timeloop-mapper in {run_dir}...")
    result = subprocess.run(
        ["timeloop-mapper", modified_yaml],
        cwd=run_dir,
        capture_output=True, text=True,
        timeout=600  # 10 min timeout
    )
    
    stats_file = os.path.join(run_dir, "timeloop-mapper.stats.txt")
    if not os.path.isfile(stats_file):
        print(f"  ERROR: No stats output! Return code: {result.returncode}")
        print(f"  STDERR: {result.stderr[:500]}")
        return False
    
    # Check utilization
    with open(stats_file) as f:
        stats = f.read()
    
    m = re.search(r'Utilization:\s+([\d.]+)%', stats)
    util = float(m.group(1)) if m else 0
    m = re.search(r'Cycles:\s*(\d+)', stats)
    cycles = int(m.group(1)) if m else 0
    m = re.search(r'Energy:\s*([\d.]+)\s*uJ', stats)
    energy = float(m.group(1)) if m else 0
    m = re.search(r'Utilized instances\s*:\s*(\d+)', stats)
    instances = int(m.group(1)) if m else 0
    
    print(f"  Result: Util={util:.2f}% PEs={instances} Cycles={cycles} Energy={energy:.2f} uJ")
    
    # Copy results to all 3 arch directories
    for arch in ARCHS:
        dst_dir = os.path.join(BASE, arch, model, "ffn/prefill/b8s4096", op)
        if not os.path.isdir(dst_dir):
            print(f"  WARNING: {dst_dir} does not exist, skipping")
            continue
        
        # Backup old stats
        old_stats = os.path.join(dst_dir, "timeloop-mapper.stats.txt")
        if os.path.isfile(old_stats):
            shutil.copy2(old_stats, old_stats + ".bak")
        
        # Copy new stats and other output files
        for fname in os.listdir(run_dir):
            if fname.startswith("timeloop-mapper."):
                src = os.path.join(run_dir, fname)
                dst = os.path.join(dst_dir, fname)
                shutil.copy2(src, dst)
        
        print(f"  Copied results to {arch}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Re-running low-utilization PPU elementwise ops")
    print("=" * 60)
    
    results = {}
    for model, op, new_p, new_q in OPS_TO_RERUN:
        print(f"\n--- {model}/{op} ---")
        ok = modify_and_run(model, op, new_p, new_q)
        results[(model, op)] = ok
    
    print("\n" + "=" * 60)
    print("Summary:")
    for (model, op), ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {model}/{op}")
    
    # Cleanup
    # shutil.rmtree(os.path.join(BASE, "_rerun_tmp"), ignore_errors=True)
    print("\nDone. Run verification next.")
