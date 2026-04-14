#!/bin/bash
# Rerun the 6 failed workloads (b8s4096 FFN prefill on server architectures)
# Failed elementwise ops on PPU_server:
#   gemma-7b:  sqrt_0, divf_1, erf_2, addf_3, mulf_4, mulf_5, mulf_6 (7 ops)
#   llama3-8b: negf_0, addf_2, divf_3, mulf_4, mulf_5 (5 ops)

set -e
cd /home/accelgen
LOGDIR=baseline_test0412/logs

ARCHS=("lego_server" "gemmini_os_server" "gemmini_ws_server")
MODELS=("gemma-7b" "llama3-8b")

total=6
idx=0

for model in "${MODELS[@]}"; do
    for arch in "${ARCHS[@]}"; do
        idx=$((idx+1))
        stem="${model}-block0-ffn-prefill-b8s4096"
        logfile="${LOGDIR}/${stem}_${arch}.log"
        echo ""
        echo "[$idx/$total] Rerunning: $stem on $arch"
        echo "  Log: $logfile"

        ./build/bin/accelgen-opt \
            "benchmark/mlir/${stem}.mlir" \
            "-pass-pipeline=model-baseline-accelerator{accelerator-name=${arch} ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/${stem}.mlir}" \
            -o /dev/null \
            2>&1 | tee "$logfile"
        echo "EXIT_CODE=$?" >> "$logfile"

        # Quick check
        fail_count=$(grep -c "FAILED" "$logfile" || true)
        if [ "$fail_count" -gt 0 ]; then
            echo "  *** Still has $fail_count FAILED ops"
        else
            echo "  ✓ All ops succeeded"
        fi
    done
done

echo ""
echo "================================"
echo "All $total reruns complete."
echo "Run: python3 baseline_test0412/batch.py --summary-only"
echo "to regenerate summary."
echo "================================"
