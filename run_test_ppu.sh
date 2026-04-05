#!/bin/bash
# Run all MLIR baselines with skip-linear (skip batch_matmul), 
# outputting results under test_ppu/
set -e
cd /home/accelgen

ACCELGEN_OPT="./build/bin/accelgen-opt"
EXAMPLE_DESIGNS_DIR="/home/accelgen/example_designs"
OUTPUT_DIR="test_ppu"
mkdir -p "${OUTPUT_DIR}/logs"

MLIR_FILES=(
  benchmark/mlir/llama3-8b-block0-attention-prefill-b8s1024.mlir
  benchmark/mlir/llama3-8b-block0-attention-decode-b8s1024.mlir
  benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir
  benchmark/mlir/llama3-8b-block0-ffn-decode-b8s1024.mlir
  benchmark/mlir/gemma-7b-block0-attention-prefill-b8s1024.mlir
  benchmark/mlir/gemma-7b-block0-attention-decode-b8s1024.mlir
  benchmark/mlir/gemma-7b-block0-ffn-prefill-b8s1024.mlir
  benchmark/mlir/gemma-7b-block0-ffn-decode-b8s1024.mlir
  benchmark/mlir/qwen3-8b-block0-attention-prefill-b8s1024.mlir
  benchmark/mlir/qwen3-8b-block0-attention-decode-b8s1024.mlir
  benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s1024.mlir
  benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s1024.mlir
)

# PPU configs: ppu-name + datamove-arch-name pairs
PPU_CONFIGS=(
  "PPU_edge:PPU_datamove_edge"
  "PPU_server:PPU_datamove_server"
)

total=${#MLIR_FILES[@]}
count=0
failed=0

for mlir_file in "${MLIR_FILES[@]}"; do
  stem=$(basename "$mlir_file" .mlir)
  for config in "${PPU_CONFIGS[@]}"; do
    ppu_name="${config%%:*}"
    dm_name="${config##*:}"
    # Use ppu_name suffix (edge/server) for output dir
    ppu_suffix="${ppu_name#PPU_}"
    
    count=$((count + 1))
    echo ""
    echo "========================================"
    echo "[$count] Running: ${stem} on ${ppu_name} (skip-linear)"
    echo "========================================"
    
    log_file="${OUTPUT_DIR}/logs/${stem}_${ppu_suffix}.log"
    
    # Output path becomes: baseline_test/<ppu_suffix>/<stem>/
    # After running, we move results to test_ppu/
    $ACCELGEN_OPT "$mlir_file" \
      -pass-pipeline="model-baseline-accelerator{accelerator-name=${ppu_suffix} ppu-name=${ppu_name} datamove-arch-name=${dm_name} config-path=${EXAMPLE_DESIGNS_DIR} mlir-file=${mlir_file} skip-linear=true}" \
      -o /dev/null \
      2>&1 | tee "$log_file"
    
    exit_code=${PIPESTATUS[0]}
    echo "EXIT_CODE=$exit_code" >> "$log_file"
    
    if [ $exit_code -ne 0 ]; then
      echo "*** FAILED: ${stem} on ${ppu_name} ***"
      failed=$((failed + 1))
    fi
  done
done

echo ""
echo "========================================"
echo "All done. Total=$count, Failed=$failed"
echo "Moving results from baseline_test/ to ${OUTPUT_DIR}/"
echo "========================================"

# Move results: baseline_test/edge/* and baseline_test/server/* → test_ppu/
for ppu_suffix in edge server; do
  if [ -d "baseline_test/${ppu_suffix}" ]; then
    mkdir -p "${OUTPUT_DIR}/${ppu_suffix}"
    cp -r "baseline_test/${ppu_suffix}/"* "${OUTPUT_DIR}/${ppu_suffix}/" 2>/dev/null || true
    rm -rf "baseline_test/${ppu_suffix}"
  fi
done

echo "Results in: ${OUTPUT_DIR}/"
