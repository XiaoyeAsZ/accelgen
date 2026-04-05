#!/bin/bash
# Run all MLIR baselines with lego_server (linear only, skip elem+datamove)
# Results saved under test_lego/server/
set -e
cd /home/accelgen

ACCELGEN_OPT="./build/bin/accelgen-opt"
EXAMPLE_DESIGNS_DIR="/home/accelgen/example_designs"
LINEAR_ARCH="lego_server"
OUTPUT_DIR="test_lego"
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

total=${#MLIR_FILES[@]}
count=0
failed=0

for mlir_file in "${MLIR_FILES[@]}"; do
  stem=$(basename "$mlir_file" .mlir)
  count=$((count + 1))

  echo ""
  echo "========================================"
  echo "[$count/$total] Running: ${stem} on ${LINEAR_ARCH} (linear only)"
  echo "========================================"

  log_file="${OUTPUT_DIR}/logs/${stem}_${LINEAR_ARCH}.log"

  $ACCELGEN_OPT "$mlir_file" \
    -pass-pipeline="model-baseline-accelerator{accelerator-name=${LINEAR_ARCH} config-path=${EXAMPLE_DESIGNS_DIR} mlir-file=${mlir_file} skip-nonlinear=true}" \
    -o /dev/null \
    2>&1 | tee "$log_file"

  exit_code=${PIPESTATUS[0]}
  echo "EXIT_CODE=$exit_code" >> "$log_file"

  if [ $exit_code -ne 0 ]; then
    echo "*** FAILED: ${stem} on ${LINEAR_ARCH} ***"
    failed=$((failed + 1))
  fi
done

echo ""
echo "========================================"
echo "All done. Total=$count, Failed=$failed"
echo "========================================"

# Move results from baseline_test/lego_server/ to test_lego/server/
if [ -d "baseline_test/${LINEAR_ARCH}" ]; then
  mkdir -p "${OUTPUT_DIR}/server"
  cp -r "baseline_test/${LINEAR_ARCH}/"* "${OUTPUT_DIR}/server/" 2>/dev/null || true
  echo "Copied baseline_test/${LINEAR_ARCH}/ -> ${OUTPUT_DIR}/server/"
fi

echo "Results in: ${OUTPUT_DIR}/"
