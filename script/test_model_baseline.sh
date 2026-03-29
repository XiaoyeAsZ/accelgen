#!/bin/bash
# Test ModelBaselineAccelerator pass with timeloop
# Usage: bash script/test_model_baseline.sh [mlir_file] [arch_name] [ppu_name]
#
# Examples:
#   bash script/test_model_baseline.sh
#   bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-ffn-prefill-b8s1024.mlir
#   bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-attention-decode-b8s1024.mlir simba_edge PPU_edge
#   bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-attention-prefill-b8s1024.mlir simba_server PPU_server

set -e

ACCELGEN_OPT="./build/bin/accelgen-opt"
EXAMPLE_DESIGNS_DIR="/home/accelgen/example_designs"

# Default MLIR file, architecture, and PPU
MLIR_FILE="${1:-benchmark/mlir/llama3-8b-block0-attention-prefill-b8s1024.mlir}"
LINEAR_ARCH="${2:-simba_like}"
PPU_ARCH="${3:-PPU}"

echo "================================================================"
echo "  ModelBaselineAccelerator Test"
echo "================================================================"
echo "  MLIR file:    $MLIR_FILE"
echo "  Config path:  $EXAMPLE_DESIGNS_DIR"
echo "  Linear arch:  $LINEAR_ARCH"
echo "  PPU arch:     $PPU_ARCH"
echo "================================================================"
echo ""

$ACCELGEN_OPT "$MLIR_FILE" \
  -pass-pipeline="builtin.module(model-baseline-accelerator{accelerator-name=$LINEAR_ARCH ppu-name=$PPU_ARCH config-path=$EXAMPLE_DESIGNS_DIR mlir-file=$MLIR_FILE})" \
  -o /dev/null \
  2>&1

echo ""
echo "Done."
