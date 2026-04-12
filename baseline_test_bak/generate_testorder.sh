#!/bin/bash
# Generate testorder.txt with all 360 test commands
# Usage: bash generate_testorder.sh > testorder.txt
cd /home/accelgen

ACCELGEN_OPT="./build/bin/accelgen-opt"
EXAMPLE_DESIGNS_DIR="/home/accelgen/example_designs"
MLIR_DIR="benchmark/mlir"

# Architecture groups: linear_arch:ppu_name:datamove_arch:batch_tag
ARCH_GROUPS=(
  "gemmini_os_edge:PPU_edge:PPU_datamove_edge:b1"
  "gemmini_ws_edge:PPU_edge:PPU_datamove_edge:b1"
  "lego_edge:PPU_edge:PPU_datamove_edge:b1"
  "gemmini_os_server:PPU_server:PPU_datamove_server:b8"
  "gemmini_ws_server:PPU_server:PPU_datamove_server:b8"
  "lego_server:PPU_server:PPU_datamove_server:b8"
)

for group in "${ARCH_GROUPS[@]}"; do
  IFS=':' read -r linear_arch ppu_name dm_name batch_tag <<< "$group"
  
  if [ "$batch_tag" = "b1" ]; then
    files=($(ls ${MLIR_DIR}/*-b1s*.mlir 2>/dev/null | sort))
  else
    files=($(ls ${MLIR_DIR}/*-b8s*.mlir 2>/dev/null | sort))
  fi

  for mlir_file in "${files[@]}"; do
    echo "$ACCELGEN_OPT $mlir_file -pass-pipeline=\"model-baseline-accelerator{accelerator-name=${linear_arch} ppu-name=${ppu_name} datamove-arch-name=${dm_name} config-path=${EXAMPLE_DESIGNS_DIR} mlir-file=${mlir_file}}\" -o /dev/null"
  done
done
