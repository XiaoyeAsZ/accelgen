#!/bin/bash
# =============================================================================
# Baseline Test: Run ALL MLIR files across 6 architecture groups
#
# Architecture groups (linear + PPU + datamove):
#   Edge (b1 MLIR files):
#     1. gemmini_os_edge  + PPU_edge + PPU_datamove_edge
#     2. gemmini_ws_edge  + PPU_edge + PPU_datamove_edge
#     3. lego_edge        + PPU_edge + PPU_datamove_edge
#   Server (b8 MLIR files):
#     4. gemmini_os_server + PPU_server + PPU_datamove_server
#     5. gemmini_ws_server + PPU_server + PPU_datamove_server
#     6. lego_server       + PPU_server + PPU_datamove_server
#
# Each invocation runs ALL op types:
#   - batch_matmul    → linear arch (gemmini_os/ws, lego)
#   - elementwise/red → PPU arch
#   - datamove         → PPU_datamove arch
#
# Results: baseline_test/<linear_arch>/<mlir_stem>/
# =============================================================================
set -e
cd /home/accelgen

ACCELGEN_OPT="./build/bin/accelgen-opt"
EXAMPLE_DESIGNS_DIR="/home/accelgen/example_designs"
MLIR_DIR="benchmark/mlir"
OUTPUT_DIR="baseline_test"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ---- Architecture groups ----
# Format: "linear_arch:ppu_name:datamove_arch:batch_tag"
#   batch_tag = "b1" for edge, "b8" for server
ARCH_GROUPS=(
  "gemmini_os_edge:PPU_edge:PPU_datamove_edge:b1"
  "gemmini_ws_edge:PPU_edge:PPU_datamove_edge:b1"
  "lego_edge:PPU_edge:PPU_datamove_edge:b1"
  "gemmini_os_server:PPU_server:PPU_datamove_server:b8"
  "gemmini_ws_server:PPU_server:PPU_datamove_server:b8"
  "lego_server:PPU_server:PPU_datamove_server:b8"
)

# ---- Count total tasks ----
B1_FILES=($(ls ${MLIR_DIR}/*-b1s*.mlir 2>/dev/null | sort))
B8_FILES=($(ls ${MLIR_DIR}/*-b8s*.mlir 2>/dev/null | sort))
total=$(( 3 * ${#B1_FILES[@]} + 3 * ${#B8_FILES[@]} ))
count=0
failed=0
succeeded=0

echo "============================================="
echo " Baseline Test — Full Architecture Sweep"
echo "============================================="
echo " Architecture groups: ${#ARCH_GROUPS[@]}"
echo " b1 MLIR files: ${#B1_FILES[@]}"
echo " b8 MLIR files: ${#B8_FILES[@]}"
echo " Total tasks: ${total}"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

run_one() {
  local linear_arch="$1"
  local ppu_name="$2"
  local dm_name="$3"
  local mlir_file="$4"

  local stem
  stem=$(basename "$mlir_file" .mlir)
  count=$((count + 1))

  echo ""
  echo "========================================"
  echo "[$count/$total] ${stem}"
  echo "  Linear: ${linear_arch}  PPU: ${ppu_name}  DM: ${dm_name}"
  echo "========================================"

  local log_file="${LOG_DIR}/${stem}_${linear_arch}.log"
  mkdir -p "${OUTPUT_DIR}/${linear_arch}/${stem}"

  $ACCELGEN_OPT "$mlir_file" \
    -pass-pipeline="model-baseline-accelerator{\
accelerator-name=${linear_arch} \
ppu-name=${ppu_name} \
datamove-arch-name=${dm_name} \
config-path=${EXAMPLE_DESIGNS_DIR} \
mlir-file=${mlir_file}}" \
    -o /dev/null \
    2>&1 | tee "$log_file"

  local exit_code=${PIPESTATUS[0]}
  echo "EXIT_CODE=$exit_code" >> "$log_file"

  if [ $exit_code -ne 0 ]; then
    echo "*** FAILED: ${stem} on ${linear_arch} ***"
    failed=$((failed + 1))
  else
    succeeded=$((succeeded + 1))
  fi
}

# ---- Main loop ----
for group in "${ARCH_GROUPS[@]}"; do
  IFS=':' read -r linear_arch ppu_name dm_name batch_tag <<< "$group"

  echo ""
  echo "############################################################"
  echo "# Group: ${linear_arch} (${batch_tag} files)"
  echo "#   PPU: ${ppu_name}  Datamove: ${dm_name}"
  echo "############################################################"

  if [ "$batch_tag" = "b1" ]; then
    files=("${B1_FILES[@]}")
  else
    files=("${B8_FILES[@]}")
  fi

  for mlir_file in "${files[@]}"; do
    run_one "$linear_arch" "$ppu_name" "$dm_name" "$mlir_file"
  done
done

# ---- Summary ----
echo ""
echo "============================================="
echo " Baseline Test Complete"
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Total: ${count}  Succeeded: ${succeeded}  Failed: ${failed}"
echo "============================================="
echo " Results: ${OUTPUT_DIR}/<arch>/<mlir_stem>/"
echo " Logs:    ${LOG_DIR}/"
echo "============================================="
