#!/bin/bash
# Run the llama3-70b baseline subset with:
#   gemmini_os_server + PPU_server + PPU_datamove_server
#
# Usage:
#   bash script/run_llama70b_baseline.sh [num_parallel]
#
# Results:
#   baseline_test_70b/logs/*.log
#   baseline_test_70b/testorder.txt

set -e
cd /home/accelgen

NUM_PARALLEL="${1:-4}"
SESSION="llama70b_baseline"
ACCELGEN_OPT="./build/bin/accelgen-opt"
EXAMPLE_DESIGNS_DIR="/home/accelgen/example_designs"
OUT_DIR="${OUT_DIR:-baseline_test_70b}"
LOG_DIR="${OUT_DIR}/logs"
TIMELOOP_PYTHON_DIR="/tmp/timeloop-python"
TIMELOOP_BIN_DIR="/tmp/timeloop/bin"
TIMELOOP_LIB_DIR="/tmp/timeloop/lib"
export PYTHONPATH="${TIMELOOP_PYTHON_DIR}:${TIMELOOP_PYTHON_DIR}/pytimeloop:${PYTHONPATH:-}"
export PATH="${TIMELOOP_BIN_DIR}:${PATH}"
export LD_LIBRARY_PATH="${TIMELOOP_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH:-}"

LINEAR_ARCH="gemmini_os_server"
PPU_ARCH="PPU_server"
DM_ARCH="PPU_datamove_server"

MLIR_FILES=(
  "benchmark/mlir/llama3-70b-block0-attention-prefill-b8s4096.mlir"
  "benchmark/mlir/llama3-70b-block0-attention-decode-b8s4096.mlir"
  "benchmark/mlir/llama3-70b-block0-ffn-prefill-b8s4096.mlir"
  "benchmark/mlir/llama3-70b-block0-ffn-decode-b8s4096.mlir"
)

mkdir -p "$LOG_DIR"

COMMANDS=()
STEMS=()
: > "${OUT_DIR}/testorder.txt"
for mlir_file in "${MLIR_FILES[@]}"; do
  stem=$(basename "$mlir_file" .mlir)
  cmd="${ACCELGEN_OPT} ${mlir_file} -pass-pipeline=\"model-baseline-accelerator{accelerator-name=${LINEAR_ARCH} ppu-name=${PPU_ARCH} datamove-arch-name=${DM_ARCH} config-path=${EXAMPLE_DESIGNS_DIR} mlir-file=${mlir_file} timeloop-output-root=${OUT_DIR}}\" -o /dev/null"
  COMMANDS+=("$cmd")
  STEMS+=("$stem")
  echo "$cmd" >> "${OUT_DIR}/testorder.txt"
done

TOTAL=${#COMMANDS[@]}
echo "========================================"
echo " llama3-70b baseline subset"
echo "========================================"
echo " Linear arch:   ${LINEAR_ARCH}"
echo " PPU arch:      ${PPU_ARCH}"
echo " Datamove arch: ${DM_ARCH}"
echo " Workloads:     ${TOTAL}"
echo " Output dir:    ${OUT_DIR}"
echo " Parallel:      ${NUM_PARALLEL}"
echo "========================================"

for ((g=0; g<NUM_PARALLEL; g++)); do
  script_path="${LOG_DIR}/group_${g}.sh"
  {
    echo "#!/bin/bash"
    echo "set -e"
    echo "cd /home/accelgen"
    echo "export PYTHONPATH=/tmp/timeloop-python:/tmp/timeloop-python/pytimeloop:${PYTHONPATH:-}"
    echo "export PATH=/tmp/timeloop/bin:${PATH}"
    echo "export LD_LIBRARY_PATH=/tmp/timeloop/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
  } > "$script_path"

  idx=0
  for ((i=0; i<TOTAL; i++)); do
    if (( i % NUM_PARALLEL == g )); then
      cmd="${COMMANDS[$i]}"
      mlir_base="${STEMS[$i]}"
      log_file="${LOG_DIR}/${mlir_base}_${LINEAR_ARCH}.log"
      {
        echo ""
        echo "echo '========================================'"
        echo "echo '[Group ${g}] Running: ${mlir_base}'"
        echo "echo '========================================'"
        echo "${cmd} 2>&1 | tee ${log_file}"
        echo "exit_code=\${PIPESTATUS[0]}"
        echo "echo EXIT_CODE=\${exit_code} >> ${log_file}"
      } >> "$script_path"
      ((idx++)) || true
    fi
  done

  echo "echo 'Group ${g} done (${idx} tests)'" >> "$script_path"
  chmod +x "$script_path"
  echo "  Group ${g}: ${idx} tests -> ${script_path}"
done

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "70b" "bash ${LOG_DIR}/group_0.sh; bash"

for ((g=1; g<NUM_PARALLEL; g++)); do
  if [ -s "${LOG_DIR}/group_${g}.sh" ]; then
    tmux split-window -t "$SESSION" "bash ${LOG_DIR}/group_${g}.sh; bash"
    tmux select-layout -t "$SESSION" tiled
  fi
done

echo ""
echo "Started tmux session: ${SESSION}"
echo "  Attach:  tmux attach -t ${SESSION}"
echo "  Logs:    ${LOG_DIR}/"
echo "  Summary after completion:"
echo "           python3 script/collect_llama70b_baseline.py"
