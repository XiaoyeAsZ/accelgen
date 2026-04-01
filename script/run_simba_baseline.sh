#!/bin/bash
# Run simba_edge+PPU_edge and simba_server+PPU_server baseline tests
# Uses tmux for parallel execution, then auto-collects results
# Usage: bash script/run_simba_baseline.sh [num_parallel]
#   num_parallel: number of parallel tmux panes (default: 2)

set -e
cd /home/accelgen

NUM_PARALLEL="${1:-2}"
SESSION="simba_run"

# Create timestamped output directory and symlink baseline_test → it
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REAL_DIR="baseline_test_${TIMESTAMP}"
mkdir -p "$REAL_DIR"

# Remove old symlink or empty dir, then create symlink
rm -f baseline_test 2>/dev/null || rmdir baseline_test 2>/dev/null || true
ln -sfn "$REAL_DIR" baseline_test
echo "Output directory: $REAL_DIR (symlinked as baseline_test)"

LOG_DIR="baseline_test/logs"
mkdir -p "$LOG_DIR"

# Build the command list: simba_edge + simba_server for all 12 MLIR files
COMMANDS=()

MODELS=(
  "llama3-8b-block0-attention-prefill-b8s1024"
  "llama3-8b-block0-attention-decode-b8s1024"
  "llama3-8b-block0-ffn-prefill-b8s1024"
  "llama3-8b-block0-ffn-decode-b8s1024"
  "gemma-7b-block0-attention-prefill-b8s1024"
  "gemma-7b-block0-attention-decode-b8s1024"
  "gemma-7b-block0-ffn-prefill-b8s1024"
  "gemma-7b-block0-ffn-decode-b8s1024"
  "qwen3-8b-block0-attention-prefill-b8s1024"
  "qwen3-8b-block0-attention-decode-b8s1024"
  "qwen3-8b-block0-ffn-prefill-b8s1024"
  "qwen3-8b-block0-ffn-decode-b8s1024"
)

# Group 1: simba_edge + PPU_edge
for m in "${MODELS[@]}"; do
  COMMANDS+=("bash script/test_model_baseline.sh benchmark/mlir/${m}.mlir simba_edge PPU_edge")
done

# Group 2: simba_server + PPU_server
for m in "${MODELS[@]}"; do
  COMMANDS+=("bash script/test_model_baseline.sh benchmark/mlir/${m}.mlir simba_server PPU_server")
done

TOTAL=${#COMMANDS[@]}
echo "Total tests to run: $TOTAL (12 simba_edge + 12 simba_server)"
echo "Parallel sessions:  $NUM_PARALLEL"
echo ""

# Split commands into groups for each tmux pane
for ((g=0; g<NUM_PARALLEL; g++)); do
  SCRIPT="$LOG_DIR/group_${g}.sh"
  cat > "$SCRIPT" <<'HEADER'
#!/bin/bash
set -e
cd /home/accelgen
HEADER

  idx=0
  for ((i=0; i<TOTAL; i++)); do
    if (( i % NUM_PARALLEL == g )); then
      cmd="${COMMANDS[$i]}"
      mlir_base=$(echo "$cmd" | grep -oP 'benchmark/mlir/\K[^ ]+' | sed 's/\.mlir//')
      arch=$(echo "$cmd" | awk '{print $4}')
      log_name="${mlir_base}_${arch}"

      cat >> "$SCRIPT" <<EOF

echo "========================================"
echo "[Group $g] ($((idx+1))) Running: $log_name"
echo "========================================"
$cmd 2>&1 | tee "$LOG_DIR/${log_name}.log"
echo "EXIT_CODE=\$?" >> "$LOG_DIR/${log_name}.log"

EOF
      ((idx++)) || true
    fi
  done

  echo "echo 'Group $g done! ($idx tests)'" >> "$SCRIPT"
  chmod +x "$SCRIPT"
  echo "  Group $g: $idx tests → $SCRIPT"
done

echo ""

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create tmux session with the first group
tmux new-session -d -s "$SESSION" -n "simba" "bash $LOG_DIR/group_0.sh; bash"

# Split into panes for remaining groups
for ((g=1; g<NUM_PARALLEL; g++)); do
  if (( g % 2 == 1 )); then
    tmux split-window -t "$SESSION" -h "bash $LOG_DIR/group_${g}.sh; bash"
  else
    tmux split-window -t "$SESSION" -v "bash $LOG_DIR/group_${g}.sh; bash"
  fi
  tmux select-layout -t "$SESSION" tiled
done

echo "========================================"
echo "All $TOTAL tests launched in tmux session: $SESSION"
echo ""
echo "  tmux attach -t $SESSION        # attach to watch progress"
echo "  tmux kill-session -t $SESSION   # kill all tests"
echo ""
echo "When all done, run:  python3 script/collect_results.py"
echo "========================================"
