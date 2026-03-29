#!/bin/bash
# Run all baseline tests from testorder.txt using tmux for parallel execution
# Usage: bash script/run_all_baseline.sh [num_parallel]
#   num_parallel: number of parallel tmux panes (default: 4)

set -e
cd /home/accelgen

NUM_PARALLEL="${1:-4}"
SESSION="baseline_run"
LOG_DIR="baseline_test/logs"
mkdir -p "$LOG_DIR"

# Extract all bash commands (only lines starting with "bash ")
COMMANDS=()
while IFS= read -r line; do
  # Only keep lines that start with "bash "
  [[ "$line" =~ ^bash\  ]] || continue
  COMMANDS+=("$line")
done < testorder.txt

TOTAL=${#COMMANDS[@]}
echo "Total tests to run: $TOTAL"
echo "Parallel sessions:  $NUM_PARALLEL"
echo ""

# Split commands into groups for each tmux pane
# Create per-group scripts
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
      # Extract a short name for the log file
      # e.g. "llama3-8b-block0-attention-prefill-b8s1024_simba_edge"
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
tmux new-session -d -s "$SESSION" -n "baseline" "bash $LOG_DIR/group_0.sh; bash"

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
echo "  tmux attach -t $SESSION     # attach to watch progress"
echo "  tmux kill-session -t $SESSION  # kill all tests"
echo ""
echo "When done, run:  bash script/collect_results.sh"
echo "========================================"
