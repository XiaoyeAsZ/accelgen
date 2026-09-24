#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ae/run_stage.sh <rapid|full> <generate|linalg|schedule|dap|performance>

Environment variables:
  AE_RESULTS_DIR  Output root (default: ae/results)
  ACCELGEN_OPT    accelgen-opt executable (default: build/bin/accelgen-opt)
  PYTHON          Python executable (default: python3)
  MAX_OPERATION   Maximum operations per scheduled cluster (default: 6)
  FORCE=1         Rebuild outputs that already exist
  DRY_RUN=1       Print commands without running them
  VERBOSE=1       Also stream optimizer output to the console

The full matrix can be narrowed with space-separated AE_MODELS, AE_ACTIONS,
AE_LAYERS, AE_LENGTHS, or AE_CONFIGS environment variables.
EOF
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

suite="$1"
stage="$2"

case "$suite" in
  rapid|full) ;;
  *)
    echo "Unknown suite: $suite" >&2
    usage >&2
    exit 2
    ;;
esac

case "$stage" in
  generate|linalg|schedule|dap|performance) ;;
  *)
    echo "Unknown stage: $stage" >&2
    usage >&2
    exit 2
    ;;
esac

ae_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ae_dir/.." && pwd)"
results_dir="${AE_RESULTS_DIR:-$ae_dir/results}"
suite_dir="$results_dir/$suite"
accelgen_opt="${ACCELGEN_OPT:-$repo_root/build/bin/accelgen-opt}"
python_bin="${PYTHON:-python3}"
max_operation="${MAX_OPERATION:-6}"
force="${FORCE:-0}"
dry_run="${DRY_RUN:-0}"
verbose="${VERBOSE:-0}"
block=0

if [[ ! "$max_operation" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_OPERATION must be a positive integer: $max_operation" >&2
  exit 2
fi

if [[ "$suite" == "rapid" ]]; then
  models=("llama3-8b")
  actions=("prefill")
  layers=("ffn")
  lengths=("128")
  configs=("edge")
else
  read -r -a models <<< "${AE_MODELS:-llama3-8b llama3-70b qwen3-8b qwen3-moe gemma-7b}"
  read -r -a actions <<< "${AE_ACTIONS:-prefill decode}"
  read -r -a layers <<< "${AE_LAYERS:-attention ffn}"
  read -r -a lengths <<< "${AE_LENGTHS:-128 256 512 1024 4096}"
  read -r -a configs <<< "${AE_CONFIGS:-edge server}"
fi

mlir_dir="$suite_dir/mlir"
generic_dir="$suite_dir/generic"
scheduled_dir="$suite_dir/scheduled"
dap_dir="$suite_dir/dap"
performance_dir="$suite_dir/performance"
log_dir="$suite_dir/logs"
mkdir -p \
  "$mlir_dir" "$generic_dir" "$scheduled_dir" "$dap_dir" \
  "$performance_dir" "$log_dir"

stage_log="$log_dir/${stage}.log"
command_log="$log_dir/${stage}-commands.sh"
touch "$stage_log"
if [[ ! -s "$command_log" ]]; then
  printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$command_log"
fi
printf '\n# suite=%s stage=%s started=%(%Y-%m-%dT%H:%M:%SZ)T\n' \
  "$suite" "$stage" -1 >> "$command_log"
printf '\n=== suite=%s stage=%s started=%(%Y-%m-%dT%H:%M:%SZ)T ===\n' \
  "$suite" "$stage" -1 >> "$stage_log"

if [[ "$stage" != "generate" && "$dry_run" != "1" && ! -x "$accelgen_opt" ]]; then
  echo "accelgen-opt is not executable: $accelgen_opt" >&2
  exit 1
fi

if [[ "$stage" == "generate" && "$dry_run" != "1" ]] && ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 1
fi

batch_for_config() {
  case "$1" in
    edge) echo 1 ;;
    server) echo 8 ;;
    *)
      echo "Unsupported configuration: $1" >&2
      return 1
      ;;
  esac
}

record_command() {
  local cwd="$1"
  shift
  {
    printf '(cd %q &&' "$cwd"
    printf ' %q' "$@"
    printf ')\n'
  } >> "$command_log"
}

run_command() {
  local cwd="$1"
  shift
  record_command "$cwd" "$@"
  if [[ "$dry_run" == "1" ]]; then
    printf '  DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  if [[ "$verbose" == "1" ]]; then
    (cd "$cwd" && "$@") 2>&1 | tee -a "$stage_log"
  elif ! (cd "$cwd" && "$@") >> "$stage_log" 2>&1; then
    echo "Command failed; last 40 log lines:" >&2
    tail -n 40 "$stage_log" >&2
    return 1
  fi
}

check_input() {
  local input="$1"
  if [[ "$dry_run" != "1" && ! -s "$input" ]]; then
    echo "Missing or empty input: $input" >&2
    echo "Run the preceding AE stage first." >&2
    exit 1
  fi
}

should_skip() {
  local output="$1"
  [[ "$dry_run" != "1" && "$force" != "1" && -s "$output" ]]
}

run_workload() {
  local model="$1"
  local action="$2"
  local layer="$3"
  local length="$4"
  local config="$5"
  local batch="$6"
  local stem="${model}-block${block}-${layer}-${action}-b${batch}s${length}"
  local input output config_path pipeline

  case "$stage" in
    generate)
      output="$mlir_dir/${stem}.mlir"
      if should_skip "$output"; then
        echo "  SKIP: $output"
        return
      fi
      run_command "$suite_dir" \
        "$python_bin" "$repo_root/benchmark/dump_mlir.py" \
        "--model=$model" "--action=$action" "--block=$block" \
        "--layer=$layer" "--batch=$batch" "--length=$length"
      ;;
    linalg)
      input="$mlir_dir/${stem}.mlir"
      output="$generic_dir/${stem}-generic.mlir"
      check_input "$input"
      if should_skip "$output"; then
        echo "  SKIP: $output"
        return
      fi
      pipeline="resolve-mixed-precision,linalg-generalize-named-ops,mark-generic,eliminate-dead-op,fuse-generic,fold-tensor-op"
      run_command "$repo_root" "$accelgen_opt" "$input" \
        "-pass-pipeline=$pipeline" -o "$output"
      ;;
    schedule)
      input="$generic_dir/${stem}-generic.mlir"
      output="$scheduled_dir/${stem}-generic-scheduled-${config}-${max_operation}.mlir"
      config_path="$repo_root/config/${config}.json"
      check_input "$input"
      check_input "$config_path"
      if should_skip "$output"; then
        echo "  SKIP: $output"
        return
      fi
      pipeline="func.func(kernel-schedule{config-path=$config_path max-operation=$max_operation})"
      run_command "$repo_root" "$accelgen_opt" "$input" \
        "-pass-pipeline=$pipeline" -o "$output"
      ;;
    dap)
      input="$scheduled_dir/${stem}-generic-scheduled-${config}-${max_operation}.mlir"
      output="$dap_dir/${stem}-dap-${config}-${max_operation}.mlir"
      config_path="$repo_root/config/${config}.json"
      check_input "$input"
      check_input "$config_path"
      if should_skip "$output"; then
        echo "  SKIP: $output"
        return
      fi
      pipeline="convert-to-dap{config-path=$config_path}"
      run_command "$repo_root" "$accelgen_opt" "$input" \
        "-pass-pipeline=$pipeline" -o "$output"
      ;;
    performance)
      input="$scheduled_dir/${stem}-generic-scheduled-${config}-${max_operation}.mlir"
      output="$performance_dir/${stem}-performance-${config}-${max_operation}.mlir"
      config_path="$repo_root/config/${config}.json"
      check_input "$input"
      check_input "$config_path"
      if should_skip "$output"; then
        echo "  SKIP: $output"
        return
      fi
      if [[ "$dry_run" != "1" ]]; then
        printf 'Running: model=%s action=%s block=%s layer=%s batch=%s length=%s config=%s\n' \
          "$model" "$action" "$block" "$layer" "$batch" "$length" "$config" \
          >> "$stage_log"
      fi
      pipeline="model-performance{config-path=$config_path}"
      run_command "$repo_root" "$accelgen_opt" "$input" \
        "-pass-pipeline=$pipeline" -o "$output"
      ;;
  esac
}

total=$((${#models[@]} * ${#actions[@]} * ${#layers[@]} * ${#lengths[@]} * ${#configs[@]}))
current=0
start_time=$SECONDS

echo "AccelGen AE: suite=$suite stage=$stage workloads=$total"
echo "Results: $suite_dir"

for model in "${models[@]}"; do
  for action in "${actions[@]}"; do
    for layer in "${layers[@]}"; do
      for length in "${lengths[@]}"; do
        for config in "${configs[@]}"; do
          batch="$(batch_for_config "$config")"
          current=$((current + 1))
          echo "[$current/$total] model=$model action=$action layer=$layer length=$length config=$config batch=$batch"
          run_workload "$model" "$action" "$layer" "$length" "$config" "$batch"
        done
      done
    done
  done
done

echo "Completed suite=$suite stage=$stage in $((SECONDS - start_time)) seconds"
echo "Command record: $command_log"
