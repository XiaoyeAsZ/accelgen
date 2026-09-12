#!/bin/bash
# Run the full llama3-70b baseline sweep across all six baseline architectures.
#
# Arch/batch pairing matters: ModelBaselineAccelerator overrides the batch
# dimension from the architecture name unless preserve-input-batch is set
# (name contains "edge" -> batch=1, "server" -> batch=8), so edge archs are
# paired with the b1 MLIR and server archs with the b8 MLIR.
#
# Usage:
#   bash script/run_llama70b_baseline_all.sh [num_parallel]
#
# Results:
#   ${OUT_DIR}/logs/<mlir_stem>_<linear_arch>.log   (one log per workload)
#   ${OUT_DIR}/summary.csv  summary_detail.csv  summary.txt

set -u
cd /home/accelgen

NUM_PARALLEL="${1:-8}"
OUT_DIR="${OUT_DIR:-baseline_test_70b_new}"
LOG_DIR="${OUT_DIR}/logs"
JOB_DIR="${OUT_DIR}/jobs"
ACCELGEN_OPT="./build/bin/accelgen-opt"
EXAMPLE_DESIGNS_DIR="/home/accelgen/example_designs"

MODEL="llama3-70b"
BLOCK=0
ACTIONS=("prefill" "decode")
LAYERS=("attention" "ffn")
LENGTHS=(4096 1024 512 256 128)   # heaviest first so the tail is short

ARCH_CONFIGS=(
  "gemmini_os_server PPU_server PPU_datamove_server 8"
  "gemmini_ws_server PPU_server PPU_datamove_server 8"
  "lego_server       PPU_server PPU_datamove_server 8"
  "gemmini_os_edge   PPU_edge   PPU_datamove_edge   1"
  "gemmini_ws_edge   PPU_edge   PPU_datamove_edge   1"
  "lego_edge         PPU_edge   PPU_datamove_edge   1"
)

if [[ -e "${OUT_DIR}" ]] && find "${OUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty output directory: ${OUT_DIR}" >&2
  exit 1
fi
mkdir -p "${LOG_DIR}" "${JOB_DIR}"
: > "${OUT_DIR}/testorder.txt"

JOB_SCRIPTS=()
for length in "${LENGTHS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    for action in "${ACTIONS[@]}"; do
      for config in "${ARCH_CONFIGS[@]}"; do
        read -r linear_arch ppu_arch dm_arch batch <<< "${config}"
        stem="${MODEL}-block${BLOCK}-${layer}-${action}-b${batch}s${length}"
        mlir_file="benchmark/mlir/${stem}.mlir"
        if [[ ! -f "${mlir_file}" ]]; then
          echo "Missing MLIR: ${mlir_file}" >&2
          exit 1
        fi

        log_file="${LOG_DIR}/${stem}_${linear_arch}.log"
        pass_pipeline="model-baseline-accelerator{accelerator-name=${linear_arch} ppu-name=${ppu_arch} datamove-arch-name=${dm_arch} config-path=${EXAMPLE_DESIGNS_DIR} mlir-file=${mlir_file} timeloop-output-root=${OUT_DIR} preserve-input-batch=true}"
        cmd="${ACCELGEN_OPT} ${mlir_file} -pass-pipeline=\"${pass_pipeline}\" -o /dev/null"
        echo "${cmd}" >> "${OUT_DIR}/testorder.txt"

        job_script="${JOB_DIR}/${stem}_${linear_arch}.sh"
        {
          echo "#!/bin/bash"
          echo "cd /home/accelgen"
          echo "export PATH=/usr/local/bin:\${PATH}"
          echo "echo \"[START] ${stem} ${linear_arch} \$(date '+%F %T')\""
          echo "${cmd} > ${log_file} 2>&1"
          echo "exit_code=\$?"
          echo "echo EXIT_CODE=\${exit_code} >> ${log_file}"
          echo "echo \"[DONE ] ${stem} ${linear_arch} exit=\${exit_code} \$(date '+%F %T')\""
        } > "${job_script}"
        chmod +x "${job_script}"
        JOB_SCRIPTS+=("${job_script}")
      done
    done
  done
done

TOTAL=${#JOB_SCRIPTS[@]}
echo "========================================"
echo " llama3-70b full baseline sweep"
echo "========================================"
echo " Architectures: ${#ARCH_CONFIGS[@]}"
echo " Workloads:     ${TOTAL}"
echo " Output dir:    ${OUT_DIR}"
echo " Parallel:      ${NUM_PARALLEL}"
echo " Started:       $(date '+%F %T')"
echo "========================================"

printf '%s\n' "${JOB_SCRIPTS[@]}" | xargs -P "${NUM_PARALLEL}" -n 1 bash
sweep_status=$?

echo "Sweep finished at $(date '+%F %T') (xargs status=${sweep_status})"

# --- Statistics, step 1: same per-workload collection as the earlier 70b run
python3 script/collect_llama70b_baseline.py \
  --out-dir "${OUT_DIR}" \
  --title "Llama3-70B (6 baseline architectures)"

# --- Statistics, step 2: combined attention+FFN tables per action/config/length.
# Batch dimensions are preserved in the run, so no FFN expert-group scaling and
# FLOPs come from the summary itself. "ours" is excluded: test/performance.log
# carries no llama3-70b records.
python3 evaluation/analyze_baseline_summary.py \
  --summary "${OUT_DIR}/summary.csv" \
  --output-dir evaluation/results/llama3-70b-baseline \
  --performance-log /dev/null \
  --ffn-batch-mode preserved \
  --combined-name llama3_70b_baseline_combined.csv \
  --flops-source summary \
  --no-include-ours

exit "${sweep_status}"
