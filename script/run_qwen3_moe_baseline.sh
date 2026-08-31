#!/bin/bash
# Run Qwen3-MoE while preserving all tensor dimensions encoded in the MLIR.

set -euo pipefail
cd /home/accelgen

OUT_DIR="${OUT_DIR:-baseline_test_qwen3_moe_preserve_batch}"
MAX_PARALLEL_ARCHES="${MAX_PARALLEL_ARCHES:-3}"
LOG_DIR="${OUT_DIR}/logs"
ACCELGEN_OPT="./build/bin/accelgen-opt"
EXAMPLE_DESIGNS_DIR="/home/accelgen/example_designs"

export PYTHONPATH="/tmp/timeloop-python:/tmp/timeloop-python/pytimeloop:${PYTHONPATH:-}"
export PATH="/tmp/timeloop/bin:${PATH}"
export LD_LIBRARY_PATH="/tmp/timeloop/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

ARCH_CONFIGS=(
  "gemmini_os_edge PPU_edge PPU_datamove_edge b1"
  "gemmini_ws_edge PPU_edge PPU_datamove_edge b1"
  "lego_edge PPU_edge PPU_datamove_edge b1"
  "gemmini_os_server PPU_server PPU_datamove_server b8"
  "gemmini_ws_server PPU_server PPU_datamove_server b8"
  "lego_server PPU_server PPU_datamove_server b8"
)

if [[ -e "${OUT_DIR}" ]] && find "${OUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty output directory: ${OUT_DIR}" >&2
  exit 1
fi
mkdir -p "${LOG_DIR}"
: > "${OUT_DIR}/testorder.txt"

run_arch() {
  local linear_arch="$1"
  local ppu_arch="$2"
  local dm_arch="$3"
  local batch_tag="$4"
  local arch_log="${LOG_DIR}/${linear_arch}.log"
  local arch_status=0

  mapfile -t mlir_files < <(
    find benchmark/mlir -maxdepth 1 \
      -name "qwen3-moe-block0-*-${batch_tag}s*.mlir" -print | sort
  )
  if [[ "${#mlir_files[@]}" -ne 24 ]]; then
    echo "Expected 24 ${batch_tag} workloads, found ${#mlir_files[@]}" >&2
    return 1
  fi

  : > "${arch_log}"
  for mlir_file in "${mlir_files[@]}"; do
    local pass_pipeline
    pass_pipeline="model-baseline-accelerator{accelerator-name=${linear_arch} ppu-name=${ppu_arch} datamove-arch-name=${dm_arch} config-path=${EXAMPLE_DESIGNS_DIR} mlir-file=${mlir_file} timeloop-output-root=${OUT_DIR} preserve-input-batch=true}"
    local cmd=(
      "${ACCELGEN_OPT}" "${mlir_file}"
      "-pass-pipeline=${pass_pipeline}"
      -o /dev/null
    )

    printf '%q ' "${cmd[@]}" >> "${OUT_DIR}/testorder.txt"
    printf '\n' >> "${OUT_DIR}/testorder.txt"
    echo "[${linear_arch}] START ${mlir_file}" | tee -a "${arch_log}"

    set +e
    "${cmd[@]}" 2>&1 | tee -a "${arch_log}"
    local exit_code=${PIPESTATUS[0]}
    set -e

    echo "[${linear_arch}] EXIT ${exit_code} ${mlir_file}" | tee -a "${arch_log}"
    if [[ "${exit_code}" -ne 0 ]]; then
      arch_status=1
    fi
  done

  echo "[${linear_arch}] DONE" | tee -a "${arch_log}"
  return "${arch_status}"
}

run_group() {
  local -a pids=()
  local status=0
  local config
  for config in "$@"; do
    read -r linear_arch ppu_arch dm_arch batch_tag <<< "${config}"
    run_arch "${linear_arch}" "${ppu_arch}" "${dm_arch}" "${batch_tag}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

status=0
for ((start=0; start<${#ARCH_CONFIGS[@]}; start+=MAX_PARALLEL_ARCHES)); do
  group=("${ARCH_CONFIGS[@]:start:MAX_PARALLEL_ARCHES}")
  if ! run_group "${group[@]}"; then
    status=1
  fi
done

python3 script/collect_qwen3_moe_summary.py \
  --in-dir "${OUT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --title "Qwen3-MoE Preserve MLIR Shapes"

exit "${status}"
