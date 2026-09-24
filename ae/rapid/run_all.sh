#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for stage in \
  01_generate_mlir.sh \
  02_linalg_opt.sh \
  03_kernel_schedule.sh \
  04_lower_to_dap.sh \
  05_model_performance.sh; do
  "$script_dir/$stage"
done
