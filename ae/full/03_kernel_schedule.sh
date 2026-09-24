#!/usr/bin/env bash
set -euo pipefail
ae_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ae_dir/run_stage.sh" full schedule

