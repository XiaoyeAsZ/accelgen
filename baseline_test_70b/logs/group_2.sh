#!/bin/bash
set -e
cd /home/accelgen
export PYTHONPATH=/tmp/timeloop-python:/tmp/timeloop-python/pytimeloop:/tmp/timeloop-python:/tmp/timeloop-python/pytimeloop:
export PATH=/tmp/timeloop/bin:/tmp/timeloop/bin:/root/.codex/packages/standalone/releases/0.142.5-x86_64-unknown-linux-musl/codex-path:/root/.codex/tmp/arg0/codex-arg0m07CRA:/root/.codex/packages/standalone/releases/0.142.5-x86_64-unknown-linux-musl/codex-path:/root/.local/bin:/root/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/root/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli:/root/.vscode-server/bin/c9d77990917f3102ada88be140d28b038d1dd7c7/bin/remote-cli:/usr/local/circt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.vscode-server/extensions/ms-python.debugpy-2026.6.0/bundled/scripts/noConfigScripts
export LD_LIBRARY_PATH=/tmp/timeloop/lib:/usr/local/lib:/tmp/timeloop/lib:/usr/local/lib:

echo '========================================'
echo '[Group 2] Running: llama3-70b-block0-ffn-prefill-b8s4096'
echo '========================================'
./build/bin/accelgen-opt benchmark/mlir/llama3-70b-block0-ffn-prefill-b8s4096.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-70b-block0-ffn-prefill-b8s4096.mlir}" -o /dev/null 2>&1 | tee baseline_test_70b/logs/llama3-70b-block0-ffn-prefill-b8s4096_gemmini_os_server.log
exit_code=${PIPESTATUS[0]}
echo EXIT_CODE=${exit_code} >> baseline_test_70b/logs/llama3-70b-block0-ffn-prefill-b8s4096_gemmini_os_server.log
echo 'Group 2 done (1 tests)'
