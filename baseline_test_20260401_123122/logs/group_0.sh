#!/bin/bash
set -e
cd /home/accelgen

echo "========================================"
echo "[Group 0] (1) Running: llama3-8b-block0-attention-prefill-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-attention-prefill-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/llama3-8b-block0-attention-prefill-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-attention-prefill-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 0] (2) Running: llama3-8b-block0-ffn-prefill-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/llama3-8b-block0-ffn-prefill-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-ffn-prefill-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 0] (3) Running: gemma-7b-block0-attention-prefill-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-attention-prefill-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/gemma-7b-block0-attention-prefill-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-attention-prefill-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 0] (4) Running: gemma-7b-block0-ffn-prefill-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-ffn-prefill-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/gemma-7b-block0-ffn-prefill-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-ffn-prefill-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 0] (5) Running: qwen3-8b-block0-attention-prefill-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-attention-prefill-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-attention-prefill-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-attention-prefill-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 0] (6) Running: qwen3-8b-block0-ffn-prefill-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-ffn-prefill-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-ffn-prefill-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 0] (7) Running: llama3-8b-block0-attention-prefill-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-attention-prefill-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/llama3-8b-block0-attention-prefill-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-attention-prefill-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 0] (8) Running: llama3-8b-block0-ffn-prefill-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/llama3-8b-block0-ffn-prefill-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-ffn-prefill-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 0] (9) Running: gemma-7b-block0-attention-prefill-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-attention-prefill-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/gemma-7b-block0-attention-prefill-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-attention-prefill-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 0] (10) Running: gemma-7b-block0-ffn-prefill-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-ffn-prefill-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/gemma-7b-block0-ffn-prefill-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-ffn-prefill-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 0] (11) Running: qwen3-8b-block0-attention-prefill-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-attention-prefill-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-attention-prefill-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-attention-prefill-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 0] (12) Running: qwen3-8b-block0-ffn-prefill-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-ffn-prefill-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-ffn-prefill-b8s1024_simba_server.log"

echo 'Group 0 done! (12 tests)'
