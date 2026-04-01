#!/bin/bash
set -e
cd /home/accelgen

echo "========================================"
echo "[Group 1] (1) Running: llama3-8b-block0-attention-decode-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-attention-decode-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/llama3-8b-block0-attention-decode-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-attention-decode-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 1] (2) Running: llama3-8b-block0-ffn-decode-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-ffn-decode-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 1] (3) Running: gemma-7b-block0-attention-decode-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-attention-decode-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/gemma-7b-block0-attention-decode-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-attention-decode-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 1] (4) Running: gemma-7b-block0-ffn-decode-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-ffn-decode-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 1] (5) Running: qwen3-8b-block0-attention-decode-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-attention-decode-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-attention-decode-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-attention-decode-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 1] (6) Running: qwen3-8b-block0-ffn-decode-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s1024.mlir simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 1] (7) Running: llama3-8b-block0-attention-decode-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-attention-decode-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/llama3-8b-block0-attention-decode-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-attention-decode-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 1] (8) Running: llama3-8b-block0-ffn-decode-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-ffn-decode-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 1] (9) Running: gemma-7b-block0-attention-decode-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-attention-decode-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/gemma-7b-block0-attention-decode-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-attention-decode-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 1] (10) Running: gemma-7b-block0-ffn-decode-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-ffn-decode-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 1] (11) Running: qwen3-8b-block0-attention-decode-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-attention-decode-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-attention-decode-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-attention-decode-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 1] (12) Running: qwen3-8b-block0-ffn-decode-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s1024.mlir simba_server PPU_server 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_simba_server.log"

echo 'Group 1 done! (12 tests)'
