#!/bin/bash
set -e
cd /home/accelgen

echo "========================================"
echo "[Group 3] (1) Running: llama3-8b-block0-ffn-decode-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-ffn-decode-b8s1024.mlir        simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 3] (2) Running: gemma-7b-block0-ffn-decode-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-ffn-decode-b8s1024.mlir         simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 3] (3) Running: qwen3-8b-block0-ffn-decode-b8s1024_simba_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s1024.mlir         simba_edge PPU_edge 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_simba_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_simba_edge.log"


echo "========================================"
echo "[Group 3] (4) Running: llama3-8b-block0-ffn-decode-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-ffn-decode-b8s1024.mlir        simba_server PPU_server 2>&1 | tee "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 3] (5) Running: gemma-7b-block0-ffn-decode-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-ffn-decode-b8s1024.mlir         simba_server PPU_server 2>&1 | tee "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 3] (6) Running: qwen3-8b-block0-ffn-decode-b8s1024_simba_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s1024.mlir         simba_server PPU_server 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_simba_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_simba_server.log"


echo "========================================"
echo "[Group 3] (7) Running: llama3-8b-block0-ffn-decode-b8s1024_gemmini_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-ffn-decode-b8s1024.mlir        gemmini_edge PPU_edge 2>&1 | tee "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_gemmini_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_gemmini_edge.log"


echo "========================================"
echo "[Group 3] (8) Running: gemma-7b-block0-ffn-decode-b8s1024_gemmini_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-ffn-decode-b8s1024.mlir         gemmini_edge PPU_edge 2>&1 | tee "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_gemmini_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_gemmini_edge.log"


echo "========================================"
echo "[Group 3] (9) Running: qwen3-8b-block0-ffn-decode-b8s1024_gemmini_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s1024.mlir         gemmini_edge PPU_edge 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_gemmini_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_gemmini_edge.log"


echo "========================================"
echo "[Group 3] (10) Running: llama3-8b-block0-ffn-decode-b8s1024_gemmini_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-ffn-decode-b8s1024.mlir        gemmini_server PPU_server 2>&1 | tee "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_gemmini_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_gemmini_server.log"


echo "========================================"
echo "[Group 3] (11) Running: gemma-7b-block0-ffn-decode-b8s1024_gemmini_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-ffn-decode-b8s1024.mlir         gemmini_server PPU_server 2>&1 | tee "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_gemmini_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_gemmini_server.log"


echo "========================================"
echo "[Group 3] (12) Running: qwen3-8b-block0-ffn-decode-b8s1024_gemmini_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s1024.mlir         gemmini_server PPU_server 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_gemmini_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_gemmini_server.log"


echo "========================================"
echo "[Group 3] (13) Running: llama3-8b-block0-ffn-decode-b8s1024_lego_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-ffn-decode-b8s1024.mlir        lego_edge PPU_edge 2>&1 | tee "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_lego_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_lego_edge.log"


echo "========================================"
echo "[Group 3] (14) Running: gemma-7b-block0-ffn-decode-b8s1024_lego_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-ffn-decode-b8s1024.mlir         lego_edge PPU_edge 2>&1 | tee "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_lego_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_lego_edge.log"


echo "========================================"
echo "[Group 3] (15) Running: qwen3-8b-block0-ffn-decode-b8s1024_lego_edge"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s1024.mlir         lego_edge PPU_edge 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_lego_edge.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_lego_edge.log"


echo "========================================"
echo "[Group 3] (16) Running: llama3-8b-block0-ffn-decode-b8s1024_lego_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/llama3-8b-block0-ffn-decode-b8s1024.mlir        lego_server PPU_server 2>&1 | tee "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_lego_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/llama3-8b-block0-ffn-decode-b8s1024_lego_server.log"


echo "========================================"
echo "[Group 3] (17) Running: gemma-7b-block0-ffn-decode-b8s1024_lego_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/gemma-7b-block0-ffn-decode-b8s1024.mlir         lego_server PPU_server 2>&1 | tee "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_lego_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/gemma-7b-block0-ffn-decode-b8s1024_lego_server.log"


echo "========================================"
echo "[Group 3] (18) Running: qwen3-8b-block0-ffn-decode-b8s1024_lego_server"
echo "========================================"
bash script/test_model_baseline.sh benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s1024.mlir         lego_server PPU_server 2>&1 | tee "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_lego_server.log"
echo "EXIT_CODE=$?" >> "baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s1024_lego_server.log"

echo 'Group 3 done! (18 tests)'
