#!/bin/bash
cd /home/accelgen
echo 'Pane 7: 45 tasks'

echo '[1/45] gemma-7b-block0-attention-prefill-b1s256 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-prefill-b1s256.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-prefill-b1s256.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-prefill-b1s256_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-prefill-b1s256_gemmini_os_edge.log

echo '[2/45] gemma-7b-block0-ffn-prefill-b1s1024 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-prefill-b1s1024.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-prefill-b1s1024.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b1s1024_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b1s1024_gemmini_os_edge.log

echo '[3/45] llama3-8b-block0-attention-decode-b1s4096 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-decode-b1s4096.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-decode-b1s4096.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-decode-b1s4096_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-decode-b1s4096_gemmini_os_edge.log

echo '[4/45] llama3-8b-block0-ffn-decode-b1s128 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-decode-b1s128.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-decode-b1s128.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-decode-b1s128_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-decode-b1s128_gemmini_os_edge.log

echo '[5/45] llama3-8b-block0-ffn-prefill-b1s512 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s512.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s512.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b1s512_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b1s512_gemmini_os_edge.log

echo '[6/45] qwen3-8b-block0-attention-prefill-b1s256 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-prefill-b1s256.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-prefill-b1s256.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-prefill-b1s256_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-prefill-b1s256_gemmini_os_edge.log

echo '[7/45] qwen3-8b-block0-ffn-prefill-b1s1024 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s1024.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s1024.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b1s1024_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b1s1024_gemmini_os_edge.log

echo '[8/45] gemma-7b-block0-attention-decode-b1s4096 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-decode-b1s4096.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-decode-b1s4096.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-decode-b1s4096_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-decode-b1s4096_gemmini_ws_edge.log

echo '[9/45] gemma-7b-block0-ffn-decode-b1s128 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-decode-b1s128.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-decode-b1s128.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-decode-b1s128_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-decode-b1s128_gemmini_ws_edge.log

echo '[10/45] gemma-7b-block0-ffn-prefill-b1s512 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-prefill-b1s512.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-prefill-b1s512.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b1s512_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b1s512_gemmini_ws_edge.log

echo '[11/45] llama3-8b-block0-attention-prefill-b1s256 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-prefill-b1s256.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-prefill-b1s256.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-prefill-b1s256_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-prefill-b1s256_gemmini_ws_edge.log

echo '[12/45] llama3-8b-block0-ffn-prefill-b1s1024 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s1024.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s1024.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b1s1024_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b1s1024_gemmini_ws_edge.log

echo '[13/45] qwen3-8b-block0-attention-decode-b1s4096 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-decode-b1s4096.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-decode-b1s4096.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-decode-b1s4096_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-decode-b1s4096_gemmini_ws_edge.log

echo '[14/45] qwen3-8b-block0-ffn-decode-b1s128 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-decode-b1s128.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-decode-b1s128.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-decode-b1s128_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-decode-b1s128_gemmini_ws_edge.log

echo '[15/45] qwen3-8b-block0-ffn-prefill-b1s512 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s512.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s512.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b1s512_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b1s512_gemmini_ws_edge.log

echo '[16/45] gemma-7b-block0-attention-prefill-b1s256 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-prefill-b1s256.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-prefill-b1s256.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-prefill-b1s256_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-prefill-b1s256_lego_edge.log

echo '[17/45] gemma-7b-block0-ffn-prefill-b1s1024 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-prefill-b1s1024.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-prefill-b1s1024.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b1s1024_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b1s1024_lego_edge.log

echo '[18/45] llama3-8b-block0-attention-decode-b1s4096 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-decode-b1s4096.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-decode-b1s4096.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-decode-b1s4096_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-decode-b1s4096_lego_edge.log

echo '[19/45] llama3-8b-block0-ffn-decode-b1s128 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-decode-b1s128.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-decode-b1s128.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-decode-b1s128_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-decode-b1s128_lego_edge.log

echo '[20/45] llama3-8b-block0-ffn-prefill-b1s512 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s512.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s512.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b1s512_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b1s512_lego_edge.log

echo '[21/45] qwen3-8b-block0-attention-prefill-b1s256 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-prefill-b1s256.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-prefill-b1s256.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-prefill-b1s256_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-prefill-b1s256_lego_edge.log

echo '[22/45] qwen3-8b-block0-ffn-prefill-b1s1024 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s1024.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s1024.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b1s1024_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b1s1024_lego_edge.log

echo '[23/45] gemma-7b-block0-attention-decode-b8s4096 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-decode-b8s4096.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-decode-b8s4096.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-decode-b8s4096_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-decode-b8s4096_gemmini_os_server.log

echo '[24/45] gemma-7b-block0-ffn-decode-b8s128 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-decode-b8s128.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-decode-b8s128.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-decode-b8s128_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-decode-b8s128_gemmini_os_server.log

echo '[25/45] gemma-7b-block0-ffn-prefill-b8s512 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-prefill-b8s512.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-prefill-b8s512.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b8s512_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b8s512_gemmini_os_server.log

echo '[26/45] llama3-8b-block0-attention-prefill-b8s256 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-prefill-b8s256.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-prefill-b8s256.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-prefill-b8s256_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-prefill-b8s256_gemmini_os_server.log

echo '[27/45] llama3-8b-block0-ffn-prefill-b8s1024 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b8s1024_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b8s1024_gemmini_os_server.log

echo '[28/45] qwen3-8b-block0-attention-decode-b8s4096 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-decode-b8s4096.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-decode-b8s4096.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-decode-b8s4096_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-decode-b8s4096_gemmini_os_server.log

echo '[29/45] qwen3-8b-block0-ffn-decode-b8s128 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s128.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s128.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s128_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s128_gemmini_os_server.log

echo '[30/45] qwen3-8b-block0-ffn-prefill-b8s512 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s512.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s512.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b8s512_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b8s512_gemmini_os_server.log

echo '[31/45] gemma-7b-block0-attention-prefill-b8s256 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-prefill-b8s256.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-prefill-b8s256.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-prefill-b8s256_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-prefill-b8s256_gemmini_ws_server.log

echo '[32/45] gemma-7b-block0-ffn-prefill-b8s1024 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-prefill-b8s1024.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-prefill-b8s1024.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b8s1024_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b8s1024_gemmini_ws_server.log

echo '[33/45] llama3-8b-block0-attention-decode-b8s4096 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-decode-b8s4096.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-decode-b8s4096.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-decode-b8s4096_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-decode-b8s4096_gemmini_ws_server.log

echo '[34/45] llama3-8b-block0-ffn-decode-b8s128 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-decode-b8s128.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-decode-b8s128.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-decode-b8s128_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-decode-b8s128_gemmini_ws_server.log

echo '[35/45] llama3-8b-block0-ffn-prefill-b8s512 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s512.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s512.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b8s512_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b8s512_gemmini_ws_server.log

echo '[36/45] qwen3-8b-block0-attention-prefill-b8s256 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-prefill-b8s256.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-prefill-b8s256.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-prefill-b8s256_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-prefill-b8s256_gemmini_ws_server.log

echo '[37/45] qwen3-8b-block0-ffn-prefill-b8s1024 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s1024.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s1024.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b8s1024_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b8s1024_gemmini_ws_server.log

echo '[38/45] gemma-7b-block0-attention-decode-b8s4096 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-decode-b8s4096.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-decode-b8s4096.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-decode-b8s4096_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-attention-decode-b8s4096_lego_server.log

echo '[39/45] gemma-7b-block0-ffn-decode-b8s128 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-decode-b8s128.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-decode-b8s128.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-decode-b8s128_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-decode-b8s128_lego_server.log

echo '[40/45] gemma-7b-block0-ffn-prefill-b8s512 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-prefill-b8s512.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-prefill-b8s512.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b8s512_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/gemma-7b-block0-ffn-prefill-b8s512_lego_server.log

echo '[41/45] llama3-8b-block0-attention-prefill-b8s256 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-prefill-b8s256.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-prefill-b8s256.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-prefill-b8s256_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-attention-prefill-b8s256_lego_server.log

echo '[42/45] llama3-8b-block0-ffn-prefill-b8s1024 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b8s1024_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/llama3-8b-block0-ffn-prefill-b8s1024_lego_server.log

echo '[43/45] qwen3-8b-block0-attention-decode-b8s4096 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-decode-b8s4096.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-decode-b8s4096.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-decode-b8s4096_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-attention-decode-b8s4096_lego_server.log

echo '[44/45] qwen3-8b-block0-ffn-decode-b8s128 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s128.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s128.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s128_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-decode-b8s128_lego_server.log

echo '[45/45] qwen3-8b-block0-ffn-prefill-b8s512 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s512.mlir -pass-pipeline="model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s512.mlir}" -o /dev/null 2>&1 | tee /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b8s512_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test/logs/qwen3-8b-block0-ffn-prefill-b8s512_lego_server.log

echo 'Pane 7: ALL DONE'
read -p 'Press Enter to close...'
