#!/bin/bash
cd /home/accelgen
echo 'Pane 2: 60 tasks'

echo '[1/60] gemma-7b-block0-attention-decode-b1s512 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-decode-b1s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-decode-b1s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b1s512_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b1s512_lego_edge.log

echo '[2/60] gemma-7b-block0-attention-prefill-b1s1024 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-prefill-b1s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-prefill-b1s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b1s1024_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b1s1024_lego_edge.log

echo '[3/60] gemma-7b-block0-ffn-decode-b1s4096 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-decode-b1s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-decode-b1s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b1s4096_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b1s4096_lego_edge.log

echo '[4/60] llama3-8b-block0-attention-decode-b1s128 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-decode-b1s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-decode-b1s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b1s128_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b1s128_lego_edge.log

echo '[5/60] llama3-8b-block0-attention-prefill-b1s256 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-prefill-b1s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-prefill-b1s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b1s256_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b1s256_lego_edge.log

echo '[6/60] llama3-8b-block0-ffn-decode-b1s512 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-decode-b1s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-decode-b1s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b1s512_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b1s512_lego_edge.log

echo '[7/60] llama3-8b-block0-ffn-prefill-b1s1024 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b1s1024_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b1s1024_lego_edge.log

echo '[8/60] qwen3-8b-block0-attention-decode-b1s4096 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-decode-b1s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-decode-b1s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b1s4096_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b1s4096_lego_edge.log

echo '[9/60] qwen3-8b-block0-ffn-decode-b1s128 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-decode-b1s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-decode-b1s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b1s128_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b1s128_lego_edge.log

echo '[10/60] qwen3-8b-block0-ffn-prefill-b1s256 on lego_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b1s256_lego_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b1s256_lego_edge.log

echo '[11/60] gemma-7b-block0-attention-decode-b8s512 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-decode-b8s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-decode-b8s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b8s512_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b8s512_lego_server.log

echo '[12/60] gemma-7b-block0-attention-prefill-b8s1024 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-prefill-b8s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-prefill-b8s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b8s1024_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b8s1024_lego_server.log

echo '[13/60] gemma-7b-block0-ffn-decode-b8s4096 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-decode-b8s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-decode-b8s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b8s4096_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b8s4096_lego_server.log

echo '[14/60] llama3-8b-block0-attention-decode-b8s128 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-decode-b8s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-decode-b8s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b8s128_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b8s128_lego_server.log

echo '[15/60] llama3-8b-block0-attention-prefill-b8s256 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-prefill-b8s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-prefill-b8s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b8s256_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b8s256_lego_server.log

echo '[16/60] llama3-8b-block0-ffn-decode-b8s512 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-decode-b8s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-decode-b8s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b8s512_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b8s512_lego_server.log

echo '[17/60] llama3-8b-block0-ffn-prefill-b8s1024 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b8s1024_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b8s1024_lego_server.log

echo '[18/60] qwen3-8b-block0-attention-decode-b8s4096 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-decode-b8s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-decode-b8s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b8s4096_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b8s4096_lego_server.log

echo '[19/60] qwen3-8b-block0-ffn-decode-b8s128 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b8s128_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b8s128_lego_server.log

echo '[20/60] qwen3-8b-block0-ffn-prefill-b8s256 on lego_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=lego_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b8s256_lego_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b8s256_lego_server.log

echo '[21/60] gemma-7b-block0-attention-decode-b8s512 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-decode-b8s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-decode-b8s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b8s512_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b8s512_gemmini_os_server.log

echo '[22/60] gemma-7b-block0-attention-prefill-b8s1024 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-prefill-b8s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-prefill-b8s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b8s1024_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b8s1024_gemmini_os_server.log

echo '[23/60] gemma-7b-block0-ffn-decode-b8s4096 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-decode-b8s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-decode-b8s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b8s4096_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b8s4096_gemmini_os_server.log

echo '[24/60] llama3-8b-block0-attention-decode-b8s128 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-decode-b8s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-decode-b8s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b8s128_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b8s128_gemmini_os_server.log

echo '[25/60] llama3-8b-block0-attention-prefill-b8s256 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-prefill-b8s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-prefill-b8s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b8s256_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b8s256_gemmini_os_server.log

echo '[26/60] llama3-8b-block0-ffn-decode-b8s512 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-decode-b8s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-decode-b8s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b8s512_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b8s512_gemmini_os_server.log

echo '[27/60] llama3-8b-block0-ffn-prefill-b8s1024 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b8s1024_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b8s1024_gemmini_os_server.log

echo '[28/60] qwen3-8b-block0-attention-decode-b8s4096 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-decode-b8s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-decode-b8s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b8s4096_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b8s4096_gemmini_os_server.log

echo '[29/60] qwen3-8b-block0-ffn-decode-b8s128 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b8s128_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b8s128_gemmini_os_server.log

echo '[30/60] qwen3-8b-block0-ffn-prefill-b8s256 on gemmini_os_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b8s256_gemmini_os_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b8s256_gemmini_os_server.log

echo '[31/60] gemma-7b-block0-attention-decode-b8s512 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-decode-b8s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-decode-b8s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b8s512_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b8s512_gemmini_ws_server.log

echo '[32/60] gemma-7b-block0-attention-prefill-b8s1024 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-prefill-b8s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-prefill-b8s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b8s1024_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b8s1024_gemmini_ws_server.log

echo '[33/60] gemma-7b-block0-ffn-decode-b8s4096 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-decode-b8s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-decode-b8s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b8s4096_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b8s4096_gemmini_ws_server.log

echo '[34/60] llama3-8b-block0-attention-decode-b8s128 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-decode-b8s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-decode-b8s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b8s128_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b8s128_gemmini_ws_server.log

echo '[35/60] llama3-8b-block0-attention-prefill-b8s256 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-prefill-b8s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-prefill-b8s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b8s256_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b8s256_gemmini_ws_server.log

echo '[36/60] llama3-8b-block0-ffn-decode-b8s512 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-decode-b8s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-decode-b8s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b8s512_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b8s512_gemmini_ws_server.log

echo '[37/60] llama3-8b-block0-ffn-prefill-b8s1024 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b8s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b8s1024_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b8s1024_gemmini_ws_server.log

echo '[38/60] qwen3-8b-block0-attention-decode-b8s4096 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-decode-b8s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-decode-b8s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b8s4096_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b8s4096_gemmini_ws_server.log

echo '[39/60] qwen3-8b-block0-ffn-decode-b8s128 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-decode-b8s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b8s128_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b8s128_gemmini_ws_server.log

echo '[40/60] qwen3-8b-block0-ffn-prefill-b8s256 on gemmini_ws_server'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_server ppu-name=PPU_server datamove-arch-name=PPU_datamove_server config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b8s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b8s256_gemmini_ws_server.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b8s256_gemmini_ws_server.log

echo '[41/60] gemma-7b-block0-attention-decode-b1s512 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-decode-b1s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-decode-b1s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b1s512_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b1s512_gemmini_os_edge.log

echo '[42/60] gemma-7b-block0-attention-prefill-b1s1024 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-prefill-b1s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-prefill-b1s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b1s1024_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b1s1024_gemmini_os_edge.log

echo '[43/60] gemma-7b-block0-ffn-decode-b1s4096 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-decode-b1s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-decode-b1s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b1s4096_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b1s4096_gemmini_os_edge.log

echo '[44/60] llama3-8b-block0-attention-decode-b1s128 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-decode-b1s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-decode-b1s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b1s128_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b1s128_gemmini_os_edge.log

echo '[45/60] llama3-8b-block0-attention-prefill-b1s256 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-prefill-b1s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-prefill-b1s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b1s256_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b1s256_gemmini_os_edge.log

echo '[46/60] llama3-8b-block0-ffn-decode-b1s512 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-decode-b1s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-decode-b1s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b1s512_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b1s512_gemmini_os_edge.log

echo '[47/60] llama3-8b-block0-ffn-prefill-b1s1024 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b1s1024_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b1s1024_gemmini_os_edge.log

echo '[48/60] qwen3-8b-block0-attention-decode-b1s4096 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-decode-b1s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-decode-b1s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b1s4096_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b1s4096_gemmini_os_edge.log

echo '[49/60] qwen3-8b-block0-ffn-decode-b1s128 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-decode-b1s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-decode-b1s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b1s128_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b1s128_gemmini_os_edge.log

echo '[50/60] qwen3-8b-block0-ffn-prefill-b1s256 on gemmini_os_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_os_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b1s256_gemmini_os_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b1s256_gemmini_os_edge.log

echo '[51/60] gemma-7b-block0-attention-decode-b1s512 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-decode-b1s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-decode-b1s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b1s512_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-decode-b1s512_gemmini_ws_edge.log

echo '[52/60] gemma-7b-block0-attention-prefill-b1s1024 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-attention-prefill-b1s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-attention-prefill-b1s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b1s1024_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-attention-prefill-b1s1024_gemmini_ws_edge.log

echo '[53/60] gemma-7b-block0-ffn-decode-b1s4096 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/gemma-7b-block0-ffn-decode-b1s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/gemma-7b-block0-ffn-decode-b1s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b1s4096_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/gemma-7b-block0-ffn-decode-b1s4096_gemmini_ws_edge.log

echo '[54/60] llama3-8b-block0-attention-decode-b1s128 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-decode-b1s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-decode-b1s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b1s128_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-decode-b1s128_gemmini_ws_edge.log

echo '[55/60] llama3-8b-block0-attention-prefill-b1s256 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-attention-prefill-b1s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-attention-prefill-b1s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b1s256_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-attention-prefill-b1s256_gemmini_ws_edge.log

echo '[56/60] llama3-8b-block0-ffn-decode-b1s512 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-decode-b1s512.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-decode-b1s512.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b1s512_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-decode-b1s512_gemmini_ws_edge.log

echo '[57/60] llama3-8b-block0-ffn-prefill-b1s1024 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s1024.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/llama3-8b-block0-ffn-prefill-b1s1024.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b1s1024_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/llama3-8b-block0-ffn-prefill-b1s1024_gemmini_ws_edge.log

echo '[58/60] qwen3-8b-block0-attention-decode-b1s4096 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-attention-decode-b1s4096.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-attention-decode-b1s4096.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b1s4096_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-attention-decode-b1s4096_gemmini_ws_edge.log

echo '[59/60] qwen3-8b-block0-ffn-decode-b1s128 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-decode-b1s128.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-decode-b1s128.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b1s128_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-decode-b1s128_gemmini_ws_edge.log

echo '[60/60] qwen3-8b-block0-ffn-prefill-b1s256 on gemmini_ws_edge'
./build/bin/accelgen-opt benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s256.mlir '-pass-pipeline=model-baseline-accelerator{accelerator-name=gemmini_ws_edge ppu-name=PPU_edge datamove-arch-name=PPU_datamove_edge config-path=/home/accelgen/example_designs mlir-file=benchmark/mlir/qwen3-8b-block0-ffn-prefill-b1s256.mlir}' -o /dev/null 2>&1 | tee /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b1s256_gemmini_ws_edge.log
echo 'EXIT_CODE='$? >> /home/accelgen/baseline_test0412/logs/qwen3-8b-block0-ffn-prefill-b1s256_gemmini_ws_edge.log

echo 'Pane 2: ALL DONE'
read -p 'Press Enter to close...'
