#!/bin/bash

model="llama3-8b"
block="0"
layer="attention"
action="prefill"
batch="8"
sequence="1024"
config="server"

./build/bin/accelgen-opt ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-test.mlir \
-pass-pipeline="func.func(kernel-schedule{config-path=/home/accelgen/config/$config.json})" \
-o ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-test-dump.mlir \
-mlir-print-ir-after-all \
-mlir-print-ir-after-failure \
> ./test/log.txt 2>&1