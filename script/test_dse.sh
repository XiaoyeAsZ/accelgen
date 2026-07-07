#!/bin/bash

model="llama3-8b"
block="0"
layer="attention"
action="prefill"
batch="1"
sequence="4096"
config="edge"

./build/bin/accelgen-opt ./eval/generic/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic.mlir \
-pass-pipeline="func.func(kernel-schedule{config-path=/home/accelgen/config/$config.json max-operation=6}), \
                                        model-performance{config-path=/home/accelgen/config/$config.json}" \
-o ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-test-dump.mlir \
-mlir-print-ir-after-all \
-mlir-print-ir-after-failure \
> ./test/log.txt 2>&1