#!/bin/bash

model="llama3-8b"
block="0"
layer="attention"
action="prefill"
batch="8"
sequence="1024"

./build/bin/accelgen-opt ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-test.mlir \
-pass-pipeline="builtin.module(func.func(kernel-schedule))" \
-o ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-test-dump.mlir \
-mlir-print-ir-after-all \
-mlir-print-ir-after-failure \
> ./test/log.txt 2>&1