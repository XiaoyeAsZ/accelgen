#!/bin/bash

model="llama3-8b"
block="0"
layer="attention"
action="prefill"
batch="8"
sequence="1024"
config="/home/accelgen/config/edge.json"

./build/bin/accelgen-opt ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic.mlir \
-pass-pipeline="func.func(kernel-schedule{config-path=$config})" \
-o ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-scheduled.mlir \
2>&1 | tee ./test/log.txt