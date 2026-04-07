#!/bin/bash

model="llama3-8b"
block="0"
layer="attention"
action="prefill"
batch="8"
sequence="1024"
config="server"

./build/bin/accelgen-opt ./eval/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic.mlir \
-pass-pipeline="func.func(kernel-schedule{config-path=/home/accelgen/config/$config.json})" \
-o ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-scheduled-${config}.mlir \
2>&1 | tee ./test/log.txt