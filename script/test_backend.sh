#!/bin/bash

model="llama3-8b"
block="0"
layer="attention"
action="prefill"
batch="8"
sequence="1024"

./build/bin/accelgen-opt ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-scheduled.mlir \
-pass-pipeline="convert-to-dap" \
-o ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-dap.mlir