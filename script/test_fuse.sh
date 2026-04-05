#!/bin/bash

model="llama3-8b"
block="0"
layer="attention"
action="prefill"
batch="8"
sequence="1024"

./build/bin/accelgen-opt ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-test.mlir \
-pass-pipeline="fuse-generic" \
-o ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-test-dump.mlir 