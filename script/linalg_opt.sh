#!/bin/bash

model="llama3-8b"
block="0"
layer="attention"
action="prefill"
batch="8"
sequence="1024"

./build/bin/accelgen-opt ./benchmark/mlir/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}.mlir \
-pass-pipeline="linalg-generalize-named-ops,mark-generic,eliminate-dead-op,fuse-generic,fold-tensor-op" \
-o ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic.mlir 