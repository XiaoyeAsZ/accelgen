#!/bin/bash

model="gemma-7b"
block="0"
layer="ffn"
action="prefill"
batch="8"
sequence="1024"

./build/bin/accelgen-opt ./benchmark/mlir/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}.mlir \
-pass-pipeline="func.func(linalg-generalize-named-ops),builtin.module(mark-generic), \
builtin.module(eliminate-dead-op),builtin.module(fold-tensor-op)" \
-o ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic.mlir