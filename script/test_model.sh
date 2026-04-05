#!/bin/bash

model="llama3-8b"
block="0"
layer="attention"
action="prefill"
batch="8"
sequence="1024"
config="/home/accelgen/config/edge.json"

./build/bin/accelgen-opt ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-scheduled.mlir \
-pass-pipeline="model-performance{config-path=$config}"