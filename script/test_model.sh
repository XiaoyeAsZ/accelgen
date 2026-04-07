#!/bin/bash

model="llama3-8b"
block="0"
layer="attention"
action="prefill"
batch="8"
sequence="1024"
config="server"

./build/bin/accelgen-opt ./test/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-scheduled-${config}.mlir \
-pass-pipeline="model-performance{config-path=/home/accelgen/config/$config.json}"