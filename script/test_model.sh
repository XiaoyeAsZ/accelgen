#!/bin/bash

model="llama3-8b"
block="0"
layer="attention"
action="prefill"
batch="1"
sequence="128"
config="edge"

./build/bin/accelgen-opt ./eval/model/${model}-block${block}-${layer}-${action}-b${batch}s${sequence}-generic-scheduled-${config}.mlir \
-pass-pipeline="model-performance{config-path=/home/accelgen/config/$config.json}"