#!/bin/bash


models=("llama3-70b")
actions=("prefill" "decode")
blocks=(0)
layers=("attention" "ffn")
lengths=(128 256 512 1024 4096)
configs=("server" "edge")

for model in "${models[@]}"
do
    for action in "${actions[@]}"
    do
        for block in "${blocks[@]}"
        do
            for layer in "${layers[@]}"
            do
                for config in "${configs[@]}"
                do
                    if [ "$config" = "edge" ]; then
                        batch=1
                    else
                        batch=8
                    fi

                    for length in "${lengths[@]}"
                    do
                        /home/accelgen/build/bin/accelgen-opt "/home/accelgen/benchmark/mlir/${model}-block${block}-${layer}-${action}-b${batch}s${length}.mlir" \
                        -pass-pipeline="linalg-generalize-named-ops,mark-generic,eliminate-dead-op,fuse-generic,fold-tensor-op" \
                        -o "/home/accelgen/eval/generic/${model}-block${block}-${layer}-${action}-b${batch}s${length}-generic.mlir" \
                        -mlir-print-ir-after-all \
                        -mlir-print-ir-after-failure
                    done
                done
            done
        done
    done
done