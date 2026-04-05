#!/bin/bash

models=("llama3-8b" "qwen3-8b" "gemma-7b")
actions=("prefill" "decode")
blocks=(0)
layers=("attention" "ffn")
batches=(8)
lengths=(1024)

for model in "${models[@]}"
do
    for action in "${actions[@]}"
    do
        for block in "${blocks[@]}"
        do
            for layer in "${layers[@]}"
            do
                for batch in "${batches[@]}"
                do
                    for length in "${lengths[@]}"
                    do

                        echo "Running: model=$model action=$action block=$block layer=$layer batch=$batch length=$length"


                        /home/accelgen/build/bin/accelgen-opt "/home/accelgen/benchmark/mlir/${model}-block${block}-${layer}-${action}-b${batch}s${length}.mlir" \
                        -pass-pipeline="linalg-generalize-named-ops,mark-generic,eliminate-dead-op,fuse-generic,fold-tensor-op" \
                        -o "/home/accelgen/test/${model}-block${block}-${layer}-${action}-b${batch}s${length}-generic.mlir" 

                    done
                done
            done
        done
    done
done