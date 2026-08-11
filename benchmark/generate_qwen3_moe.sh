#!/bin/bash

models=("qwen3-moe")
actions=("prefill" "decode")
blocks=(0)
layers=("attention" "ffn")
batches=(8)
lengths=(128)

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

                        python3 dump_mlir.py \
                            --model="$model" \
                            --action="$action" \
                            --block=$block \
                            --layer="$layer" \
                            --batch=$batch \
                            --length=$length 

                    done
                done
            done
        done
    done
done