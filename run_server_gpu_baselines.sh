#!/bin/bash

models=("llama3-8b")
actions=("prefill")
blocks=(0)
layers=("attention")
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

                        python -m benchmark.gpu.stat \
                                --model=$model \
                                --action=$action \
                                --block=$block \
                                --layer=$layer \
                                --batch=$batch \
                                --length=$length

                    done
                done
            done
        done
    done
done