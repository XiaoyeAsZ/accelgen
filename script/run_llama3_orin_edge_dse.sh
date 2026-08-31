#!/bin/bash

models=("llama3-8b" "qwen3-moe")
actions=("prefill" "decode")
blocks=(0)
layers=("attention" "ffn")
lengths=(128)
configs=("orin")

: > ./test/moe_perf.log
: > ./test/moe.log

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
                    batch=1


                    for length in "${lengths[@]}"
                    do
                        {
                            echo "Running: model=$model action=$action block=$block layer=$layer batch=$batch length=$length config=$config"

                            ./build/bin/accelgen-opt ./eval/generic/${model}-block${block}-${layer}-${action}-b${batch}s${length}-generic.mlir \
                            -pass-pipeline="func.func(kernel-schedule{config-path=/home/accelgen/config/${config}.json max-operation=6}), \
                                            model-performance{config-path=/home/accelgen/config/${config}.json}" \
                            -o ./eval/model/${model}-block${block}-${layer}-${action}-b${batch}s${length}-generic-scheduled-${config}-6.mlir
                        } 1>> ./test/moe_perf.log 2>./test/moe.log

                    done
                done
            done
        done
    done
done
