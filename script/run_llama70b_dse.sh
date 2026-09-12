#!/bin/bash

models=("llama3-70b")
actions=("prefill" "decode")
blocks=(0)
layers=("attention" "ffn")
lengths=(128 256 512 1024 4096)
configs=("edge" "server")

: > ./test/llama_70b.log
# : > ./test/moe.log

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
                        {
                            echo "Running: model=$model action=$action block=$block layer=$layer batch=$batch length=$length config=$config"

                            ./build/bin/accelgen-opt ./eval/generic/${model}-block${block}-${layer}-${action}-b${batch}s${length}-generic.mlir \
                            -pass-pipeline="func.func(kernel-schedule{config-path=/home/accelgen/config/$config.json max-operation=6}), \
                                            model-performance{config-path=/home/accelgen/config/$config.json}" \
                            -o ./eval/model/${model}-block${block}-${layer}-${action}-b${batch}s${length}-generic-scheduled-${config}-6.mlir
                        } 1>> ./test/llama_70b.log 2>/dev/null

                    done
                done
            done
        done
    done
done