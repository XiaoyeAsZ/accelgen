#!/bin/bash

models=("gemma-7b" "qwen3-moe")
actions=("prefill")
blocks=(0)
layers=("attention")
lengths=(4096)
configs=("server")


: > ./test/area.log


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

                            ./build/bin/accelgen-opt ./eval/dap/${model}-block${block}-${layer}-${action}-b${batch}s${length}-dap-${config}.mlir \
                            -pass-pipeline="model-area" 
                        } 1>> ./test/area.log 2>/dev/null
                    done
                done
            done
        done
    done
done