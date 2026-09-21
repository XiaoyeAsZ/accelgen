#!/bin/bash

models=("llama3-70b")
actions=("prefill")
blocks=(0)
layers=("attention")
lengths=(4096)
configs=("edge" "server")


# : > ./test/dap_lower.log
# : > ./test/runtime.log

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

                            ./build/bin/accelgen-opt ./eval/model/${model}-block${block}-${layer}-${action}-b${batch}s${length}-generic-scheduled-${config}-6.mlir \
                            -pass-pipeline="convert-to-dap{config-path=/home/accelgen/config/${config}.json},model-area" \
                            -o ./eval/dap/${model}-block${block}-${layer}-${action}-b${batch}s${length}-dap-${config}.mlir
                        } >> ./test/dap_lower.log

                    done
                done
            done
        done
    done
done