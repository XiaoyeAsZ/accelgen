#!/bin/bash

models=("llama3-8b" "qwen3-8b" "gemma-7b")
actions=("prefill")
blocks=(0)
layers=("attention")
lengths=(128 256 512 1024 4096)
configs=("edge")
sizes=(1 2 3 4 5 6)

: > ./test/performance.log
: > ./test/runtime.log

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
                    for size in "${sizes[@]}"
                    do
                        
                        if [ "$config" = "edge" ]; then
                            batch=1
                        else
                            batch=8
                        fi

                        for length in "${lengths[@]}"
                        do
                            {
                                echo "Running: model=$model action=$action block=$block layer=$layer batch=$batch length=$length config=$config size=$size"

                                ./build/bin/accelgen-opt ./eval/generic/${model}-block${block}-${layer}-${action}-b${batch}s${length}-generic.mlir \
                                -pass-pipeline="func.func(kernel-schedule{config-path=/home/accelgen/config/$config.json max-operation=$size}), \
                                                model-performance{config-path=/home/accelgen/config/$config.json}" \
                                -o ./eval/model/${model}-block${block}-${layer}-${action}-b${batch}s${length}-generic-scheduled-${config}-${size}.mlir
                            } 1>> ./test/performance_breakdown.log 2>/dev/null

                        done

                    done

                done
            done
        done
    done
done