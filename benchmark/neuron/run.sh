#!/usr/bin/env bash
set -euo pipefail

# Defaults to one inexpensive smoke test. Override each list with a
# space-separated environment variable after the first compile succeeds.
read -r -a models <<< "${MODELS:-llama3-8b}"
read -r -a actions <<< "${ACTIONS:-prefill}"
read -r -a layers <<< "${LAYERS:-ffn}"
read -r -a lengths <<< "${LENGTHS:-128}"
read -r -a configs <<< "${CONFIGS:-edge}"

warm_up="${WARM_UP:-10}"
iters="${ITERS:-100}"
output="${OUTPUT:-benchmark/neuron/results.jsonl}"

for model in "${models[@]}"; do
    for action in "${actions[@]}"; do
        for layer in "${layers[@]}"; do
            for length in "${lengths[@]}"; do
                for config in "${configs[@]}"; do
                    if [[ "$config" == "edge" ]]; then
                        batch=1
                    elif [[ "$config" == "server" ]]; then
                        batch=8
                    else
                        echo "Unsupported config: $config" >&2
                        exit 1
                    fi

                    echo "Running model=$model action=$action layer=$layer batch=$batch length=$length config=$config"
                    python3 -m benchmark.neuron.stat \
                        --model "$model" \
                        --action "$action" \
                        --layer "$layer" \
                        --batch "$batch" \
                        --length "$length" \
                        --warm-up "$warm_up" \
                        --iters "$iters" \
                        --output "$output"
                done
            done
        done
    done
done
