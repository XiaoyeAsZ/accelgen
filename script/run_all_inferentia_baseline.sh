#!/usr/bin/env bash
set -u
set -o pipefail

# PyTorch/Neuron equivalent of the GPU baseline matrix. This intentionally does
# not run AccelGen, MLIR, DAP, or the max-operation DSE sizes.
cd /home/accelgen

read -r -a models <<< "${MODELS:-llama3-8b qwen3-8b gemma-7b qwen3-moe}"
read -r -a actions <<< "${ACTIONS:-prefill decode}"
read -r -a layers <<< "${LAYERS:-attention ffn}"
read -r -a lengths <<< "${LENGTHS:-128 256 512 1024 4096}"
read -r -a configs <<< "${CONFIGS:-edge server}"

block="${BLOCK:-0}"
warm_up="${WARM_UP:-10}"
iters="${ITERS:-100}"
artifact_dir="${ARTIFACT_DIR:-benchmark/neuron/artifacts}"
compiler_dir="${COMPILER_WORKDIR:-benchmark/neuron/compiler}"
result_file="${OUTPUT:-benchmark/neuron/results_all.jsonl}"
log_dir="${LOG_DIR:-benchmark/neuron/logs}"
dry_run="${DRY_RUN:-0}"

if [[ "$dry_run" != "1" ]]; then
    mkdir -p "$log_dir" "$artifact_dir" "$compiler_dir"
    mkdir -p "$(dirname "$result_file")"
    if [[ "${RESET_RESULTS:-1}" == "1" ]]; then
        : > "$result_file"
    fi
fi

total=0
failed=0
for model in "${models[@]}"; do
    for action in "${actions[@]}"; do
        for layer in "${layers[@]}"; do
            for length in "${lengths[@]}"; do
                for config in "${configs[@]}"; do
                    case "$config" in
                        edge) batch=1 ;;
                        server) batch=8 ;;
                        *) echo "Unsupported config: $config" >&2; exit 2 ;;
                    esac

                    total=$((total + 1))
                    stem="${model}-block${block}-${layer}-${action}-b${batch}s${length}-${config}"
                    log_file="${log_dir}/${stem}.log"
                    echo "[$total] model=$model action=$action layer=$layer batch=$batch length=$length config=$config"
                    neuron_cmd=(python3 -m benchmark.neuron.stat \
                        --model "$model" \
                        --action "$action" \
                        --layer "$layer" \
                        --block "$block" \
                        --batch "$batch" \
                        --length "$length" \
                        --config "$config" \
                        --warm-up "$warm_up" \
                        --iters "$iters" \
                        --artifact-dir "$artifact_dir" \
                        --compiler-workdir "$compiler_dir" \
                        --output "$result_file" \
                    )
                    if [[ "$dry_run" == "1" ]]; then
                        printf 'DRY RUN:'
                        printf ' %q' "${neuron_cmd[@]}"
                        printf '\n'
                        continue
                    fi
                    if "${neuron_cmd[@]}" > "$log_file" 2>&1; then
                        sed -n '/^{/,$p' "$log_file" || cat "$log_file"
                    else
                        status=$?
                        failed=$((failed + 1))
                        echo "FAILED (exit ${status}); see ${log_file}" >&2
                        cat "$log_file" >&2
                    fi
                done
            done
        done
    done
done

echo "Completed ${total} workloads; failures=${failed}"
echo "Results: ${result_file}"
echo "Logs: ${log_dir}"
exit "$([[ "$failed" -eq 0 ]] && echo 0 || echo 1)"
