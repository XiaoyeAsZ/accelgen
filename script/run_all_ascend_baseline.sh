#!/usr/bin/env bash
set -u
set -o pipefail

# PyTorch/torch_npu equivalent of the GPU baseline matrix. This uses eager
# PyTorch on Ascend and does not invoke AccelGen, MLIR, DAP, or Neuron.
cd /home/accelgen

read -r -a models <<< "${MODELS:-llama3-8b qwen3-8b gemma-7b qwen3-moe}"
read -r -a actions <<< "${ACTIONS:-prefill decode}"
read -r -a layers <<< "${LAYERS:-attention ffn}"
read -r -a lengths <<< "${LENGTHS:-128 256 512 1024 4096}"
read -r -a configs <<< "${CONFIGS:-edge server}"

block="${BLOCK:-0}"
device="${DEVICE:-npu:0}"
warm_up="${WARM_UP:-10}"
iters="${ITERS:-100}"
result_file="${OUTPUT:-benchmark/ascend/results_all.jsonl}"
log_dir="${LOG_DIR:-benchmark/ascend/logs}"
run_log="${RUN_LOG:-benchmark/ascend/run.log}"
dry_run="${DRY_RUN:-0}"

if [[ "$dry_run" != "1" ]]; then
    mkdir -p "$log_dir" "$(dirname "$result_file")"
    mkdir -p "$(dirname "$run_log")"
    if [[ "${RESET_RESULTS:-1}" == "1" ]]; then
        : > "$result_file"
    fi
    if [[ "${RESET_RUN_LOG:-1}" == "1" ]]; then
        : > "$run_log"
    fi
    exec > >(tee -a "$run_log") 2>&1
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
                    echo "[$total] model=$model action=$action layer=$layer batch=$batch length=$length config=$config device=$device"
                    ascend_cmd=(python3 -m benchmark.ascend.stat \
                        --model "$model" \
                        --action "$action" \
                        --layer "$layer" \
                        --block "$block" \
                        --batch "$batch" \
                        --length "$length" \
                        --config "$config" \
                        --device "$device" \
                        --warm-up "$warm_up" \
                        --iters "$iters" \
                        --output "$result_file" \
                    )
                    if [[ "$dry_run" == "1" ]]; then
                        printf 'DRY RUN:'
                        printf ' %q' "${ascend_cmd[@]}"
                        printf '\n'
                        continue
                    fi
                    if "${ascend_cmd[@]}" > "$log_file" 2>&1; then
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
