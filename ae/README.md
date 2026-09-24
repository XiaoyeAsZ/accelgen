# AccelGen Artifact Evaluation Scripts

This directory provides two reproducible, five-stage evaluation suites. Both
suites generate source MLIR, run the Linalg preparation pipeline, schedule the
kernels, lower the scheduled program to the DAP dialect, and run the model
performance estimator.

## Rapid evaluation

The rapid suite covers block 0 attention and FFN for this configuration:

| Model | Action | Length | Hardware config | Batch |
| --- | --- | ---: | --- | ---: |
| Llama3-8B | prefill | 128 | edge | 1 |

Run the complete pipeline from the repository root:

```bash
bash ae/rapid/run_all.sh
```

To inspect or run one stage at a time:

```bash
bash ae/rapid/01_generate_mlir.sh
bash ae/rapid/02_linalg_opt.sh
bash ae/rapid/03_kernel_schedule.sh
bash ae/rapid/04_lower_to_dap.sh
bash ae/rapid/05_model_performance.sh
```

## Full evaluation

The full suite evaluates all model builders supported by
`benchmark/dump_mlir.py`:

- Models: Llama3-8B, Llama3-70B, Qwen3-8B, Qwen3-MoE, and Gemma-7B
- Actions: prefill and decode
- Layers: attention and FFN from block 0
- Sequence lengths: 128, 256, 512, 1024, and 4096
- Configurations: edge (batch 1) and server (batch 8)

This is 200 workloads per stage and can take a long time, particularly during
kernel scheduling. Run it with:

```bash
bash ae/full/run_all.sh
```

The numbered scripts in `ae/full/` can also be run separately. Each stage
requires the output of the preceding stage.

## Outputs and reruns

Outputs are isolated from checked-in benchmark data:

```text
ae/results/
  rapid/
    mlir/ generic/ scheduled/ dap/ performance/ logs/
  full/
    mlir/ generic/ scheduled/ dap/ performance/ logs/
```

An existing nonempty output is skipped, which allows an interrupted full run to
resume. Set `FORCE=1` to rebuild it. Every stage records the exact shell-escaped
commands in `logs/<stage>-commands.sh` and combined tool output in
`logs/<stage>.log`. Logs are appended across resumed or forced runs. Tool output
is kept in the log by default; use `VERBOSE=1` to stream it to the console too.
The model-performance metrics are written to `logs/performance.log`; its
workload headers match the format consumed by
`evaluation/analyze_performance_log.py`. The `performance/` directory contains
the corresponding MLIR emitted by the analysis pass.

Useful controls:

```bash
# Print the rapid commands without executing them.
DRY_RUN=1 bash ae/rapid/run_all.sh

# Rebuild the rapid pipeline with a different scheduling cluster limit.
FORCE=1 MAX_OPERATION=4 bash ae/rapid/run_all.sh

# Smoke-test a subset of the full matrix.
AE_MODELS="llama3-8b gemma-7b" \
AE_LENGTHS="128" \
bash ae/full/run_all.sh

# Store generated artifacts outside the repository.
AE_RESULTS_DIR=/path/to/results bash ae/rapid/run_all.sh
```

The full suite accepts space-separated `AE_MODELS`, `AE_ACTIONS`, `AE_LAYERS`,
`AE_LENGTHS`, and `AE_CONFIGS` overrides. Supported configurations are `edge`
and `server` because they define the suite's batch mapping.

## Requirements

- `build/bin/accelgen-opt` must exist and be executable.
- Python model dependencies used by `benchmark/dump_mlir.py` must be installed.
- Local model configurations must remain under `benchmark/model/`.
- Hardware configurations must remain under `config/edge.json` and
  `config/server.json`.

Override the executables with `ACCELGEN_OPT=/path/to/accelgen-opt` and
`PYTHON=/path/to/python` when needed.
