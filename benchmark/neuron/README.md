# PyTorch workloads on AWS Neuron

This benchmark calls the existing `benchmark.model.*.build_model()` functions
directly. It does not invoke AccelGen, MLIR, DAP, or the analytical performance
model.

Activate the PyTorch Neuron environment and verify the device:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
neuron-ls
python -m pip install "transformers==4.56.2"
```

Run one FFN smoke test first:

```bash
python3 -m benchmark.neuron.stat \
  --model llama3-8b \
  --action prefill \
  --layer ffn \
  --batch 1 \
  --length 128
```

The compiled artifact is cached by model, layer, action, batch, and length.
Compilation time is reported separately and is never included in inference
latency. Use `--recompile` to replace an artifact.

After the smoke test succeeds, run a matrix with environment variables:

```bash
MODELS="llama3-8b qwen3-8b gemma-7b qwen3-moe" \
ACTIONS="prefill decode" \
LAYERS="attention ffn" \
LENGTHS="128 4096" \
CONFIGS="edge server" \
ITERS=100 \
bash benchmark/neuron/run.sh
```

To run the complete framework matrix directly, use:

```bash
bash script/run_all_inferentia_baseline.sh
```

The complete runner covers four models, prefill/decode, attention/FFN, edge
and server batches, and lengths 128, 256, 512, 1024, and 4096 (160 workloads).
Override `MODELS`, `ACTIONS`, `LAYERS`, `LENGTHS`, or `CONFIGS` to reduce the
matrix. Results are appended to `benchmark/neuron/results_all.jsonl`, and each
workload has a separate log under `benchmark/neuron/logs/`.

To verify the generated commands without compiling anything:

```bash
DRY_RUN=1 bash script/run_all_inferentia_baseline.sh
```

The runner continues after a compilation failure and exits nonzero if any
workload failed. Qwen3-MoE may require a larger Inf2 instance because its
static grouped expert weights are substantially larger than the other layer
benchmarks.

Neuron tracing uses fixed shapes. Each batch, length, action, and layer tuple is
compiled and cached separately. Decode inputs include a KV cache whose shape
depends on `length`.

Do not install the repository's CUDA-oriented `requirements.txt` into the
Neuron virtual environment. Install only missing model dependencies while
preserving the DLAMI's matching `torch`, `torch-neuronx`, and `neuronx-cc`
versions.
