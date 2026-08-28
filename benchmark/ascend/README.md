# PyTorch workloads on Ascend

This benchmark calls the existing `benchmark.model.*.build_model()` functions
directly with `torch_npu` and an Ascend `npu` device. It does not invoke
AccelGen, MLIR, DAP, or the AWS Neuron backend.

Before running, activate the CANN environment required by your installation and
use a Python environment whose `torch` and `torch_npu` versions are matched to
the installed CANN and 910B driver/firmware. A typical verification is:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())'
```

Run one smoke test:

```bash
python3 -m benchmark.ascend.stat \
  --model llama3-8b \
  --action prefill \
  --layer ffn \
  --batch 1 \
  --length 128 \
  --device npu:0
```

Run the complete matrix:

```bash
bash script/run_all_ascend_baseline.sh
```

The complete runner covers four models, prefill/decode, attention/FFN, edge
and server batches, and lengths 128, 256, 512, 1024, and 4096 (160 workloads).
Results are appended to `benchmark/ascend/results_all.jsonl`, and each
workload has a log under `benchmark/ascend/logs/`; the aggregate script output
is written to `benchmark/ascend/run.log`. Override `RUN_LOG` to choose another
aggregate log path. Override `MODELS`,
`ACTIONS`, `LAYERS`, `LENGTHS`, or `CONFIGS` to reduce the matrix.

Check the generated commands without executing NPU work:

```bash
DRY_RUN=1 bash script/run_all_ascend_baseline.sh
```

The timing loop performs warm-up calls, synchronizes the NPU, then synchronizes
after every measured call. Compilation time is not included because this is an
eager PyTorch/torch_npu benchmark. Decode workloads have fixed KV-cache shapes
determined by `length`.

Do not install the repository's CUDA-oriented `requirements.txt` wholesale into
the Ascend environment. Keep the vendor-matched PyTorch/torch_npu/CANN stack
and install only missing model dependencies such as `transformers`.
