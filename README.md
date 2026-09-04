# AccelGen

AccelGen is an MLIR/CIRCT-based framework for exploring programmable
accelerator data paths for tensor and language-model workloads. It combines
kernel scheduling, architecture-aware design-space exploration, a dedicated
DAP (data-path) dialect, analytical performance/area models, and baseline
comparisons in one workflow.

## Why AccelGen

Modern DNN layers mix matrix multiplication, elementwise math, reductions,
reshapes, broadcasts, and data movement. AccelGen represents that structure at
the MLIR level, searches for a schedule that fits a target resource pool, and
lowers the scheduled computation into explicit compute, memory, and routing
operations.

The framework is intended for architecture studies, not as a production model
runtime. Its outputs include scheduled MLIR, DAP MLIR, cycle/energy estimates,
area breakdowns, and editable analysis artifacts.

## Workflow

```text
PyTorch/model workload
          |
          v
      Linalg MLIR
          |
          v
  generic-op fusion/reordering
          |
          v
  kernel scheduling and DSE
          |
          v
    scheduled MLIR
          |
          +--------------------+
          |                    |
          v                    v
  model-performance     convert-to-dap
  cycles/energy/FLOPs          |
                               v
                           DAP MLIR
                               |
                               v
                         model-area / analysis
```

The main command-line tool is `accelgen-opt`, an MLIR optimizer extended with
AccelGen dialects and passes.

## Capabilities

- Schedule `linalg.generic` kernels against configurable compute, SRAM, register,
  and bandwidth resources.
- Explore fusion, loop ordering, tiling, and unrolling choices with the kernel
  scheduler and DSE engine.
- Lower arithmetic, math, tensor, memory, and routing structure into the DAP
  dialect.
- Estimate latency, FLOPs, SRAM/DRAM traffic, energy, and area.
- Compare the generated design with Timeloop/Accelergy baselines.
- Benchmark the same PyTorch model layers on CUDA, AWS Neuron, and Ascend
  `torch_npu` environments.

## Repository Map

| Path | Purpose |
| --- | --- |
| `include/accelgen/Dialect/` | TableGen definitions for DAP, SPE, and TileGraph dialects |
| `lib/Passes/` | MLIR transformations, scheduling integration, lowering, and analysis passes |
| `lib/DSE/` | Kernel scheduling and design-space exploration implementation |
| `opt/` | `accelgen-opt` command-line tool |
| `benchmark/model/` | PyTorch model builders and layer workloads |
| `benchmark/mlir/` | Source Linalg MLIR workloads |
| `eval/generic/` | Generic scheduled MLIR inputs/outputs |
| `eval/model/` | Scheduled MLIR with model-performance attributes |
| `eval/dap/` | DAP-lowered MLIR |
| `config/` | AccelGen resource configurations such as `edge.json` and `server.json` |
| `example_designs/` | Timeloop/Accelergy architecture and component descriptions |
| `evaluation/` | Reproducible scripts and generated tables/figures |

## Requirements

The native build requires:

- C++17 compiler
- CMake 3.20 or newer
- LLVM MLIR development files
- CIRCT development files
- OpenMP

The baseline design simulations additionally require the Timeloop/Accelergy
toolchain described in [`example_designs/README.md`](example_designs/README.md).
PyTorch backend benchmarks require their vendor-matched environments; see the
backend guides below.

## Build

Set `MLIR_DIR` and `CIRCT_DIR` to the package configuration directories for the
LLVM/MLIR and CIRCT installations on your system:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMLIR_DIR=/path/to/lib/cmake/mlir \
  -DCIRCT_DIR=/path/to/lib/cmake/circt
cmake --build build --parallel
```

The resulting optimizer is:

```text
build/bin/accelgen-opt
```

## Quick Start

Run the scheduler and performance model on one generic workload:

```bash
./build/bin/accelgen-opt \
  benchmark/mlir/llama3-8b-block0-attention-prefill-b1s128.mlir \
  -pass-pipeline='func.func(kernel-schedule{config-path=/path/to/accelgen/config/edge.json max-operation=6}),model-performance{config-path=/path/to/accelgen/config/edge.json}' \
  -o eval/model/llama3-8b-block0-attention-prefill-b1s128-generic-scheduled-edge-6.mlir
```

Lower a scheduled workload to DAP:

```bash
./build/bin/accelgen-opt \
  eval/model/llama3-8b-block0-attention-prefill-b1s128-generic-scheduled-edge-6.mlir \
  -pass-pipeline='convert-to-dap{config-path=/path/to/accelgen/config/edge.json}' \
  -o eval/dap/llama3-8b-block0-attention-prefill-b1s128-dap-edge.mlir
```

The repository scripts automate the workload matrices. For example:

```bash
bash script/run_all_model.sh
bash script/test_dump_dap.sh
```

Inspect each script before running a large matrix because paths, workload
lengths, and output files are configurable at the script level.

## Models and Backends

The included workload matrix contains Llama3-8B, Qwen3-8B, Gemma-7B, and
Qwen3-MoE attention/FFN layers. Backend-specific instructions are available in:

- [`benchmark/neuron/README.md`](benchmark/neuron/README.md)
- [`benchmark/ascend/README.md`](benchmark/ascend/README.md)
- [`benchmark/gpu/stat.py`](benchmark/gpu/stat.py)

The PyTorch backend scripts measure layer latency directly. They do not invoke
the MLIR/DAP pipeline.

## Evaluation

The evaluation directory contains scripts for turning generated logs and MLIR
into tables and figures:

- Cluster topology and partition analysis: [`evaluation/README.md`](evaluation/README.md)
- Model performance: [`evaluation/results/model-performance/README.md`](evaluation/results/model-performance/README.md)
- Orin-scaled edge analysis: [`evaluation/results/model-performance-orin/README.md`](evaluation/results/model-performance-orin/README.md)
- Qwen3-MoE baseline comparison: [`evaluation/results/qwen3-moe-baseline-preserve-batch/README.md`](evaluation/results/qwen3-moe-baseline-preserve-batch/README.md)
- SRAM/MAC/mux area analysis: [`evaluation/results/area-breakdown/README.md`](evaluation/results/area-breakdown/README.md)

Most analysis scripts write both CSV and Markdown outputs, so numerical results
can be reused in reports or post-processed with other tools.

## DAP Dialect

DAP makes the hardware-facing structure explicit. Its operations include:

- Arithmetic and math operations such as `dap.mulf`, `dap.add`, `dap.div`,
  `dap.exp`, and `dap.rsqrt`.
- Data movement and storage operations such as `dap.data_path`, `dap.sram`,
  and `dap.data_node`.
- Routing and shape operations such as `dap.concat`, `dap.replicate`,
  `dap.broadcast`, `dap.decompose`, `dap.compose`, `dap.mux`, and `dap.demux`.

The operation definitions live in
[`include/accelgen/Dialect/Dap/DapOps.td`](include/accelgen/Dialect/Dap/DapOps.td).

## Reproducibility Notes

Generated outputs can be large. Keep source MLIR, configuration JSON, command
lines, and tool versions alongside any reported result. In particular, record
whether a baseline preserves tensor batch dimensions, which bandwidth
interpretation is used, and whether energy is measured or estimated from a
fixed coefficient.

## Development

Pass declarations are maintained in
[`include/accelgen/Passes/AccelgenPasses.td`](include/accelgen/Passes/AccelgenPasses.td),
and pass implementations are under `lib/Passes/`. When adding a dialect
operation, update its TableGen definition and regenerate/rebuild before using it
from C++ or an MLIR pass.
