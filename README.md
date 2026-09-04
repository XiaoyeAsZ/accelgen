# AccelGen

> Automating spatial accelerator generation for large-scale Transformer models.

AccelGen is the compiler and hardware-generation implementation of TranSAGE,
an end-to-end framework for automatically generating spatial accelerators for
large-scale Transformer models. TranSAGE jointly searches operation fusion and
dataflow under multi-operation pipelining, then lowers the selected design to a
reconfigurable datapath and RTL-oriented implementation.

## Core Idea

Existing accelerator generators commonly optimize one operation at a time or
support only fixed fusion patterns. TranSAGE instead treats pipeline clusters as
first-class execution and hardware-generation units. A Transformer computation
is represented as a dependency graph; connected operations are grouped into
pipeline-friendly clusters, and the full model executes as a sequence of those
clusters while reusing shared compute and on-chip memory resources.

The framework addresses two coupled problems:

1. **Operation fusion and dataflow optimization.** A dynamic-programming graph
   partitioner selects clusters, while a decomposed search optimizes tiling,
   unrolling, loop ordering, and inter-operation data movement.
2. **Flexible accelerator generation.** Spatial architecture primitives (SAPs)
   abstract arithmetic, storage, and routing components. The lowering flow
   maps scheduled MLIR clusters to a shared, configurable datapath that can
   support multiple dataflows.

The architectural scope is a single spatial-accelerator compute core with
distributed compute resources, hierarchical on-chip memory, and configurable
processing-unit interconnects. The design can be scaled to larger systems, but
inter-core and inter-package integration is outside this repository's core
scope.

The framework is intended for architecture studies, not as a production model
runtime. Its outputs include scheduled MLIR, DAP/SAP-level MLIR, cycle/energy
estimates, area breakdowns, and RTL-generation inputs.

## Workflow

```text
PyTorch program / Transformer workload
          |
          v
      Linalg MLIR
          |
          v
  operation graph construction
          |
          v
  DP cluster partitioning
  + intra-cluster dataflow DSE
          |
          v
  scheduled cluster MLIR
          |
          +----------------------+
          |                      |
          v                      v
  model-performance     SAP/DAP lowering
  cycles/energy/FLOPs          |
                               v
                 shared reconfigurable datapath
                               |
                               v
                    RTL / binary-generation inputs
```

The main command-line tool is `accelgen-opt`, an MLIR optimizer extended with
AccelGen dialects and passes.

## Capabilities

- Partition topologically ordered operation graphs into connected,
  pipeline-friendly clusters.
- Explore inter-operation pipelining together with tiling, loop ordering, and
  unrolling choices.
- Optimize cluster dataflow under shared compute, SRAM, register, and bandwidth
  constraints.
- Lower arithmetic, math, tensor, memory, and routing structure into spatial
  architecture primitives represented by the DAP dialect.
- Reorganize primitive graphs to share physical resources across dataflows and
  expose runtime-selectable routing.
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

The paper uses SAP (spatial architecture primitive) for the hardware
abstraction. In this repository, the hardware-facing lowering is represented by
the DAP dialect and its operations:

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
