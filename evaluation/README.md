# Evaluation analyses

## Llama3-8B cluster topology

Generate topology figures and cluster statistics from the generic and scheduled
MLIR already produced by the DSE scripts:

```bash
python3 evaluation/analyze_clusters.py
```

The default analysis selects block 0, `max-operation=6`, lengths 128 and 4096,
and all attention/FFN, prefill/decode, and edge/server combinations. It
writes SVG and editable Draw.io versions of the generic-operation cluster
matrix, run/cluster/partition CSV files, and a validation summary to
`evaluation/results/llama3-8b-clusters/`. Matrix columns are grouped by named
model operations and aligned across configurations.

Use `--lengths` to select another set, for example:

```bash
python3 evaluation/analyze_clusters.py --lengths 128 256 512 1024 2048 4096
```

## Performance breakdown

Summarize attention + FFN execution time and EDP from the DSE performance log:

```bash
python3 evaluation/analyze_performance_breakdown.py
```

This writes four model tables (one section per model) to
`evaluation/results/performance-breakdown/performance_breakdown_summary.md` and
the same rows to CSV. The current `performance_breakdown.log` contains only
length 4096; use a log containing additional lengths to obtain a length sweep.

## Area breakdown

Generate the SRAM, BF16 MAC, and mux breakdown directly from the DAP MLIR:

```bash
python3 evaluation/analyze_area_breakdown.py
```

The default workload is block-0 attention prefill at length 4096 for all four
models and edge/server configurations. Results are written to
`evaluation/results/area-breakdown/` as CSV, Markdown, and SVG.

## Ascend PyTorch baseline

Run the existing PyTorch layer workloads with `torch_npu` on Ascend 910B:

```bash
bash script/run_all_ascend_baseline.sh
```

See `benchmark/ascend/README.md` for environment checks and matrix overrides.

## Ascend 910B3 performance summary

Combine measured attention and FFN mean latency from `test/results_all.jsonl`
with FLOPs from `test/performance.log`, using a constant 300 W power estimate:

```bash
python3 evaluation/analyze_ascend_results.py
```

The generated report and CSV are written to
`evaluation/results/ascend-910b3/`.

## Model performance summary

Combine attention and FFN latency, energy, and FLOPs from
`test/performance.log` and compute throughput and energy efficiency:

```bash
python3 evaluation/analyze_performance_log.py
```

The generated report and CSV are written to
`evaluation/results/model-performance/`.

## Orin-scaled edge performance

Scale edge results to the Orin configuration using the valid `config=orin`
anchors in `test/moe_perf.log`:

```bash
python3 evaluation/scale_model_performance_orin.py
```

Llama3-8B anchor ratios are reused for Llama3-8B, Qwen3-8B, and Gemma-7B;
Qwen3-MoE anchor ratios are reused for Qwen3-MoE. Results are written to
`evaluation/results/model-performance-orin/`.

## GPU baseline summary

Combine attention and FFN latency from `test/perf_a100.txt` and
`test/perf_orin.txt` with FLOPs from `test/performance.log`. The default fixed
power assumptions are 250 W for A100 and 9 W for Orin:

```bash
python3 evaluation/analyze_gpu_results.py
```

The generated report and CSV are written to
`evaluation/results/gpu-baselines/`.

## Qwen3-MoE baseline summary

Combine attention and FFN using baseline latency/energy from
`baseline_test_qwen3_moe/summary.csv`. The `ours` results and the common FLOP
counts come from `test/performance.log`:

```bash
python3 evaluation/analyze_baseline_summary.py
```

The four generated tables cover prefill/decode and edge/server, with paired
throughput/energy-efficiency columns for `ours`, `gemmini_os`, `gemmini_ws`, and
`lego`. Baseline FFN latency and energy use an expert-group approximation: `16x`
for edge prefill, `2x` for server prefill, and no scaling for decode. Length 2048
is omitted because `performance.log` has no matching records.

For the rerun that preserves exact batch dimensions, use the summary's own
FLOPs without applying the older FFN approximation:

```bash
python3 evaluation/analyze_baseline_summary.py \
  --summary baseline_test_qwen3_moe_preserve_batch/summary.csv \
  --output-dir evaluation/results/qwen3-moe-baseline-preserve-batch \
  --ffn-batch-mode preserved \
  --flops-source summary \
  --no-include-ours
```

For the quick bandwidth-aligned revision, multiply baseline FFN latency and the
memory-bound portion of attention latency by two while leaving access energy
and FLOPs unchanged. Attention memory latency is approximated as total cycles
minus linear cycles, covering softmax elementwise work and data movement:

```bash
python3 evaluation/analyze_baseline_summary.py \
  --summary baseline_test_qwen3_moe_preserve_batch/summary.csv \
  --output-dir evaluation/results/qwen3-moe-baseline-bandwidth-corrected \
  --ffn-batch-mode preserved \
  --ffn-latency-scale 2 \
  --attention-memory-latency-scale 2 \
  --baseline-dram-energy-pj-per-bit 16 \
  --flops-source summary \
  --no-include-ours
```

The scheduled pass does not preserve source operation IDs. The analyzer matches
normalized operation structure and reports structurally ambiguous matches in
the output. A nonzero unmatched count makes the command fail.
