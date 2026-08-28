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

The scheduled pass does not preserve source operation IDs. The analyzer matches
normalized operation structure and reports structurally ambiguous matches in
the output. A nonzero unmatched count makes the command fail.
