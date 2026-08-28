# Performance breakdown summary

Source: `test/performance_breakdown.log`

Execution time is the sum of attention and FFN `Execution time (ms)` values. The `original` column is the logged `timeNaive` estimate, not a separately measured hardware baseline. EDP is computed as `(attention latency + FFN latency) * (attention energy + FFN energy)` using the logged Model Performance values.

The active log contains only the lengths listed below. Missing lengths cannot be reconstructed from this file.

Lengths: 4096

## gemma-7b

Action: `prefill`, config: `edge`, batch: `1`, block: `0`

| Length | Size | Ours execution (ms) | Naive estimate (ms) | Total latency (logged) | Total energy (logged) | EDP (logged units) |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 522 | 522 | 789.615 | 8.89626 | 7024.62 |
| 4096 | 2 | 1353 | 488968 | 717.754 | 7.7189 | 5540.28 |
| 4096 | 3 | 14439 | 1.85266e+08 | 694.161 | 7.39136 | 5130.8 |
| 4096 | 4 | 48897 | 1.18245e+11 | 692.064 | 7.33787 | 5078.28 |
| 4096 | 5 | 95505 | 7.80642e+13 | 675.283 | 7.19878 | 4861.21 |
| 4096 | 6 | 151483 | 3.98764e+16 | 666.894 | 7.06134 | 4709.17 |

## llama3-8b

Action: `prefill`, config: `edge`, batch: `1`, block: `0`

| Length | Size | Ours execution (ms) | Naive estimate (ms) | Total latency (logged) | Total energy (logged) | EDP (logged units) |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 544 | 544 | 805.118 | 10.3923 | 8367.06 |
| 4096 | 2 | 1346 | 532957 | 664.543 | 8.11372 | 5391.92 |
| 4096 | 3 | 21549 | 3.7426e+08 | 605.692 | 7.3957 | 4479.52 |
| 4096 | 4 | 92426 | 3.35844e+11 | 605.168 | 7.36566 | 4457.46 |
| 4096 | 5 | 167246 | 3.21377e+14 | 563.741 | 6.77716 | 3820.56 |
| 4096 | 6 | 245951 | 2.09256e+17 | 563.741 | 6.74798 | 3804.11 |

## qwen3-8b

Action: `prefill`, config: `edge`, batch: `1`, block: `0`

| Length | Size | Ours execution (ms) | Naive estimate (ms) | Total latency (logged) | Total energy (logged) | EDP (logged units) |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 696 | 696 | 759.816 | 10.0356 | 7625.21 |
| 4096 | 2 | 1760 | 653249 | 616.457 | 7.71064 | 4753.28 |
| 4096 | 3 | 25179 | 4.08111e+08 | 556.376 | 6.9725 | 3879.33 |
| 4096 | 4 | 83926 | 3.17107e+11 | 555.197 | 6.93232 | 3848.81 |
| 4096 | 5 | 156462 | 2.6589e+14 | 513.77 | 6.34382 | 3259.27 |
| 4096 | 6 | 232660 | 2.09908e+17 | 513.77 | 6.31464 | 3244.28 |

## qwen3-moe

Action: `prefill`, config: `edge`, batch: `1`, block: `0`

| Length | Size | Ours execution (ms) | Naive estimate (ms) | Total latency (logged) | Total energy (logged) | EDP (logged units) |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 902 | 902 | 1170.61 | 16.8837 | 19764.2 |
| 4096 | 2 | 2150 | 1.40958e+06 | 889.037 | 12.3134 | 10947.1 |
| 4096 | 3 | 35396 | 2.56343e+09 | 769.436 | 10.8463 | 8345.52 |
| 4096 | 4 | 111796 | 4.96145e+12 | 765.701 | 10.7509 | 8231.94 |
| 4096 | 5 | 197377 | 5.21549e+15 | 682.843 | 9.57379 | 6537.4 |
| 4096 | 6 | 302010 | 4.00992e+17 | 682.843 | 9.51543 | 6497.54 |
