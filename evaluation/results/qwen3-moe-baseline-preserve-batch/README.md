# Qwen3-MoE combined baseline

Latency and energy source: `baseline_test_qwen3_moe_preserve_batch/summary.csv`

FLOPs source: `baseline_test_qwen3_moe_preserve_batch/summary.csv`

Attention and FFN are summed before deriving metrics. Baseline architectures use cycles and energy from the summary. Baseline FLOPs use the summary's per-layer FLOP counts. No expert-group FFN scaling is applied because the source preserves batch dimensions. Throughput is reported in GFLOPS and energy efficiency in GFLOPS/J.

## Prefill / edge (batch 1)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 701.3484 | 86.8992 | 1926.7771 | 82.5908 | 2035.8735 | 72.5996 |
| 256 | 1222.7444 | 108.0204 | 1958.6798 | 126.9885 | 2271.3503 | 101.6946 |
| 512 | 2030.8603 | 110.7740 | 1953.7776 | 169.5518 | 2242.8128 | 110.0880 |
| 1024 | 3556.6599 | 223.6152 | 1919.0177 | 197.4726 | 2305.3474 | 126.2854 |
| 2048 | 3298.2742 | 222.7897 | 1848.5827 | 199.7714 | 2372.5194 | 98.9528 |
| 4096 | 2934.8587 | 197.2155 | 1739.0470 | 180.8471 | 1678.1627 | 110.2329 |

## Prefill / server (batch 8)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 19072.9788 | 203.1584 | 55377.0126 | 214.4019 | 38105.8717 | 166.1001 |
| 256 | 23441.1759 | 221.4421 | 88256.4184 | 254.7051 | 37913.3595 | 160.8367 |
| 512 | 46426.9848 | 265.0270 | 63842.3286 | 258.5849 | 52843.9015 | 190.1599 |
| 1024 | 99675.9491 | 277.2684 | 61658.9240 | 266.6247 | 57510.9724 | 218.2206 |
| 2048 | 52683.4042 | 237.1507 | 42183.0429 | 229.2716 | 36839.8393 | 184.9597 |
| 4096 | 49838.6073 | 194.8729 | 35284.9923 | 188.6763 | 30841.1557 | 138.9602 |

## Decode / edge (batch 1)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 55.0930 | 6.6139 | 155.4109 | 6.6238 | 155.3972 | 6.6270 |
| 256 | 55.0527 | 6.6003 | 155.0213 | 6.6102 | 155.0078 | 6.6133 |
| 512 | 54.9744 | 6.5741 | 154.2693 | 6.5839 | 154.2562 | 6.5857 |
| 1024 | 54.7884 | 6.5251 | 152.5671 | 6.5346 | 152.5547 | 6.5375 |
| 2048 | 54.4924 | 6.4387 | 149.8656 | 6.4482 | 149.8545 | 6.4493 |
| 4096 | 54.1562 | 6.3017 | 146.6738 | 6.3107 | 146.6644 | 6.3101 |

## Decode / server (batch 8)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 493.7805 | 7.1616 | 1868.9367 | 7.1633 | 1867.9730 | 7.1559 |
| 256 | 492.3426 | 7.1104 | 1846.8114 | 7.1121 | 1845.6388 | 7.1041 |
| 512 | 485.6031 | 7.0133 | 1809.9574 | 7.0158 | 1808.7954 | 7.0085 |
| 1024 | 474.4816 | 6.8351 | 1755.7507 | 6.8389 | 1754.7479 | 6.8323 |
| 2048 | 456.2492 | 6.5416 | 1670.7989 | 6.5425 | 1667.5662 | 6.5336 |
| 4096 | 427.3218 | 6.1052 | 1532.2710 | 6.1070 | 1530.7409 | 6.1009 |

## Notes

- `gemmini_os`, `gemmini_ws`, and `lego` are kept as separate architecture columns.
- No expert-group workload scaling is applied to the preserved-batch source.
- Edge/server rows use the batches encoded by the source files: batch 1 for edge and batch 8 for server.
- Included lengths: 128, 256, 512, 1024, 2048, 4096.
- The summary contains successful rows only; failed rows are excluded and would cause an error if either attention or FFN were missing.
