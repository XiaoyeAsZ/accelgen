# Qwen3-MoE combined baseline

Latency and energy source: `baseline_test_qwen3_moe_preserve_batch/summary.csv`

FLOPs source: `baseline_test_qwen3_moe_preserve_batch/summary.csv`

Attention and FFN are summed before deriving metrics. Baseline architectures use cycles and energy from the summary. Baseline FLOPs use the summary's per-layer FLOP counts. No expert-group FFN scaling is applied because the source preserves batch dimensions. Baseline FFN latency is additionally multiplied by 2 to align aggregate bandwidth assumptions; energy and FLOPs are unchanged. Baseline attention elementwise and data-movement latency is multiplied by 2; attention linear latency, energy, and FLOPs are unchanged. Baseline DRAM energy is normalized from 8 to 16 pJ/bit using recorded 16-bit scalar DRAM accesses. Throughput is reported in GFLOPS and energy efficiency in GFLOPS/J.

## Prefill / edge (batch 1)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 360.7690 | 56.8624 | 1137.0740 | 55.0597 | 1119.3355 | 49.5933 |
| 256 | 643.3531 | 79.6922 | 1163.0434 | 90.8123 | 1265.7179 | 76.2425 |
| 512 | 1110.4124 | 92.3678 | 1166.7699 | 130.5206 | 1258.2965 | 79.4376 |
| 1024 | 2116.1124 | 176.2005 | 1155.3158 | 159.3338 | 1311.1279 | 99.3577 |
| 2048 | 1973.7174 | 176.8047 | 1127.1373 | 161.4448 | 1513.6524 | 70.6578 |
| 4096 | 1768.6255 | 152.8761 | 1080.3672 | 142.6563 | 1068.8847 | 83.3825 |

## Prefill / server (batch 8)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 10476.3022 | 165.8425 | 37157.7418 | 173.9633 | 21766.5400 | 139.3192 |
| 256 | 13808.3686 | 183.5707 | 52937.3018 | 216.8499 | 22739.0652 | 130.0145 |
| 512 | 24004.6378 | 228.9297 | 41622.9827 | 225.3102 | 30590.3522 | 156.5825 |
| 1024 | 56515.8812 | 237.0116 | 38146.7835 | 229.1907 | 33417.0597 | 185.2130 |
| 2048 | 27717.9265 | 197.1040 | 23267.8756 | 190.5497 | 21007.5422 | 151.9579 |
| 4096 | 27048.5173 | 154.7098 | 21201.2786 | 149.9441 | 18910.4591 | 109.9984 |

## Decode / edge (batch 1)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32.0559 | 3.8975 | 86.2680 | 3.9010 | 86.2601 | 3.9019 |
| 256 | 32.1275 | 3.8893 | 86.2071 | 3.8927 | 86.1993 | 3.8937 |
| 512 | 32.2675 | 3.8735 | 86.0889 | 3.8769 | 86.0813 | 3.8773 |
| 1024 | 32.5088 | 3.8437 | 85.6776 | 3.8470 | 85.6703 | 3.8479 |
| 2048 | 32.9780 | 3.7915 | 85.1203 | 3.7948 | 85.1136 | 3.7951 |
| 4096 | 33.8931 | 3.7087 | 84.9230 | 3.7118 | 84.9170 | 3.7115 |

## Decode / server (batch 8)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 294.6795 | 4.7743 | 1098.7875 | 4.7751 | 1098.2179 | 4.7695 |
| 256 | 294.6923 | 4.7387 | 1087.1183 | 4.7395 | 1086.4828 | 4.7337 |
| 512 | 293.3878 | 4.6712 | 1068.2590 | 4.6723 | 1067.6370 | 4.6670 |
| 1024 | 291.7214 | 4.5472 | 1042.7302 | 4.5494 | 1042.1769 | 4.5446 |
| 2048 | 289.1347 | 4.3439 | 1003.2312 | 4.3443 | 1001.8928 | 4.3387 |
| 4096 | 283.6530 | 4.0433 | 932.8693 | 4.0441 | 932.1706 | 4.0401 |

## Notes

- `gemmini_os`, `gemmini_ws`, and `lego` are kept as separate architecture columns.
- No expert-group workload scaling is applied to the preserved-batch source.
- FFN latency uses a `2x` bandwidth correction. The latency correction itself does not scale energy; the DRAM normalization below is applied separately.
- Attention memory latency is defined as `total_cycles - linear_cycles`, and uses a `2x` correction. This includes softmax elementwise work and data movement.
- DRAM energy correction: `accesses * (16 - 8) pJ/bit * 16 bits`, converted to uJ.
- Edge/server rows use the batches encoded by the source files: batch 1 for edge and batch 8 for server.
- Included lengths: 128, 256, 512, 1024, 2048, 4096.
- The summary contains successful rows only; failed rows are excluded and would cause an error if either attention or FFN were missing.
