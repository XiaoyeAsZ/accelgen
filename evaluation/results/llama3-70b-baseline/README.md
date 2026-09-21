# Llama3-70B combined baseline

Latency and energy source: `baseline_test_70b_new/summary.csv`

FLOPs source: `baseline_test_70b_new/summary.csv`

Attention and FFN are summed before deriving metrics. Baseline architectures use cycles and energy from the summary. Baseline FLOPs use the summary's per-layer FLOP counts. No workload scaling is applied because the source preserves input dimensions. Throughput is reported in GFLOPS and energy efficiency in GFLOPS/J.

## Prefill / edge (batch 1)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 3235.4575 | 201.4936 | 1810.8266 | 181.1441 | 2334.9460 | 104.1846 |
| 256 | 3577.3410 | 248.6611 | 1913.5919 | 221.9234 | 2885.6743 | 184.3978 |
| 512 | 3756.6318 | 277.6453 | 1964.5613 | 246.7841 | 2565.1043 | 146.5279 |
| 1024 | 3812.9378 | 285.3298 | 1981.2287 | 253.7289 | 3105.5042 | 127.0166 |
| 4096 | 3607.2007 | 265.0107 | 1931.3090 | 239.6203 | 1819.9987 | 120.3719 |

## Prefill / server (batch 8)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 38045.6786 | 247.9524 | 92776.1994 | 256.6315 | 36688.7427 | 164.9652 |
| 256 | 64256.1202 | 301.9739 | 110446.0101 | 300.3660 | 80559.9498 | 252.4889 |
| 512 | 196192.0271 | 348.1964 | 99499.1774 | 318.7783 | 45198.5204 | 233.9294 |
| 1024 | 173751.7415 | 345.5334 | 98427.2737 | 323.6029 | 66264.4278 | 218.5697 |
| 4096 | 110385.2698 | 294.0170 | 70317.9180 | 281.6187 | 49014.5411 | 200.8379 |

## Decode / edge (batch 1)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 42.6867 | 3.7417 | 85.3893 | 3.7449 | 85.3800 | 3.7461 |
| 256 | 42.7044 | 3.7446 | 85.4545 | 3.7478 | 85.4452 | 3.7490 |
| 512 | 42.7395 | 3.7504 | 85.5841 | 3.7536 | 85.5748 | 3.7547 |
| 1024 | 42.8023 | 3.7619 | 85.8148 | 3.7651 | 85.8055 | 3.7663 |
| 4096 | 43.2058 | 3.8277 | 87.3258 | 3.8311 | 87.3167 | 3.8319 |

## Decode / server (batch 8)

| Length | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 480.4670 | 6.3420 | 1631.8925 | 6.3442 | 1632.6471 | 6.3436 |
| 256 | 480.1397 | 6.3332 | 1627.9196 | 6.3354 | 1628.6193 | 6.3346 |
| 512 | 478.4679 | 6.3158 | 1621.0989 | 6.3182 | 1621.7771 | 6.3175 |
| 1024 | 475.5222 | 6.2815 | 1610.5301 | 6.2841 | 1611.2048 | 6.2835 |
| 4096 | 459.4588 | 6.0980 | 1552.2816 | 6.1002 | 1552.5813 | 6.0990 |

## Notes

- `gemmini_os`, `gemmini_ws`, and `lego` are kept as separate architecture columns.
- No workload scaling is applied because the source preserves input dimensions.
- Edge/server rows use the batches encoded by the source files: batch 1 for edge and batch 8 for server.
- Included lengths: 128, 256, 512, 1024, 4096.
- The summary contains successful rows only; failed rows are excluded and would cause an error if either attention or FFN were missing.
