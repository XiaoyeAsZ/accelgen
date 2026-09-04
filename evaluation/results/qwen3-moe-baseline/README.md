# Qwen3-MoE combined baseline

Latency and energy source: `baseline_test_qwen3_moe/summary.csv`

FLOPs source: `test/performance.log`

Attention and FFN are summed before deriving metrics. Baseline architectures use cycles and energy from the summary. Baseline FLOPs use matching records from `performance.log`. Baseline FFN latency and energy are scaled by 16 for edge prefill and 2 for server prefill to approximate the full expert-group workload. `ours` uses latency, energy, and FLOPs from `performance.log`. Throughput is reported in GFLOPS and energy efficiency in GFLOPS/J.

## Prefill / edge (batch 1)

| Length | ours throughput (GFLOPS) | ours efficiency (GFLOPS/J) | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1332.8839 | 75.2629 | 702.5719 | 79.8916 | 1942.2624 | 82.9913 | 2171.5196 | 78.2081 |
| 256 | 2366.0809 | 129.8508 | 1328.0284 | 138.3661 | 1989.7164 | 128.3083 | 2502.1435 | 101.9946 |
| 512 | 3806.8960 | 197.9518 | 2370.6063 | 193.4784 | 2016.6032 | 174.3033 | 3343.8732 | 125.1179 |
| 1024 | 3817.4890 | 258.9542 | 3781.3169 | 237.1067 | 2040.6183 | 209.8687 | 3711.7592 | 171.6050 |
| 4096 | 3489.1182 | 250.3851 | 3499.9850 | 236.2620 | 2118.3017 | 217.4383 | 2864.8605 | 136.4813 |

## Prefill / server (batch 8)

| Length | ours throughput (GFLOPS) | ours efficiency (GFLOPS/J) | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 77389.8999 | 353.6907 | 36229.9560 | 222.1885 | 87018.0342 | 219.5438 | 37215.9326 | 159.4618 |
| 256 | 126698.2322 | 455.7965 | 61576.9318 | 257.4514 | 101837.7495 | 266.0544 | 40966.1742 | 170.8184 |
| 512 | 177568.6033 | 513.7050 | 28356.1389 | 200.6296 | 87994.5788 | 289.9032 | 49800.4968 | 198.3050 |
| 1024 | 155484.6956 | 514.8624 | 113525.6383 | 296.7061 | 81721.3822 | 287.5065 | 53236.0138 | 202.3502 |
| 4096 | 92496.3305 | 362.9790 | 47883.6405 | 228.3787 | 41810.2197 | 222.9620 | 32598.0245 | 172.6160 |

## Decode / edge (batch 1)

| Length | ours throughput (GFLOPS) | ours efficiency (GFLOPS/J) | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 126.1819 | 7.5023 | 55.5255 | 6.6522 | 156.2703 | 6.6622 | 156.2564 | 6.6654 |
| 256 | 124.9920 | 7.4035 | 55.9187 | 6.6767 | 156.7320 | 6.6866 | 156.7182 | 6.6898 |
| 512 | 122.7408 | 7.2176 | 56.6998 | 6.7247 | 157.6380 | 6.7346 | 157.6243 | 6.7377 |
| 1024 | 118.6941 | 6.8879 | 58.1959 | 6.8174 | 159.0564 | 6.8272 | 159.0429 | 6.8303 |
| 4096 | 101.9945 | 5.6701 | 66.9254 | 7.2974 | 168.4465 | 7.3067 | 168.4341 | 7.3097 |

## Decode / server (batch 8)

| Length | ours throughput (GFLOPS) | ours efficiency (GFLOPS/J) | gemmini_os throughput (GFLOPS) | gemmini_os efficiency (GFLOPS/J) | gemmini_ws throughput (GFLOPS) | gemmini_ws efficiency (GFLOPS/J) | lego throughput (GFLOPS) | lego efficiency (GFLOPS/J) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1396.5258 | 10.4770 | 502.7021 | 7.2233 | 1885.0123 | 7.2251 | 1883.7768 | 7.2173 |
| 256 | 1373.7856 | 10.2788 | 505.6413 | 7.2318 | 1877.9874 | 7.2336 | 1876.7727 | 7.2259 |
| 512 | 1331.4082 | 9.9142 | 511.7280 | 7.2492 | 1869.0634 | 7.2510 | 1867.8825 | 7.2434 |
| 1024 | 1257.9559 | 9.2897 | 524.7610 | 7.2817 | 1866.2569 | 7.2835 | 1865.1143 | 7.2761 |
| 4096 | 996.0917 | 7.1289 | 596.5361 | 7.4407 | 1846.2289 | 7.4422 | 1845.3042 | 7.4358 |

## Notes

- `gemmini_os`, `gemmini_ws`, and `lego` are kept as separate architecture columns.
- `ours` converts logged milliseconds to cycles at the framework's 1 GHz assumption and logged joules to uJ before applying the shared formulas.
- Baseline FFN prefill scaling approximates the 16 expert groups that the baseline pass modeled as 1 group on edge and 8 groups on server. FLOPs are not scaled again because `performance.log` already contains the full workload.
- Attention is not scaled; its internal head dimensions require per-operation correction rather than one global factor.
- Edge/server rows use the batches encoded by the source files: batch 1 for edge and batch 8 for server.
- Included lengths: 128, 256, 512, 1024, 4096.
- The summary contains successful rows only; failed rows are excluded and would cause an error if either attention or FFN were missing.
