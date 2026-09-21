# Ascend 910B3 performance

Latency sources: `test/results_all.jsonl`, `test/perf_910b_llama70b.log` (mean latency)

FLOPs sources: `test/performance.log`, `test/llama_70b.log`

Power assumption: `300 W` constant board power

Attention and FFN latency and FLOPs are added before computing throughput. Energy is estimated as constant power multiplied by combined latency; it is not a measured energy value.

## gemma-7b

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | server | 8 | 128 | 3.3520 | 569.3075 | 169842.4241 | 1.005592 | 566.1414 |
| prefill | server | 8 | 256 | 6.0327 | 1142.9479 | 189458.0802 | 1.809817 | 631.5269 |
| prefill | server | 8 | 512 | 11.8415 | 2303.2270 | 194504.4761 | 3.552453 | 648.3483 |
| prefill | server | 8 | 1024 | 25.6393 | 4675.7765 | 182367.4733 | 7.691794 | 607.8916 |
| decode | server | 8 | 128 | 1.3276 | 4.4477 | 3350.2080 | 0.398278 | 11.1674 |
| decode | server | 8 | 256 | 1.2955 | 4.4646 | 3446.3794 | 0.388637 | 11.4879 |
| decode | server | 8 | 512 | 1.3057 | 4.4985 | 3445.3647 | 0.391699 | 11.4845 |
| decode | server | 8 | 1024 | 1.3481 | 4.5662 | 3387.1614 | 0.404426 | 11.2905 |

## llama3-70b

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | server | 8 | 128 | 8.7525 | 1756.9345 | 200736.0168 | 2.625739 | 669.1201 |
| prefill | server | 8 | 256 | 17.1929 | 3522.6268 | 204888.3132 | 5.157874 | 682.9610 |
| prefill | server | 8 | 512 | 36.4981 | 7080.2850 | 193990.2965 | 10.949442 | 646.6343 |
| prefill | server | 8 | 1024 | 79.9770 | 14300.6920 | 178810.1471 | 23.993088 | 596.0338 |
| decode | server | 8 | 128 | 2.2646 | 13.7261 | 6061.1370 | 0.679380 | 20.2038 |
| decode | server | 8 | 256 | 2.4187 | 13.7603 | 5689.0745 | 0.725615 | 18.9636 |
| decode | server | 8 | 512 | 2.3863 | 13.8287 | 5795.0249 | 0.715891 | 19.3167 |
| decode | server | 8 | 1024 | 2.6291 | 13.9655 | 5311.8067 | 0.788744 | 17.7060 |

## llama3-8b

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | server | 8 | 128 | 2.8673 | 448.9724 | 156580.9779 | 0.860205 | 521.9366 |
| prefill | server | 8 | 256 | 5.0187 | 902.3238 | 179793.0657 | 1.505604 | 599.3102 |
| prefill | server | 8 | 512 | 10.1034 | 1822.1626 | 180352.1206 | 3.031008 | 601.1737 |
| prefill | server | 8 | 1024 | 23.9862 | 3714.3879 | 154855.0390 | 7.195868 | 516.1835 |
| decode | server | 8 | 128 | 1.3239 | 3.5076 | 2649.5197 | 0.397158 | 8.8317 |
| decode | server | 8 | 256 | 1.3454 | 3.5247 | 2619.8900 | 0.403609 | 8.7330 |
| decode | server | 8 | 512 | 1.3482 | 3.5589 | 2639.8289 | 0.404448 | 8.7994 |
| decode | server | 8 | 1024 | 1.3704 | 3.6273 | 2646.9186 | 0.411119 | 8.8231 |

## qwen3-8b

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | server | 8 | 128 | 3.1102 | 397.4518 | 127791.8043 | 0.933045 | 425.9727 |
| prefill | server | 8 | 256 | 4.5670 | 799.2826 | 175014.1106 | 1.370088 | 583.3804 |
| prefill | server | 8 | 512 | 9.5940 | 1616.0811 | 168446.4303 | 2.878211 | 561.4881 |
| prefill | server | 8 | 1024 | 23.0219 | 3302.2228 | 143438.2401 | 6.906574 | 478.1275 |
| decode | server | 8 | 128 | 1.7279 | 3.1051 | 1796.9818 | 0.518385 | 5.9899 |
| decode | server | 8 | 256 | 1.7168 | 3.1222 | 1818.5657 | 0.515054 | 6.0619 |
| decode | server | 8 | 512 | 1.6919 | 3.1564 | 1865.6460 | 0.507557 | 6.2188 |
| decode | server | 8 | 1024 | 1.7804 | 3.2248 | 1811.3253 | 0.534111 | 6.0378 |

## qwen3-moe

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | server | 8 | 128 | 7.8868 | 459.8877 | 58311.0533 | 2.366040 | 194.3702 |
| prefill | server | 8 | 256 | 9.8252 | 928.5289 | 94504.6301 | 2.947566 | 315.0154 |
| prefill | server | 8 | 512 | 21.6952 | 1892.0890 | 87212.3735 | 6.508557 | 290.7079 |
| prefill | server | 8 | 1024 | 48.3407 | 3924.3000 | 81180.0014 | 14.502217 | 270.6000 |
| decode | server | 8 | 128 | 3.7562 | 3.5929 | 956.5020 | 1.126874 | 3.1883 |
| decode | server | 8 | 256 | 3.6537 | 3.6271 | 992.7055 | 1.096115 | 3.3090 |
| decode | server | 8 | 512 | 3.8868 | 3.6955 | 950.7898 | 1.166026 | 3.1693 |
| decode | server | 8 | 1024 | 3.9919 | 3.8323 | 960.0300 | 1.197564 | 3.2001 |

## Formulas

```text
combined_latency = attention_mean_latency + ffn_mean_latency
throughput_GFLOPS = total_FLOPs / (combined_latency_ms * 1e6)
estimated_energy_J = 300 W * combined_latency_ms / 1000
energy_efficiency_GFLOPS_per_J = throughput_GFLOPS / 300 W
```
