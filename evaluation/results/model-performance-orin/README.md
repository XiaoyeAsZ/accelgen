# Orin-scaled edge model performance

Source edge results: `evaluation/results/model-performance/model_performance.csv`

Orin anchor measurements: `test/moe_perf.log` (config=orin only)

The edge results are scaled to an Orin configuration using per-layer ratios at length 128. Llama3-8B ratios are reused for Llama3-8B, Qwen3-8B, and Gemma-7B; Qwen3-MoE ratios are reused for Qwen3-MoE. FLOPs are unchanged.

## Anchor ratios

| Anchor | Action | Layer | Latency scale | Energy scale |
|:--|:--|:--|--:|--:|
| llama3-8b | prefill | attention | 0.32685314 | 0.88557772 |
| llama3-8b | prefill | ffn | 0.27429415 | 0.87137469 |
| llama3-8b | decode | attention | 0.99986229 | 0.99522526 |
| llama3-8b | decode | ffn | 1.00000000 | 0.99152123 |
| qwen3-moe | prefill | attention | 0.34480302 | 0.89311465 |
| qwen3-moe | prefill | ffn | 0.99634709 | 0.97097767 |
| qwen3-moe | decode | attention | 1.00000000 | 0.99421631 |
| qwen3-moe | decode | ffn | 0.99891863 | 0.99402534 |

## gemma-7b

| Action | Config | Batch | Length | Latency (ms) | Energy (J) | FLOPs (G) | Throughput (GFLOPS) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | orin | 1 | 128 | 5.067521 | 0.198771 | 71.163450 | 14043.0498 | 358.0173 |
| prefill | orin | 1 | 256 | 10.202026 | 0.337420 | 142.868520 | 14003.9359 | 423.4150 |
| prefill | orin | 1 | 512 | 20.650388 | 0.627434 | 287.903300 | 13941.7862 | 458.8586 |
| prefill | orin | 1 | 1024 | 42.285795 | 1.277527 | 584.472000 | 13821.9467 | 457.5027 |
| prefill | orin | 1 | 4096 | 193.362085 | 6.190028 | 2545.859100 | 13166.2787 | 411.2839 |
| decode | orin | 1 | 128 | 4.370745 | 0.073644 | 0.555964 | 127.2013 | 7.5493 |
| decode | orin | 1 | 256 | 4.379928 | 0.073878 | 0.558080 | 127.4176 | 7.5540 |
| decode | orin | 1 | 512 | 4.398294 | 0.074346 | 0.562311 | 127.8476 | 7.5634 |
| decode | orin | 1 | 1024 | 4.497032 | 0.076352 | 0.570774 | 126.9223 | 7.4756 |
| decode | orin | 1 | 4096 | 4.902480 | 0.085181 | 0.621548 | 126.7823 | 7.2968 |

## llama3-8b

| Action | Config | Batch | Length | Latency (ms) | Energy (J) | FLOPs (G) | Throughput (GFLOPS) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | orin | 1 | 128 | 3.995648 | 0.154775 | 56.121560 | 14045.6717 | 362.6008 |
| prefill | orin | 1 | 256 | 8.087773 | 0.263678 | 112.790470 | 13945.8014 | 427.7590 |
| prefill | orin | 1 | 512 | 16.485813 | 0.505078 | 227.770400 | 13816.1464 | 450.9606 |
| prefill | orin | 1 | 1024 | 34.128339 | 1.057698 | 464.298400 | 13604.4828 | 438.9708 |
| prefill | orin | 1 | 4096 | 165.184741 | 5.927368 | 2067.378400 | 12515.5531 | 348.7852 |
| decode | orin | 1 | 128 | 3.472352 | 0.058492 | 0.438450 | 126.2688 | 7.4959 |
| decode | orin | 1 | 256 | 3.507931 | 0.059206 | 0.440588 | 125.5976 | 7.4416 |
| decode | orin | 1 | 512 | 3.579089 | 0.060635 | 0.444864 | 124.2953 | 7.3367 |
| decode | orin | 1 | 1024 | 3.721406 | 0.063493 | 0.453417 | 121.8401 | 7.1412 |
| decode | orin | 1 | 4096 | 4.640847 | 0.081416 | 0.504731 | 108.7584 | 6.1994 |

## qwen3-8b

| Action | Config | Batch | Length | Latency (ms) | Energy (J) | FLOPs (G) | Throughput (GFLOPS) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | orin | 1 | 128 | 3.578357 | 0.136165 | 49.681480 | 13883.8790 | 364.8634 |
| prefill | orin | 1 | 256 | 7.253192 | 0.238011 | 99.910330 | 13774.6699 | 419.7716 |
| prefill | orin | 1 | 512 | 14.811291 | 0.457850 | 202.010110 | 13638.9262 | 441.2144 |
| prefill | orin | 1 | 1024 | 30.779296 | 0.965041 | 412.777800 | 13410.8913 | 427.7307 |
| prefill | orin | 1 | 4096 | 151.572856 | 5.550521 | 1861.296900 | 12279.8827 | 335.3373 |
| decode | orin | 1 | 128 | 3.071180 | 0.051771 | 0.388137 | 126.3803 | 7.4972 |
| decode | orin | 1 | 256 | 3.106759 | 0.052486 | 0.390275 | 125.6212 | 7.4359 |
| decode | orin | 1 | 512 | 3.177917 | 0.053915 | 0.394551 | 124.1539 | 7.3181 |
| decode | orin | 1 | 1024 | 3.320234 | 0.056773 | 0.403103 | 121.4081 | 7.1003 |
| decode | orin | 1 | 4096 | 4.239675 | 0.074695 | 0.454418 | 107.1823 | 6.0836 |

## qwen3-moe

| Action | Config | Batch | Length | Latency (ms) | Energy (J) | FLOPs (G) | Throughput (GFLOPS) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | orin | 1 | 128 | 39.711564 | 0.736409 | 57.485700 | 1447.5809 | 78.0622 |
| prefill | orin | 1 | 256 | 42.010966 | 0.857965 | 116.066110 | 2762.7575 | 135.2808 |
| prefill | orin | 1 | 512 | 46.977794 | 1.138583 | 236.511030 | 5034.5282 | 207.7241 |
| prefill | orin | 1 | 1024 | 93.561308 | 1.785668 | 490.537600 | 5242.9537 | 274.7081 |
| prefill | orin | 1 | 4096 | 434.451969 | 8.734234 | 2382.521000 | 5483.9687 | 272.7796 |
| decode | orin | 1 | 128 | 3.556643 | 0.059509 | 0.449107 | 126.2727 | 7.5469 |
| decode | orin | 1 | 256 | 3.624739 | 0.060878 | 0.453383 | 125.0802 | 7.4475 |
| decode | orin | 1 | 512 | 3.760955 | 0.063624 | 0.461937 | 122.8243 | 7.2605 |
| decode | orin | 1 | 1024 | 4.033379 | 0.069138 | 0.479042 | 118.7694 | 6.9288 |
| decode | orin | 1 | 4096 | 5.700395 | 0.101984 | 0.581670 | 102.0403 | 5.7035 |

## Formulas

```text
scaled_attention = edge_attention * latency_scale
scaled_ffn = edge_ffn * latency_scale
scaled_energy = edge_energy * energy_scale
throughput_GFLOPS = total_FLOPs / (scaled_latency_ms * 1e6)
energy_efficiency_GFLOPS_per_J = total_FLOPs / (scaled_energy_J * 1e9)
```

The current `moe_perf.log` contains stale interleaved duplicate records; records with mismatched model/layer FLOPs were ignored.