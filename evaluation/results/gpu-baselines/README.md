# GPU baseline performance

A100 latency source: `test/perf_a100.txt`

Orin latency source: `test/perf_orin.txt`

FLOPs source: `test/performance.log`

Attention and FFN mean latency and FLOPs are added before computing throughput. Energy uses a fixed-power estimate and is not a measured value.

## A100

Power assumption: `250 W`

### gemma-7b

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | server | 8 | 128 | 3.2825 | 569.3075 | 173437.1668 | 0.820625 | 693.7487 |
| prefill | server | 8 | 256 | 6.8707 | 1142.9479 | 166351.0123 | 1.717675 | 665.4040 |
| prefill | server | 8 | 512 | 14.0543 | 2303.2270 | 163880.5917 | 3.513575 | 655.5224 |
| prefill | server | 8 | 1024 | 29.6796 | 4675.7765 | 157541.7627 | 7.419900 | 630.1671 |
| decode | server | 8 | 128 | 0.9659 | 4.4477 | 4604.7369 | 0.241475 | 18.4189 |
| decode | server | 8 | 256 | 0.9576 | 4.4646 | 4662.3226 | 0.239400 | 18.6493 |
| decode | server | 8 | 512 | 0.9671 | 4.4985 | 4651.5246 | 0.241775 | 18.6061 |
| decode | server | 8 | 1024 | 1.1286 | 4.5662 | 4045.8870 | 0.282150 | 16.1835 |

### llama3-8b

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | server | 8 | 128 | 2.7346 | 448.9724 | 164182.1180 | 0.683650 | 656.7285 |
| prefill | server | 8 | 256 | 5.2217 | 902.3238 | 172802.6888 | 1.305425 | 691.2108 |
| prefill | server | 8 | 512 | 11.4577 | 1822.1626 | 159033.8899 | 2.864425 | 636.1356 |
| prefill | server | 8 | 1024 | 26.4015 | 3714.3879 | 140688.5177 | 6.600375 | 562.7541 |
| decode | server | 8 | 128 | 0.8333 | 3.5076 | 4209.2847 | 0.208325 | 16.8371 |
| decode | server | 8 | 256 | 0.8323 | 3.5247 | 4234.8934 | 0.208075 | 16.9396 |
| decode | server | 8 | 512 | 0.8289 | 3.5589 | 4293.5355 | 0.207225 | 17.1741 |
| decode | server | 8 | 1024 | 0.9966 | 3.6273 | 3639.7062 | 0.249150 | 14.5588 |

### qwen3-8b

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | server | 8 | 128 | 4.2234 | 397.4518 | 94107.0796 | 1.055850 | 376.4283 |
| prefill | server | 8 | 256 | 8.7834 | 799.2826 | 90999.2258 | 2.195850 | 363.9969 |
| prefill | server | 8 | 512 | 18.0202 | 1616.0811 | 89681.6406 | 4.505050 | 358.7266 |
| prefill | server | 8 | 1024 | 39.2847 | 3302.2228 | 84058.7506 | 9.821175 | 336.2350 |
| decode | server | 8 | 128 | 1.2669 | 3.1051 | 2450.9376 | 0.316725 | 9.8038 |
| decode | server | 8 | 256 | 1.2665 | 3.1222 | 2465.2174 | 0.316625 | 9.8609 |
| decode | server | 8 | 512 | 1.2792 | 3.1564 | 2467.4856 | 0.319800 | 9.8699 |
| decode | server | 8 | 1024 | 1.4329 | 3.2248 | 2250.5598 | 0.358225 | 9.0022 |

### qwen3-moe

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | server | 8 | 128 | 5.8273 | 459.8877 | 78919.5168 | 1.456825 | 315.6781 |
| prefill | server | 8 | 256 | 9.1725 | 928.5289 | 101229.6430 | 2.293125 | 404.9186 |
| prefill | server | 8 | 512 | 18.1956 | 1892.0890 | 103986.0736 | 4.548900 | 415.9443 |
| prefill | server | 8 | 1024 | 43.3490 | 3924.3000 | 90528.0399 | 10.837250 | 362.1122 |
| decode | server | 8 | 128 | 2.8501 | 3.5929 | 1260.6070 | 0.712525 | 5.0424 |
| decode | server | 8 | 256 | 2.8497 | 3.6271 | 1272.7887 | 0.712425 | 5.0912 |
| decode | server | 8 | 512 | 2.9695 | 3.6955 | 1244.4809 | 0.742375 | 4.9779 |
| decode | server | 8 | 1024 | 3.2884 | 3.8323 | 1165.4072 | 0.822100 | 4.6616 |

## Orin

Power assumption: `9 W`

### gemma-7b

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | edge | 1 | 128 | 13.6488 | 71.1634 | 5213.8979 | 0.122839 | 579.3220 |
| prefill | edge | 1 | 256 | 20.0298 | 142.8685 | 7132.7981 | 0.180268 | 792.5331 |
| prefill | edge | 1 | 512 | 37.8459 | 287.9033 | 7607.2520 | 0.340613 | 845.2502 |
| prefill | edge | 1 | 1024 | 85.3034 | 584.4720 | 6851.6847 | 0.767731 | 761.2983 |
| decode | edge | 1 | 128 | 8.8971 | 0.5560 | 62.4883 | 0.080074 | 6.9431 |
| decode | edge | 1 | 256 | 8.8775 | 0.5581 | 62.8645 | 0.079898 | 6.9849 |
| decode | edge | 1 | 512 | 8.6269 | 0.5623 | 65.1811 | 0.077642 | 7.2423 |
| decode | edge | 1 | 1024 | 9.1266 | 0.5708 | 62.5396 | 0.082139 | 6.9488 |

### llama3-8b

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | edge | 1 | 128 | 9.6691 | 56.1216 | 5804.2176 | 0.087022 | 644.9131 |
| prefill | edge | 1 | 256 | 15.8621 | 112.7905 | 7110.6896 | 0.142759 | 790.0766 |
| prefill | edge | 1 | 512 | 30.0806 | 227.7704 | 7572.0032 | 0.270725 | 841.3337 |
| prefill | edge | 1 | 1024 | 67.6672 | 464.2984 | 6861.4986 | 0.609005 | 762.3887 |
| decode | edge | 1 | 128 | 7.5979 | 0.4384 | 57.7067 | 0.068381 | 6.4119 |
| decode | edge | 1 | 256 | 8.1463 | 0.4406 | 54.0844 | 0.073317 | 6.0094 |
| decode | edge | 1 | 512 | 7.8826 | 0.4449 | 56.4362 | 0.070943 | 6.2707 |
| decode | edge | 1 | 1024 | 8.4329 | 0.4534 | 53.7676 | 0.075896 | 5.9742 |

### qwen3-8b

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | edge | 1 | 128 | 9.4144 | 49.6815 | 5277.1796 | 0.084730 | 586.3533 |
| prefill | edge | 1 | 256 | 14.1077 | 99.9103 | 7081.9715 | 0.126969 | 786.8857 |
| prefill | edge | 1 | 512 | 27.8230 | 202.0101 | 7260.5438 | 0.250407 | 806.7271 |
| prefill | edge | 1 | 1024 | 62.6777 | 412.7778 | 6585.7203 | 0.564099 | 731.7467 |
| decode | edge | 1 | 128 | 5.6620 | 0.3881 | 68.5511 | 0.050958 | 7.6168 |
| decode | edge | 1 | 256 | 6.3266 | 0.3903 | 61.6879 | 0.056939 | 6.8542 |
| decode | edge | 1 | 512 | 8.6119 | 0.3946 | 45.8146 | 0.077507 | 5.0905 |
| decode | edge | 1 | 1024 | 8.0521 | 0.4031 | 50.0619 | 0.072469 | 5.5624 |

### qwen3-moe

| Action | Config | Batch | Length | Latency (ms) | FLOPs (G) | Throughput (GFLOPS) | Estimated energy (J) | Efficiency (GFLOPS/J) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| prefill | edge | 1 | 128 | 60.9984 | 57.4857 | 942.4132 | 0.548986 | 104.7126 |
| prefill | edge | 1 | 256 | 66.4800 | 116.0661 | 1745.8801 | 0.598320 | 193.9867 |
| prefill | edge | 1 | 512 | 78.6149 | 236.5110 | 3008.4759 | 0.707534 | 334.2751 |
| prefill | edge | 1 | 1024 | 114.7558 | 490.5376 | 4274.6214 | 1.032802 | 474.9579 |
| decode | edge | 1 | 128 | 12.8750 | 0.4491 | 34.8821 | 0.115875 | 3.8758 |
| decode | edge | 1 | 256 | 11.7870 | 0.4534 | 38.4647 | 0.106083 | 4.2739 |
| decode | edge | 1 | 512 | 12.5027 | 0.4619 | 36.9470 | 0.112524 | 4.1052 |
| decode | edge | 1 | 1024 | 14.4840 | 0.4790 | 33.0739 | 0.130356 | 3.6749 |

## Formulas

```text
combined_latency = attention_mean_latency + ffn_mean_latency
throughput_GFLOPS = total_FLOPs / (combined_latency_ms * 1e6)
estimated_energy_J = assumed_power_W * combined_latency_ms / 1000
energy_efficiency_GFLOPS_per_J = throughput_GFLOPS / assumed_power_W
```
