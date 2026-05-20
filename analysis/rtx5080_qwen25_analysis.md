# RTX 5080 Qwen2.5 KV-cache Benchmark Analysis

## Scope

- GPU: NVIDIA GeForce RTX 5080
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- dtype: fp16
- sequence lengths: 512, 1024, 2048, 4096, 8192
- batch sizes: 1, 2, 4, 8
- cache modes: `dynamic`, `quantized`, `offloaded`, `no_cache`
- max new tokens: 64

## Main findings

1. **Pipeline and sweep completed.** The sweep produced 80 rows: 76 successful rows and 4 OOM rows.
2. **OOM boundary:** all four cache modes hit OOM at `batch_size=8`, `seq_len=8192`.
3. **KV formula validation:** every successful row had `kv_actual_over_theory = 1.0`, so measured `past_key_values` tensor footprint matched the theoretical KV-cache formula.
4. **Throughput trend:** `dynamic` was the highest-throughput baseline on average. Across the 19 common non-OOM cases, mean throughput relative to dynamic was: quantized 0.744×, offloaded 0.594×, no_cache 0.094×.
5. **Memory trade-off:** across common non-OOM cases, mean peak allocated delta relative to dynamic was: quantized 0.786×, offloaded 0.705×, no_cache 0.739×.
6. **No-cache is not a performance strategy.** It reduced peak deltas but collapsed decode throughput at long contexts; for `batch_size=4`, `seq_len=8192`, no_cache reached only 2.95 tokens/s versus dynamic 124.96 tokens/s.
7. **Offload is a capacity/memory strategy with throughput cost.** At `batch_size=4`, `seq_len=8192`, offloaded reduced peak delta from 3.12 GB to 2.22 GB but throughput fell from 124.96 to 45.45 tokens/s.
8. **Quantized cache reduced peak memory with smaller throughput loss than offload in long-context examples.** At `batch_size=4`, `seq_len=8192`, quantized reduced peak delta from 3.12 GB to 2.45 GB and reached 108.24 tokens/s.

## Important caveat

The quantized run sometimes generated fewer than `max_new_tokens` because generation can stop early on EOS. Therefore `tokens_per_sec` is the safer normalized throughput metric, while raw `latency_ms` is not always an apples-to-apples cache-mode comparison for quantized rows. A stricter follow-up can disable EOS or force exactly fixed decode length for all modes.

## Status count

```text
        count
status       
ok         76
oom         4
```

## OOM cases

```text
cache_mode  batch_size  seq_len status
   dynamic           8     8192    oom
 offloaded           8     8192    oom
  no_cache           8     8192    oom
 quantized           8     8192    oom
```

## KV theory check

```text
            count  mean  min  max
cache_mode                       
dynamic        19   1.0  1.0  1.0
no_cache       19   1.0  1.0  1.0
offloaded      19   1.0  1.0  1.0
quantized      19   1.0  1.0  1.0
```

## Throughput summary: tokens/sec

```text
                             mean         min         max
cache_mode batch_size                                    
dynamic    1           102.954588   74.551975  118.278553
           2           172.011250   99.379165  221.578650
           4           279.011810  124.957829  408.561142
           8           508.949045  268.722474  740.651525
no_cache   1            17.787288    2.960828   41.596855
           2            20.039026    2.930566   50.028289
           4            21.052358    2.951665   53.765102
           8            26.042407    6.449184   56.464489
offloaded  1            73.897790   36.712032   99.532593
           2           115.953085   41.750073  188.018660
           4           164.930572   45.451941  307.658002
           8           261.764054   99.165609  459.756918
quantized  1            71.138315   59.873431   79.974596
           2           125.327550   87.020473  147.428885
           4           211.979120  108.237820  268.107895
           8           349.807877  230.620982  402.819044
```

## Latency summary: ms

```text
                               mean          min           max
cache_mode batch_size                                         
dynamic    1             640.097803   541.095561    858.461491
           2             810.260738   577.672985   1287.996326
           4            1101.231246   626.589202   2048.691160
           8            1162.733421   691.283259   1905.311425
no_cache   1            8064.956741  1538.577859  21615.576961
           2           15962.827941  2558.552411  43677.565360
           4           31821.988189  4761.452872  86730.706343
           8           36254.311950  9067.646104  79389.887200
offloaded  1             992.261003   643.005455   1743.297664
           2            1467.587838   680.783493   3065.862919
           4            2423.829177   832.092774   5632.322672
           8            2692.747169  1113.631965   5163.080273
quantized  1             605.305042   233.813546   1068.921534
           2             756.510418   217.053802   1470.918234
           4            1146.331055   246.935051   2365.162187
           8            1155.805287   323.168048   2220.092878
```

## Peak CUDA allocated delta summary: bytes

```text
                               mean        min         max
cache_mode batch_size                                     
dynamic    1           3.053708e+08   59394560   780405248
           2           6.056535e+08   98601472  1560807936
           4           1.210256e+09  198248960  3121613312
           8           1.463258e+09  390203904  3121613312
no_cache   1           2.184055e+08   39796736   554037760
           2           4.360484e+08   81369600  1103500800
           4           8.683245e+08  160704000  2204519936
           8           1.067269e+09  317761024  2226460160
offloaded  1           2.156914e+08   35670528   553912832
           2           4.301219e+08   70289920  1107823104
           4           8.591926e+08  141625856  2215643648
           8           1.038584e+09  276957696  2215643648
quantized  1           2.399528e+08   48843264   611584512
           2           4.748174e+08   77498880  1223166464
           4           9.485837e+08  156043776  2446330368
           8           1.146719e+09  305793536  2446330368
```

## Ratios vs dynamic on common non-OOM cases

```text
            throughput_vs_dynamic_mean  throughput_vs_dynamic_min  throughput_vs_dynamic_max  latency_vs_dynamic_mean  peak_delta_vs_dynamic_mean
cache_mode                                                                                                                                       
quantized                     0.744150                   0.534771                   0.875641                 0.904539                    0.786496
offloaded                     0.594451                   0.363738                   0.848541                 1.807345                    0.705081
no_cache                      0.093942                   0.023621                   0.351686                19.019696                    0.738751
```

## Key long-context examples

```text
cache_mode  batch_size  seq_len  tokens_per_sec   latency_ms  peak_delta_bytes  generated_tokens_total  kv_actual_over_theory
   dynamic           4     8192      124.957829  2048.691160        3121613312                     256                    1.0
  no_cache           4     8192        2.951665 86730.706343        2204519936                     256                    1.0
 offloaded           4     8192       45.451941  5632.322672        2215643648                     256                    1.0
 quantized           4     8192      108.237820  2365.162187        2446330368                     256                    1.0
   dynamic           8     4096      268.722474  1905.311425        3121613312                     512                    1.0
  no_cache           8     4096        6.449184 79389.887200        2226460160                     512                    1.0
 offloaded           8     4096       99.165609  5163.080273        2215643648                     512                    1.0
 quantized           8     4096      230.620982  2220.092878        2446330368                     512                    1.0
```

## Report-safe conclusion

The RTX 5080 sweep supports the claim that the benchmark measures KV-cache memory pressure and decode degradation on a consumer GPU. The measured KV-cache tensor footprint matched the theoretical formula exactly across all successful rows. Throughput generally decreased as sequence length increased, and the largest tested setting (`batch_size=8`, `seq_len=8192`) reached the capacity boundary for every cache mode. Dynamic cache provided the strongest throughput baseline, while quantized and offloaded cache reduced peak memory deltas at the cost of throughput. No-cache avoided cache reuse and became dramatically slower at long context lengths.
