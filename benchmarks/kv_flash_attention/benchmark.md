Phase 3 Summary — FlashAttention via SDPA
Model: Qwen2.5-0.5B-Instruct
GPU:   RTX 4070
----------------------------------------------------------------------------------------------------------
ForgeServe Phase 3 Benchmark — EAGER vs SDPA
Model: Qwen/Qwen2.5-0.5B-Instruct
Warmup: 2 | Benchmark: 5
Scenarios designed to stress FlashAttention

=================================================================
  BENCHMARK RESULTS
  Prompt tokens : 37
  Generated     : 100 (EAGER) / 100 (SDPA)
  Runs          : 5
=================================================================
Metric                                EAGER           SDPA
-----------------------------------------------------------------
TTFT (ms)                             37.3          36.8
TPOT (ms)                             31.7          31.9  ±0.3
Total time (ms)                     3172.8        3191.7
Throughput (tok/s)                    31.5          31.3
Peak memory (MB)                     980.7         981.8
-----------------------------------------------------------------
Total speedup                         0.99x
TPOT speedup                          0.99x
Memory delta (MB)                     +1.2
=================================================================

  ℹ  Small benefit (0.99x) — sequence may still be short
  ℹ  Memory delta negligible — sequence too short for visible saving

>>> Scenario: medium_prompt_400tok_200gen
=================================================================
  BENCHMARK RESULTS
  Prompt tokens : 196
  Generated     : 200 (EAGER) / 200 (SDPA)
  Runs          : 5
=================================================================
Metric                                EAGER           SDPA
-----------------------------------------------------------------
TTFT (ms)                             43.5          49.0
TPOT (ms)                             36.4          33.0  ±0.3
Total time (ms)                     7296.6        6625.1
Throughput (tok/s)                    27.4          30.2
Peak memory (MB)                    1077.0        1077.0
-----------------------------------------------------------------
Total speedup                         1.10x
TPOT speedup                          1.10x
Memory delta (MB)                     +0.0
=================================================================

  ✅ Moderate FlashAttention benefit: 1.10x TPOT speedup
  ℹ  Memory delta negligible — sequence too short for visible saving

>>> Scenario: long_prompt_800tok_50gen 
================================================================
  BENCHMARK RESULTS
  Prompt tokens : 364
  Generated     : 50 (EAGER) / 50 (SDPA)
  Runs          : 5
=================================================================
Metric                                EAGER           SDPA
-----------------------------------------------------------------
TTFT (ms)                             46.8          44.0
TPOT (ms)                             34.1          33.0  ±0.6
Total time (ms)                     1715.4        1660.5
Throughput (tok/s)                    29.2          30.1
Peak memory (MB)                    1179.6        1179.6
-----------------------------------------------------------------
Total speedup                         1.03x
TPOT speedup                          1.03x
Memory delta (MB)                     +0.0
=================================================================

  ℹ  Small benefit (1.03x) — sequence may still be short
  ℹ  Memory delta negligible — sequence too short for visible saving

>>> Scenario: long_prompt_800tok_300gen   
================================================================
  BENCHMARK RESULTS
  Prompt tokens : 364
  Generated     : 300 (EAGER) / 300 (SDPA)
  Runs          : 5
=================================================================
Metric                                EAGER           SDPA
-----------------------------------------------------------------
TTFT (ms)                             51.8          45.4
TPOT (ms)                             36.5          33.0  ±0.5
Total time (ms)                    10958.7        9926.1
Throughput (tok/s)                    27.4          30.2
Peak memory (MB)                    1179.6        1179.6
-----------------------------------------------------------------
Total speedup                         1.10x
TPOT speedup                          1.10x
Memory delta (MB)                     +0.0
=================================================================

  ✅ Moderate FlashAttention benefit: 1.10x TPOT speedup
  ℹ  Memory delta negligible — sequence too short for visible saving