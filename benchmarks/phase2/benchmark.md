ForgeServe Phase 2 Benchmark
Model: Qwen/Qwen2.5-0.5B-Instruct
Warmup runs: 2 | Benchmark runs: 5
[2026-08-06 22:05:49,235] forgeserve.model.runtime - INFO - Initializing Runtime with model: Qwen/Qwen2.5-0.5B-Instruct
[2026-08-06 22:05:49,235] forgeserve.model.runtime - INFO - ModelLoader initialized successfully for Qwen/Qwen2.5-0.5B-Instruct.

>>> Scenario: prompt=4 tokens, generate=50 tokens
[2026-08-06 22:05:54,232] runner - INFO - Warming up with 2 runs before benchmarking
[2026-08-06 22:05:58,029] runner - INFO - Warmup complete. Starting 5 benchmark runs.

=================================================================
  BENCHMARK RESULTS
  Prompt tokens : 33
  Generated     : 21 (naive) / 21 (kv_cache)
  Runs          : 5
=================================================================
Metric                                Naive       KV Cache
-----------------------------------------------------------------
TTFT (ms)                             37.8          42.6
TPOT (ms)                             40.3          34.5  ±2.2
Total time (ms)                      843.7         733.0
Throughput (tok/s)                    25.0          28.7
Peak memory (MB)                     990.7         978.3
-----------------------------------------------------------------
Total speedup                         1.15x
TPOT speedup                          1.17x
Memory overhead (MB)                 -12.4
=================================================================


>>> Scenario: prompt=4 tokens, generate=200 tokens
[2026-08-06 22:06:06,330] runner - INFO - Warming up with 2 runs before benchmarking
[2026-08-06 22:06:09,768] runner - INFO - Warmup complete. Starting 5 benchmark runs.

=================================================================
  BENCHMARK RESULTS
  Prompt tokens : 33
  Generated     : 21 (naive) / 21 (kv_cache)
  Runs          : 5
=================================================================
Metric                                Naive       KV Cache
-----------------------------------------------------------------
TTFT (ms)                             39.1          44.9
TPOT (ms)                             41.3          39.2  ±4.6
Total time (ms)                      864.7         828.6
Throughput (tok/s)                    24.4          25.6
Peak memory (MB)                     990.7         978.3
-----------------------------------------------------------------
Total speedup                         1.04x
TPOT speedup                          1.05x
Memory overhead (MB)                 -12.4
=================================================================


>>> Scenario: prompt=17 tokens, generate=50 tokens
[2026-08-06 22:06:18,676] runner - INFO - Warming up with 2 runs before benchmarking
[2026-08-06 22:06:22,314] runner - INFO - Warmup complete. Starting 5 benchmark runs.

=================================================================
  BENCHMARK RESULTS
  Prompt tokens : 47
  Generated     : 22 (naive) / 22 (kv_cache)
  Runs          : 5
=================================================================
Metric                                Naive       KV Cache
-----------------------------------------------------------------
TTFT (ms)                             38.7          38.6
TPOT (ms)                             37.9          32.9  ±1.2
Total time (ms)                      835.7         729.9
Throughput (tok/s)                    26.3          30.2
Peak memory (MB)                    1000.0         987.1
-----------------------------------------------------------------
Total speedup                         1.14x
TPOT speedup                          1.15x
Memory overhead (MB)                 -12.9
=================================================================


>>> Scenario: prompt=17 tokens, generate=200 tokens
[2026-08-06 22:06:30,548] runner - INFO - Warming up with 2 runs before benchmarking
[2026-08-06 22:06:33,841] runner - INFO - Warmup complete. Starting 5 benchmark runs.

=================================================================
  BENCHMARK RESULTS
  Prompt tokens : 47
  Generated     : 22 (naive) / 22 (kv_cache)
  Runs          : 5
=================================================================
Metric                                Naive       KV Cache
-----------------------------------------------------------------
TTFT (ms)                             38.4          38.7
TPOT (ms)                             38.0          33.4  ±1.3
Total time (ms)                      837.3         739.3
Throughput (tok/s)                    26.3          29.8
Peak memory (MB)                    1000.0         987.1
-----------------------------------------------------------------
Total speedup                         1.13x
TPOT speedup                          1.14x
Memory overhead (MB)                 -12.9
=================================================================#