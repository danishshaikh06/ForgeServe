ForgeServe Phase 5 Benchmark — Continuous Batching
Model     : Qwen/Qwen2.5-0.5B-Instruct
KV blocks : 512

Loading model...
[2026-09-16 15:06:41,234] forgeserve.model.runtime - INFO - Initializing Runtime: model=Qwen/Qwen2.5-0.5B-Instruct attention=sdpa
[2026-09-16 15:06:41,234] forgeserve.model.loader - INFO - ModelLoader configured: model=Qwen/Qwen2.5-0.5B-Instruct device=cuda dtype=torch.bfloat16 attention=sdpa
[2026-09-16 15:06:41,234] forgeserve.model.runtime - INFO - ModelLoader initialized successfully for Qwen/Qwen2.5-0.5B-Instruct.
[2026-09-16 15:06:41,234] forgeserve.model.loader - INFO - Loading tokenizer for Qwen/Qwen2.5-0.5B-Instruct
[2026-09-16 15:06:43,196] forgeserve.model.loader - INFO - Loading model with torch_dtype=torch.bfloat16 attn_implementation=sdpa
[2026-09-16 15:06:46,846] forgeserve.model.loader - INFO - Attention backend: SDPA (FlashAttention eligible). dtype=torch.bfloat16 device=cuda
[2026-09-16 15:06:46,846] forgeserve.model.paged_runtime - INFO - PagedRuntime initialized. Awaiting block manager attachment.
[2026-09-16 15:06:46,846] forgeserve.page_attention.block_manager - INFO - Pre-allocating 512 KV blocks (block_size=16). Total KV memory: 96.0 MB
[2026-09-16 15:06:46,963] forgeserve.page_attention.block_manager - INFO - BlockManager ready. 512 blocks available
[2026-09-16 15:06:46,963] forgeserve.model.paged_runtime - INFO - PagedRuntime initialized. Block pool: 512 blocks, block_size=16, memory=96.0 MB
Model loaded. Pool: 512 blocks × 16 tokens

─────────────────────────────────────────────────────────────────
SCENARIO 1 — Uniform request lengths (control)
Expected: small continuous batching advantage
─────────────────────────────────────────────────────────────────
[2026-09-16 15:06:46,971] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_00 max_tokens=80
[2026-09-16 15:06:52,298] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_00 tokens=80 time=5.327s blocks_used=9
[2026-09-16 15:06:52,298] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_01 max_tokens=80
[2026-09-16 15:06:56,687] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_01 tokens=80 time=4.387s blocks_used=9
[2026-09-16 15:06:56,687] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_02 max_tokens=80
[2026-09-16 15:07:01,019] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_02 tokens=80 time=4.332s blocks_used=9
[2026-09-16 15:07:01,021] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_03 max_tokens=80
[2026-09-16 15:07:05,235] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_03 tokens=80 time=4.214s blocks_used=9
[2026-09-16 15:07:05,235] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=None total_blocks=512 block_size=16
[2026-09-16 15:07:05,235] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=80 waiting=1
[2026-09-16 15:07:05,235] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_01 max_new_tokens=80 waiting=2
[2026-09-16 15:07:05,235] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_02 max_new_tokens=80 waiting=3
[2026-09-16 15:07:05,235] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_03 max_new_tokens=80 waiting=4
[2026-09-16 15:07:05,404] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=51 blocks_used=4 running=1 waiting=3 free_blocks=508
[2026-09-16 15:07:05,628] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_01 prompt_tokens=51 blocks_used=4 running=2 waiting=2 free_blocks=504
[2026-09-16 15:07:05,817] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_02 prompt_tokens=51 blocks_used=4 running=3 waiting=1 free_blocks=500
[2026-09-16 15:07:05,986] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_03 prompt_tokens=51 blocks_used=4 running=4 waiting=0 free_blocks=496
[2026-09-16 15:07:23,269] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=80 finish_reason=length released_blocks=9 running=3 waiting=0 free_blocks=485
[2026-09-16 15:07:23,269] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_01 generated=80 finish_reason=length released_blocks=9 running=2 waiting=0 free_blocks=494
[2026-09-16 15:07:23,269] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_02 generated=80 finish_reason=length released_blocks=9 running=1 waiting=0 free_blocks=503
[2026-09-16 15:07:23,269] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_03 generated=80 finish_reason=length released_blocks=9 running=0 waiting=0 free_blocks=512

=================================================================
  SEQUENTIAL
  Scenario : all
  Requests : 4
  Wall time: 18265 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 0.22
Throughput (tok/s)                                 17.5
-----------------------------------------------------------------
Mean latency (ms)                                4566.2
P50 latency (ms)                                 4360.6
P95 latency (ms)                                 5187.5
P99 latency (ms)                                 5300.4
-----------------------------------------------------------------
Mean queue wait (ms)                                0.0
Max queue wait (ms)                                 0.0
-----------------------------------------------------------------
Total tokens generated                               320
=================================================================


=================================================================
  CONTINUOUS BATCHING
  Scenario : all
  Requests : 4
  Wall time: 18039 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 0.22
Throughput (tok/s)                                 17.7
-----------------------------------------------------------------
Mean latency (ms)                               18037.1
P50 latency (ms)                                18037.1
P95 latency (ms)                                18037.2
P99 latency (ms)                                18037.2
-----------------------------------------------------------------
Mean queue wait (ms)                              473.1
Max queue wait (ms)                               749.7
-----------------------------------------------------------------
Total tokens generated                               320
Total decode steps                                    80
Avg batch size per step                            4.00
=================================================================


======================================================================
  COMPARISON: Sequential vs Continuous Batching
======================================================================
Metric                               Sequential      Cont. Batch
----------------------------------------------------------------------
Throughput (req/s)                        0.22            0.22
Throughput (tok/s)                        17.5            17.7
Mean latency (ms)                       4566.2         18037.1
P99 latency (ms)                        5300.4         18037.2
Mean queue wait (ms)                       0.0           473.1
----------------------------------------------------------------------
Token throughput gain                     1.01x
Request throughput gain                   0.22x
Mean latency change                       3.95x
======================================================================

  Interpretation:
  ℹ  Small throughput gain: 1.01x
     Expected with uniform-length requests or small batch.
  ℹ  Individual latency increased 3.95x — expected trade-off for higher throughput.


─────────────────────────────────────────────────────────────────
SCENARIO 2 — Mixed request lengths (real-world simulation)
Expected: moderate continuous batching advantage
─────────────────────────────────────────────────────────────────
[2026-09-16 15:07:23,269] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_00 max_tokens=10
[2026-09-16 15:07:23,790] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_00 tokens=9 time=0.521s blocks_used=3
[2026-09-16 15:07:23,800] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_01 max_tokens=30
[2026-09-16 15:07:24,943] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_01 tokens=21 time=1.144s blocks_used=4
[2026-09-16 15:07:24,943] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_02 max_tokens=80
[2026-09-16 15:07:29,485] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_02 tokens=80 time=4.544s blocks_used=8
[2026-09-16 15:07:29,485] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_03 max_tokens=120
[2026-09-16 15:07:36,296] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_03 tokens=120 time=6.812s blocks_used=10
[2026-09-16 15:07:36,296] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=None total_blocks=512 block_size=16
[2026-09-16 15:07:36,296] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=10 waiting=1
[2026-09-16 15:07:36,296] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_01 max_new_tokens=30 waiting=2
[2026-09-16 15:07:36,296] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_02 max_new_tokens=80 waiting=3
[2026-09-16 15:07:36,296] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_03 max_new_tokens=120 waiting=4
[2026-09-16 15:07:36,432] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=37 blocks_used=3 running=1 waiting=3 free_blocks=509
[2026-09-16 15:07:36,560] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_01 prompt_tokens=37 blocks_used=3 running=2 waiting=2 free_blocks=506
[2026-09-16 15:07:36,696] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_02 prompt_tokens=40 blocks_used=3 running=3 waiting=1 free_blocks=503
[2026-09-16 15:07:36,915] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_03 prompt_tokens=36 blocks_used=3 running=4 waiting=0 free_blocks=500
[2026-09-16 15:07:39,036] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=9 finish_reason=eos released_blocks=3 running=3 waiting=0 free_blocks=502
[2026-09-16 15:07:40,993] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_01 generated=21 finish_reason=eos released_blocks=4 running=2 waiting=0 free_blocks=504
[2026-09-16 15:07:47,052] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_02 generated=80 finish_reason=length released_blocks=8 running=1 waiting=0 free_blocks=504
[2026-09-16 15:07:49,134] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_03 generated=120 finish_reason=length released_blocks=10 running=0 waiting=0 free_blocks=512

=================================================================
  SEQUENTIAL
  Scenario : all
  Requests : 4
  Wall time: 13025 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 0.31
Throughput (tok/s)                                 17.7
-----------------------------------------------------------------
Mean latency (ms)                                3256.1
P50 latency (ms)                                 2844.8
P95 latency (ms)                                 6472.7
P99 latency (ms)                                 6744.9
-----------------------------------------------------------------
Mean queue wait (ms)                                0.0
Max queue wait (ms)                                 0.0
-----------------------------------------------------------------
Total tokens generated                               230
=================================================================


=================================================================
  CONTINUOUS BATCHING
  Scenario : all
  Requests : 4
  Wall time: 12836 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 0.31
Throughput (tok/s)                                 17.9
-----------------------------------------------------------------
Mean latency (ms)                                7752.7
P50 latency (ms)                                 7721.3
P95 latency (ms)                                12521.7
P99 latency (ms)                                12771.7
-----------------------------------------------------------------
Mean queue wait (ms)                              353.1
Max queue wait (ms)                               614.0
-----------------------------------------------------------------
Total tokens generated                               230
Total decode steps                                   120
Avg batch size per step                            1.92
=================================================================


======================================================================
  COMPARISON: Sequential vs Continuous Batching
======================================================================
Metric                               Sequential      Cont. Batch
----------------------------------------------------------------------
Throughput (req/s)                        0.31            0.31
Throughput (tok/s)                        17.7            17.9
Mean latency (ms)                       3256.1          7752.7
P99 latency (ms)                        6744.9         12771.7
Mean queue wait (ms)                       0.0           353.1
----------------------------------------------------------------------
Token throughput gain                     1.01x
Request throughput gain                   0.31x
Mean latency change                       2.38x
======================================================================

  Interpretation:
  ℹ  Small throughput gain: 1.01x
     Expected with uniform-length requests or small batch.
  ℹ  Individual latency increased 2.38x — expected trade-off for higher throughput.

  Per-request detail — Sequential:

  ID            Generated   Reason   Wait(ms)  Total(ms)    tok/s
  --------------------------------------------------------------
  req_00                9      eos        0.0      522.1     17.2
  req_01               21      eos        0.0     1144.8     18.3
  req_02               80   length        0.0     4544.7     17.6
  req_03              120   length        0.0     6812.9     17.6

  Per-request detail — Continuous Batching:

  ID            Generated   Reason   Wait(ms)  Total(ms)    tok/s
  --------------------------------------------------------------
  req_00                9      eos      133.1     2734.0      3.3
  req_01               21      eos      265.4     4692.1      4.5
  req_02               80   length      399.9    10750.5      7.4
  req_03              120   length      614.0    12834.2      9.3


─────────────────────────────────────────────────────────────────
SCENARIO 3 — Throughput scaling with concurrent requests
Expected: throughput grows as concurrency increases
─────────────────────────────────────────────────────────────────

   Concurrency   Throughput (tok/s)    Mean latency (ms)
  -------------------------------------------------------
[2026-09-16 15:07:49,134] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=None total_blocks=512 block_size=16
[2026-09-16 15:07:49,144] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=10 waiting=1
[2026-09-16 15:07:49,307] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=37 blocks_used=3 running=1 waiting=0 free_blocks=509
[2026-09-16 15:07:49,762] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=9 finish_reason=eos released_blocks=3 running=0 waiting=0 free_blocks=512
             1                 14.3                626.8
[2026-09-16 15:07:49,762] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=None total_blocks=512 block_size=16
[2026-09-16 15:07:49,772] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=10 waiting=1
[2026-09-16 15:07:49,772] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_01 max_new_tokens=30 waiting=2
[2026-09-16 15:07:49,901] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=37 blocks_used=3 running=1 waiting=1 free_blocks=509
[2026-09-16 15:07:50,036] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_01 prompt_tokens=37 blocks_used=3 running=2 waiting=0 free_blocks=506
[2026-09-16 15:07:50,927] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=9 finish_reason=eos released_blocks=3 running=1 waiting=0 free_blocks=509
[2026-09-16 15:07:51,518] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_01 generated=21 finish_reason=eos released_blocks=4 running=0 waiting=0 free_blocks=512
             2                 17.1               1455.5
[2026-09-16 15:07:51,518] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=None total_blocks=512 block_size=16
[2026-09-16 15:07:51,518] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=10 waiting=1
[2026-09-16 15:07:51,518] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_01 max_new_tokens=30 waiting=2
[2026-09-16 15:07:51,518] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_02 max_new_tokens=80 waiting=3
[2026-09-16 15:07:51,528] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_03 max_new_tokens=120 waiting=4
[2026-09-16 15:07:51,666] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=37 blocks_used=3 running=1 waiting=3 free_blocks=509
[2026-09-16 15:07:51,820] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_01 prompt_tokens=37 blocks_used=3 running=2 waiting=2 free_blocks=506
[2026-09-16 15:07:51,978] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_02 prompt_tokens=40 blocks_used=3 running=3 waiting=1 free_blocks=503
[2026-09-16 15:07:52,112] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_03 prompt_tokens=36 blocks_used=3 running=4 waiting=0 free_blocks=500
[2026-09-16 15:07:53,886] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=9 finish_reason=eos released_blocks=3 running=3 waiting=0 free_blocks=502
[2026-09-16 15:07:56,056] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_01 generated=21 finish_reason=eos released_blocks=4 running=2 waiting=0 free_blocks=504
[2026-09-16 15:08:03,581] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_02 generated=80 finish_reason=length released_blocks=8 running=1 waiting=0 free_blocks=504
[2026-09-16 15:08:06,862] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_03 generated=120 finish_reason=length released_blocks=10 running=0 waiting=0 free_blocks=512
             4                 15.0               8572.4

─────────────────────────────────────────────────────────────────
SCENARIO 4 — Queue pressure (12 requests, batch cap=4)
Expected: largest continuous batching advantage
Freed slots immediately admit waiting requests
─────────────────────────────────────────────────────────────────
[2026-09-16 15:08:06,864] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_00 max_tokens=10
[2026-09-16 15:08:07,765] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_00 tokens=8 time=0.898s blocks_used=3
[2026-09-16 15:08:07,766] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_01 max_tokens=20
[2026-09-16 15:08:09,972] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_01 tokens=20 time=2.207s blocks_used=4
[2026-09-16 15:08:09,972] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_02 max_tokens=40
[2026-09-16 15:08:13,852] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_02 tokens=40 time=3.883s blocks_used=5
[2026-09-16 15:08:13,852] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_03 max_tokens=10
[2026-09-16 15:08:14,485] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_03 tokens=8 time=0.626s blocks_used=3
[2026-09-16 15:08:14,485] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_04 max_tokens=80
[2026-09-16 15:08:20,426] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_04 tokens=80 time=5.947s blocks_used=8
[2026-09-16 15:08:20,426] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_05 max_tokens=30
[2026-09-16 15:08:23,062] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_05 tokens=30 time=2.634s blocks_used=5
[2026-09-16 15:08:23,062] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_06 max_tokens=50
[2026-09-16 15:08:26,381] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_06 tokens=50 time=3.315s blocks_used=6
[2026-09-16 15:08:26,381] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_07 max_tokens=15
[2026-09-16 15:08:27,708] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_07 tokens=15 time=1.320s blocks_used=4
[2026-09-16 15:08:27,709] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_08 max_tokens=20
[2026-09-16 15:08:29,129] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_08 tokens=20 time=1.428s blocks_used=4
[2026-09-16 15:08:29,139] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_09 max_tokens=60
[2026-09-16 15:08:32,984] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_09 tokens=60 time=3.848s blocks_used=6
[2026-09-16 15:08:32,984] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_10 max_tokens=10
[2026-09-16 15:08:33,757] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_10 tokens=10 time=0.774s blocks_used=4
[2026-09-16 15:08:33,757] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_11 max_tokens=45
[2026-09-16 15:08:36,466] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_11 tokens=45 time=2.710s blocks_used=5
[2026-09-16 15:08:36,466] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=4 total_blocks=512 block_size=16
[2026-09-16 15:08:36,476] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=10 waiting=1
[2026-09-16 15:08:36,476] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_01 max_new_tokens=20 waiting=2
[2026-09-16 15:08:36,476] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_02 max_new_tokens=40 waiting=3
[2026-09-16 15:08:36,476] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_03 max_new_tokens=10 waiting=4
[2026-09-16 15:08:36,476] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_04 max_new_tokens=80 waiting=5
[2026-09-16 15:08:36,476] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_05 max_new_tokens=30 waiting=6
[2026-09-16 15:08:36,476] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_06 max_new_tokens=50 waiting=7
[2026-09-16 15:08:36,476] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_07 max_new_tokens=15 waiting=8
[2026-09-16 15:08:36,476] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_08 max_new_tokens=20 waiting=9
[2026-09-16 15:08:36,476] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_09 max_new_tokens=60 waiting=10
[2026-09-16 15:08:36,476] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_10 max_new_tokens=10 waiting=11
[2026-09-16 15:08:36,481] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_11 max_new_tokens=45 waiting=12
[2026-09-16 15:08:36,670] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=36 blocks_used=3 running=1 waiting=11 free_blocks=509
[2026-09-16 15:08:36,838] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_01 prompt_tokens=34 blocks_used=3 running=2 waiting=10 free_blocks=506
[2026-09-16 15:08:37,009] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_02 prompt_tokens=35 blocks_used=3 running=3 waiting=9 free_blocks=503
[2026-09-16 15:08:37,189] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_03 prompt_tokens=36 blocks_used=3 running=4 waiting=8 free_blocks=500
[2026-09-16 15:08:39,882] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=8 finish_reason=eos released_blocks=3 running=3 waiting=8 free_blocks=503
[2026-09-16 15:08:39,882] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_03 generated=8 finish_reason=eos released_blocks=3 running=2 waiting=8 free_blocks=506
[2026-09-16 15:08:40,058] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_04 prompt_tokens=34 blocks_used=3 running=3 waiting=7 free_blocks=503
[2026-09-16 15:08:40,235] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_05 prompt_tokens=36 blocks_used=3 running=4 waiting=6 free_blocks=500
[2026-09-16 15:08:43,431] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_01 generated=20 finish_reason=length released_blocks=4 running=3 waiting=6 free_blocks=502
[2026-09-16 15:08:43,668] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_06 prompt_tokens=34 blocks_used=3 running=4 waiting=5 free_blocks=499
[2026-09-16 15:08:48,776] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_05 generated=30 finish_reason=length released_blocks=5 running=3 waiting=5 free_blocks=499
[2026-09-16 15:08:48,960] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_07 prompt_tokens=36 blocks_used=3 running=4 waiting=4 free_blocks=496
[2026-09-16 15:08:49,574] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_02 generated=40 finish_reason=length released_blocks=5 running=3 waiting=4 free_blocks=500
[2026-09-16 15:08:49,820] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_08 prompt_tokens=34 blocks_used=3 running=4 waiting=3 free_blocks=497
[2026-09-16 15:08:52,910] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_07 generated=15 finish_reason=length released_blocks=4 running=3 waiting=3 free_blocks=499
[2026-09-16 15:08:53,045] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_09 prompt_tokens=34 blocks_used=3 running=4 waiting=2 free_blocks=496
[2026-09-16 15:08:54,843] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_08 generated=20 finish_reason=length released_blocks=4 running=3 waiting=2 free_blocks=498
[2026-09-16 15:08:55,029] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_10 prompt_tokens=40 blocks_used=3 running=4 waiting=1 free_blocks=495
[2026-09-16 15:08:57,539] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_06 generated=50 finish_reason=length released_blocks=6 running=3 waiting=1 free_blocks=498
[2026-09-16 15:08:57,549] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_10 generated=10 finish_reason=length released_blocks=4 running=2 waiting=1 free_blocks=502
[2026-09-16 15:08:57,714] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_11 prompt_tokens=35 blocks_used=3 running=3 waiting=0 free_blocks=499
[2026-09-16 15:09:01,439] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_04 generated=80 finish_reason=length released_blocks=8 running=2 waiting=0 free_blocks=503
[2026-09-16 15:09:04,868] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_09 generated=60 finish_reason=length released_blocks=6 running=1 waiting=0 free_blocks=507
[2026-09-16 15:09:05,016] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_11 generated=45 finish_reason=length released_blocks=5 running=0 waiting=0 free_blocks=512

=================================================================
  SEQUENTIAL
  Scenario : all
  Requests : 12
  Wall time: 29610 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 0.41
Throughput (tok/s)                                 13.0
-----------------------------------------------------------------
Mean latency (ms)                                2467.5
P50 latency (ms)                                 2422.4
P95 latency (ms)                                 4813.3
P99 latency (ms)                                 5720.9
-----------------------------------------------------------------
Mean queue wait (ms)                                0.0
Max queue wait (ms)                                 0.0
-----------------------------------------------------------------
Total tokens generated                               386
=================================================================


=================================================================
  CONTINUOUS BATCHING
  Scenario : all
  Requests : 12
  Wall time: 28544 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 0.42
Throughput (tok/s)                                 13.5
-----------------------------------------------------------------
Mean latency (ms)                               16500.2
P50 latency (ms)                                17399.3
P95 latency (ms)                                28460.4
P99 latency (ms)                                28522.4
-----------------------------------------------------------------
Mean queue wait (ms)                             8212.1
Max queue wait (ms)                             21241.0
-----------------------------------------------------------------
Total tokens generated                               386
Total decode steps                                   115
Avg batch size per step                            3.36
=================================================================


======================================================================
  COMPARISON: Sequential vs Continuous Batching
======================================================================
Metric                               Sequential      Cont. Batch
----------------------------------------------------------------------
Throughput (req/s)                        0.41            0.42
Throughput (tok/s)                        13.0            13.5
Mean latency (ms)                       2467.5         16500.2
P99 latency (ms)                        5720.9         28522.4
Mean queue wait (ms)                       0.0          8212.1
----------------------------------------------------------------------
Token throughput gain                     1.04x
Request throughput gain                   0.42x
Mean latency change                       6.69x
======================================================================

  Interpretation:
  ℹ  Small throughput gain: 1.04x
     Expected with uniform-length requests or small batch.
  ℹ  Individual latency increased 6.69x — expected trade-off for higher throughput.

  Per-request — Sequential:

  ID            Generated   Reason   Wait(ms)  Total(ms)    tok/s
  --------------------------------------------------------------
  req_00                8      eos        0.0      900.0      8.9
  req_01               20   length        0.0     2209.1      9.1
  req_02               40   length        0.0     3885.2     10.3
  req_03                8      eos        0.0      627.6     12.7
  req_04               80   length        0.0     5947.8     13.5
  req_05               30   length        0.0     2635.7     11.4
  req_06               50   length        0.0     3316.4     15.1
  req_07               15   length        0.0     1321.4     11.4
  req_08               20   length        0.0     1429.9     14.0
  req_09               60   length        0.0     3849.1     15.6
  req_10               10   length        0.0      775.7     12.9
  req_11               45   length        0.0     2712.0     16.6

  Per-request — Continuous Batching:

  ID            Generated   Reason   Wait(ms)  Total(ms)    tok/s
  --------------------------------------------------------------
  req_00                8      eos      194.6     3407.4      2.3
  req_01               20   length      365.4     6953.6      2.9
  req_02               40   length      539.1    13102.3      3.1
  req_03                8      eos      711.8     3406.6      2.3
  req_04               80   length     3583.6    24960.9      3.2
  req_05               30   length     3758.2    12305.4      2.4
  req_06               50   length     7192.6    21065.3      2.4
  req_07               15   length    12489.0    16433.7      0.9
  req_08               20   length    13343.4    18364.8      1.1
  req_09               60   length    16568.7    28396.9      2.1
  req_10               10   length    18557.6    21068.0      0.5
  req_11               45   length    21241.0    28537.9      1.6