ForgeServe Phase 5B — Batched Forward Pass Benchmark
Model     : Qwen/Qwen2.5-0.5B-Instruct
KV blocks : 512

Loading model...
[2026-09-27 16:26:26,625] forgeserve.model.runtime - INFO - Initializing Runtime: model=Qwen/Qwen2.5-0.5B-Instruct attention=sdpa
[2026-09-27 16:26:26,625] forgeserve.model.loader - INFO - ModelLoader configured: model=Qwen/Qwen2.5-0.5B-Instruct device=cuda dtype=torch.bfloat16 attention=sdpa
[2026-09-27 16:26:26,625] forgeserve.model.runtime - INFO - ModelLoader initialized successfully for Qwen/Qwen2.5-0.5B-Instruct.
[2026-09-27 16:26:26,625] forgeserve.model.loader - INFO - Loading tokenizer for Qwen/Qwen2.5-0.5B-Instruct
[2026-09-27 16:26:30,735] forgeserve.model.loader - INFO - Loading model with torch_dtype=torch.bfloat16 attn_implementation=sdpa
[2026-09-27 16:26:40,498] forgeserve.model.loader - INFO - Attention backend: SDPA (FlashAttention eligible). dtype=torch.bfloat16 device=cuda
[2026-09-27 16:26:40,498] forgeserve.model.paged_runtime - INFO - PagedRuntime initialized. Awaiting block manager attachment.
[2026-09-27 16:26:40,498] forgeserve.page_attention.block_manager - INFO - Pre-allocating 512 KV blocks (block_size=16). Total KV memory: 96.0 MB
[2026-09-27 16:26:40,564] forgeserve.page_attention.block_manager - INFO - BlockManager ready. 512 blocks available
[2026-09-27 16:26:40,564] forgeserve.model.paged_runtime - INFO - PagedRuntime initialized. Block pool: 512 blocks, block_size=16, memory=96.0 MB
Ready. Pool: 512 blocks × 16 tokens = 8,192 cached tokens max

─────────────────────────────────────────────────────────────────
SCENARIO 1 — Uniform lengths, N=4
Control: all same length, no early EOS
Batching should show clearest gain here
─────────────────────────────────────────────────────────────────
[2026-09-27 16:26:40,568] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_00 max_tokens=80
[2026-09-27 16:26:44,704] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_00 tokens=80 time=4.142s blocks_used=9
[2026-09-27 16:26:44,704] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_01 max_tokens=80
[2026-09-27 16:26:48,728] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_01 tokens=80 time=4.018s blocks_used=9
[2026-09-27 16:26:48,731] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_02 max_tokens=80
[2026-09-27 16:26:52,496] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_02 tokens=80 time=3.771s blocks_used=9
[2026-09-27 16:26:52,496] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_03 max_tokens=80
[2026-09-27 16:26:56,000] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_03 tokens=80 time=3.498s blocks_used=9
[2026-09-27 16:26:56,000] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=None total_blocks=512 block_size=16
[2026-09-27 16:26:56,000] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=80 waiting=1
[2026-09-27 16:26:56,000] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_01 max_new_tokens=80 waiting=2
[2026-09-27 16:26:56,000] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_02 max_new_tokens=80 waiting=3
[2026-09-27 16:26:56,000] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_03 max_new_tokens=80 waiting=4
[2026-09-27 16:26:56,135] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=53 blocks_used=4 running=1 waiting=3 free_blocks=508
[2026-09-27 16:26:56,259] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_01 prompt_tokens=53 blocks_used=4 running=2 waiting=2 free_blocks=504
[2026-09-27 16:26:56,377] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_02 prompt_tokens=53 blocks_used=4 running=3 waiting=1 free_blocks=500
[2026-09-27 16:26:56,507] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_03 prompt_tokens=53 blocks_used=4 running=4 waiting=0 free_blocks=496
[2026-09-27 16:27:00,307] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=80 finish_reason=length released_blocks=9 running=3 waiting=0 free_blocks=485
[2026-09-27 16:27:00,307] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_01 generated=80 finish_reason=length released_blocks=9 running=2 waiting=0 free_blocks=494
[2026-09-27 16:27:00,307] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_02 generated=80 finish_reason=length released_blocks=9 running=1 waiting=0 free_blocks=503
[2026-09-27 16:27:00,307] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_03 generated=80 finish_reason=length released_blocks=9 running=0 waiting=0 free_blocks=512

=================================================================
  SEQUENTIAL
  Requests : 4  |  Wall time: 15433 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 0.26
Throughput (tok/s)                                 20.7
-----------------------------------------------------------------
Mean latency (ms)                                3858.2
P50  latency (ms)                                3895.6
P95  latency (ms)                                4124.5
P99  latency (ms)                                4139.4
-----------------------------------------------------------------
Mean queue wait (ms)                                0.0
Max  queue wait (ms)                                0.0
-----------------------------------------------------------------
Total tokens generated                               320
=================================================================

=================================================================
  BATCHED FORWARD
  Requests : 4  |  Wall time: 4307 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 0.93
Throughput (tok/s)                                 74.3
-----------------------------------------------------------------
Mean latency (ms)                                4305.3
P50  latency (ms)                                4305.3
P95  latency (ms)                                4305.8
P99  latency (ms)                                4305.9
-----------------------------------------------------------------
Mean queue wait (ms)                              319.2
Max  queue wait (ms)                              504.2
-----------------------------------------------------------------
Total tokens generated                               320
Total decode steps                                    80
Avg batch size / step                              4.00
=================================================================

======================================================================
  COMPARISON  Sequential  →  Batched Forward
======================================================================
Metric                               Sequential          Batched
----------------------------------------------------------------------
Wall time (ms)                           15433            4307
Throughput (tok/s)                        20.7            74.3
Throughput (req/s)                        0.26            0.93
Mean latency (ms)                       3858.2          4305.3
P99 latency (ms)                        4139.4          4305.9
Mean wait (ms)                             0.0           319.2
Avg batch / step                            N/A            4.00
----------------------------------------------------------------------
Wall time speedup                         3.58x
Token throughput gain                     3.58x
Request throughput gain                   3.58x
Mean latency change                       1.12x
======================================================================

  Interpretation:
  ✅ Strong batching gain: 3.58x — GPU is bandwidth-bound
     Weight loading cost shared across 4.0 requests/step


─────────────────────────────────────────────────────────────────
SCENARIO 2 — Mixed lengths, N=4
Realistic workload: short and long requests together
─────────────────────────────────────────────────────────────────
[2026-09-27 16:27:00,307] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_00 max_tokens=10
[2026-09-27 16:27:00,738] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_00 tokens=9 time=0.427s blocks_used=3
[2026-09-27 16:27:00,738] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_01 max_tokens=30
[2026-09-27 16:27:01,610] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_01 tokens=21 time=0.872s blocks_used=4
[2026-09-27 16:27:01,610] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_02 max_tokens=80
[2026-09-27 16:27:04,964] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_02 tokens=80 time=3.353s blocks_used=8
[2026-09-27 16:27:04,964] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_03 max_tokens=120
[2026-09-27 16:27:10,106] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_03 tokens=120 time=5.142s blocks_used=10
[2026-09-27 16:27:10,108] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=None total_blocks=512 block_size=16
[2026-09-27 16:27:10,108] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=10 waiting=1
[2026-09-27 16:27:10,108] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_01 max_new_tokens=30 waiting=2
[2026-09-27 16:27:10,108] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_02 max_new_tokens=80 waiting=3
[2026-09-27 16:27:10,108] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_03 max_new_tokens=120 waiting=4
[2026-09-27 16:27:10,220] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=37 blocks_used=3 running=1 waiting=3 free_blocks=509
[2026-09-27 16:27:10,317] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_01 prompt_tokens=37 blocks_used=3 running=2 waiting=2 free_blocks=506
[2026-09-27 16:27:10,455] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_02 prompt_tokens=40 blocks_used=3 running=3 waiting=1 free_blocks=503
[2026-09-27 16:27:10,550] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_03 prompt_tokens=36 blocks_used=3 running=4 waiting=0 free_blocks=500
[2026-09-27 16:27:10,655] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=2 finish_reason=eos released_blocks=3 running=3 waiting=0 free_blocks=503
[2026-09-27 16:27:11,274] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_01 generated=17 finish_reason=eos released_blocks=4 running=2 waiting=0 free_blocks=504
[2026-09-27 16:27:13,964] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_02 generated=80 finish_reason=length released_blocks=8 running=1 waiting=0 free_blocks=504
[2026-09-27 16:27:15,509] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_03 generated=120 finish_reason=length released_blocks=10 running=0 waiting=0 free_blocks=512

=================================================================
  SEQUENTIAL
  Requests : 4  |  Wall time: 9797 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 0.41
Throughput (tok/s)                                 23.5
-----------------------------------------------------------------
Mean latency (ms)                                2449.2
P50  latency (ms)                                2113.0
P95  latency (ms)                                4874.3
P99  latency (ms)                                5089.0
-----------------------------------------------------------------
Mean queue wait (ms)                                0.0
Max  queue wait (ms)                                0.0
-----------------------------------------------------------------
Total tokens generated                               230
=================================================================

=================================================================
  BATCHED FORWARD
  Requests : 4  |  Wall time: 5406 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 0.74
Throughput (tok/s)                                 40.5
-----------------------------------------------------------------
Mean latency (ms)                                2747.4
P50  latency (ms)                                2516.7
P95  latency (ms)                                5173.0
P99  latency (ms)                                5358.1
-----------------------------------------------------------------
Mean queue wait (ms)                              276.9
Max  queue wait (ms)                              440.5
-----------------------------------------------------------------
Total tokens generated                               219
Total decode steps                                   120
Avg batch size / step                              1.82
=================================================================

======================================================================
  COMPARISON  Sequential  →  Batched Forward
======================================================================
Metric                               Sequential          Batched
----------------------------------------------------------------------
Wall time (ms)                            9797            5406
Throughput (tok/s)                        23.5            40.5
Throughput (req/s)                        0.41            0.74
Mean latency (ms)                       2449.2          2747.4
P99 latency (ms)                        5089.0          5358.1
Mean wait (ms)                             0.0           276.9
Avg batch / step                            N/A            1.82
----------------------------------------------------------------------
Wall time speedup                         1.81x
Token throughput gain                     1.73x
Request throughput gain                   1.81x
Mean latency change                       1.12x
======================================================================

  Interpretation:
  ✅ Moderate batching gain: 1.73x
     Partial compute overlap. Try larger batch or longer sequences.


  Sequential per-request:
  ID            Tokens   Reason   Wait(ms)  Total(ms)    tok/s
  ------------------------------------------------------------
  req_00             9      eos        0.0      428.1     21.0
  req_01            21      eos        0.0      872.4     24.1
  req_02            80   length        0.0     3353.6     23.9
  req_03           120   length        0.0     5142.7     23.3

  Batched per-request:
  ID            Tokens   Reason   Wait(ms)  Total(ms)    tok/s
  ------------------------------------------------------------
  req_00             2      eos      112.5      551.7      3.6
  req_01            17      eos      208.7     1171.5     14.5
  req_02            80   length      346.0     3861.9     20.7
  req_03           120   length      440.5     5404.4     22.2

─────────────────────────────────────────────────────────────────
SCENARIO 3 — Throughput scaling
Key test: does tok/s grow with concurrency?
Sequential loop showed flat/declining — batched should rise
─────────────────────────────────────────────────────────────────

   Concurrency    Seq tok/s  Batch tok/s     Gain  Avg batch
  ------------------------------------------------------------
[2026-09-27 16:27:15,514] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_00 max_tokens=10
[2026-09-27 16:27:15,920] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_00 tokens=9 time=0.405s blocks_used=3
[2026-09-27 16:27:15,920] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=None total_blocks=512 block_size=16
[2026-09-27 16:27:15,920] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=10 waiting=1
[2026-09-27 16:27:16,060] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=37 blocks_used=3 running=1 waiting=0 free_blocks=509
[2026-09-27 16:27:16,376] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=9 finish_reason=eos released_blocks=3 running=0 waiting=0 free_blocks=512
             1         22.2         19.9    0.90x       1.00
[2026-09-27 16:27:16,376] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_00 max_tokens=10
[2026-09-27 16:27:16,768] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_00 tokens=9 time=0.392s blocks_used=3
[2026-09-27 16:27:16,768] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_01 max_tokens=30
[2026-09-27 16:27:17,605] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_01 tokens=21 time=0.837s blocks_used=4
[2026-09-27 16:27:17,605] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=None total_blocks=512 block_size=16
[2026-09-27 16:27:17,605] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=10 waiting=1
[2026-09-27 16:27:17,605] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_01 max_new_tokens=30 waiting=2
[2026-09-27 16:27:17,709] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=37 blocks_used=3 running=1 waiting=1 free_blocks=509
[2026-09-27 16:27:17,840] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_01 prompt_tokens=37 blocks_used=3 running=2 waiting=0 free_blocks=506
[2026-09-27 16:27:18,236] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=9 finish_reason=eos released_blocks=3 running=1 waiting=0 free_blocks=509
[2026-09-27 16:27:18,609] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_01 generated=19 finish_reason=eos released_blocks=4 running=0 waiting=0 free_blocks=512
             2         24.4         27.9    1.14x       1.47
[2026-09-27 16:27:18,609] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_00 max_tokens=10
[2026-09-27 16:27:19,000] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_00 tokens=9 time=0.391s blocks_used=3
[2026-09-27 16:27:19,000] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_01 max_tokens=30
[2026-09-27 16:27:19,959] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_01 tokens=21 time=0.960s blocks_used=4
[2026-09-27 16:27:19,964] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_02 max_tokens=80
[2026-09-27 16:27:23,594] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_02 tokens=80 time=3.632s blocks_used=8
[2026-09-27 16:27:23,594] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_03 max_tokens=120
[2026-09-27 16:27:29,384] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_03 tokens=120 time=5.793s blocks_used=10
[2026-09-27 16:27:29,384] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=None total_blocks=512 block_size=16
[2026-09-27 16:27:29,384] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=10 waiting=1
[2026-09-27 16:27:29,384] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_01 max_new_tokens=30 waiting=2
[2026-09-27 16:27:29,384] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_02 max_new_tokens=80 waiting=3
[2026-09-27 16:27:29,384] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_03 max_new_tokens=120 waiting=4
[2026-09-27 16:27:29,490] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=37 blocks_used=3 running=1 waiting=3 free_blocks=509
[2026-09-27 16:27:29,593] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_01 prompt_tokens=37 blocks_used=3 running=2 waiting=2 free_blocks=506
[2026-09-27 16:27:29,694] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_02 prompt_tokens=40 blocks_used=3 running=3 waiting=1 free_blocks=503
[2026-09-27 16:27:29,796] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_03 prompt_tokens=36 blocks_used=3 running=4 waiting=0 free_blocks=500
[2026-09-27 16:27:29,874] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=2 finish_reason=eos released_blocks=3 running=3 waiting=0 free_blocks=503
[2026-09-27 16:27:30,474] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_01 generated=17 finish_reason=eos released_blocks=4 running=2 waiting=0 free_blocks=504
[2026-09-27 16:27:33,132] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_02 generated=80 finish_reason=length released_blocks=8 running=1 waiting=0 free_blocks=504
[2026-09-27 16:27:34,744] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_03 generated=120 finish_reason=length released_blocks=10 running=0 waiting=0 free_blocks=512
             4         21.3         40.9    1.92x       1.82
[2026-09-27 16:27:34,747] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_00 max_tokens=10
[2026-09-27 16:27:35,254] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_00 tokens=9 time=0.506s blocks_used=3
[2026-09-27 16:27:35,254] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_01 max_tokens=30
[2026-09-27 16:27:36,200] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_01 tokens=21 time=0.945s blocks_used=4
[2026-09-27 16:27:36,200] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_02 max_tokens=80
[2026-09-27 16:27:39,614] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_02 tokens=80 time=3.416s blocks_used=8
[2026-09-27 16:27:39,614] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_03 max_tokens=120
[2026-09-27 16:27:45,232] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_03 tokens=120 time=5.616s blocks_used=10
[2026-09-27 16:27:45,232] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_04 max_tokens=20
[2026-09-27 16:27:46,208] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_04 tokens=20 time=0.976s blocks_used=4
[2026-09-27 16:27:46,208] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_05 max_tokens=100
[2026-09-27 16:27:50,720] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_05 tokens=100 time=4.510s blocks_used=9
[2026-09-27 16:27:50,720] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_06 max_tokens=5
[2026-09-27 16:27:51,145] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_06 tokens=5 time=0.427s blocks_used=3
[2026-09-27 16:27:51,145] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_07 max_tokens=90
[2026-09-27 16:27:55,144] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_07 tokens=90 time=3.999s blocks_used=9
[2026-09-27 16:27:55,144] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=None total_blocks=512 block_size=16
[2026-09-27 16:27:55,144] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=10 waiting=1
[2026-09-27 16:27:55,144] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_01 max_new_tokens=30 waiting=2
[2026-09-27 16:27:55,144] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_02 max_new_tokens=80 waiting=3
[2026-09-27 16:27:55,144] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_03 max_new_tokens=120 waiting=4
[2026-09-27 16:27:55,144] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_04 max_new_tokens=20 waiting=5
[2026-09-27 16:27:55,144] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_05 max_new_tokens=100 waiting=6
[2026-09-27 16:27:55,144] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_06 max_new_tokens=5 waiting=7
[2026-09-27 16:27:55,144] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_07 max_new_tokens=90 waiting=8
[2026-09-27 16:27:55,250] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=37 blocks_used=3 running=1 waiting=7 free_blocks=509
[2026-09-27 16:27:55,344] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_01 prompt_tokens=37 blocks_used=3 running=2 waiting=6 free_blocks=506
[2026-09-27 16:27:55,449] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_02 prompt_tokens=40 blocks_used=3 running=3 waiting=5 free_blocks=503
[2026-09-27 16:27:55,544] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_03 prompt_tokens=36 blocks_used=3 running=4 waiting=4 free_blocks=500
[2026-09-27 16:27:55,632] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_04 prompt_tokens=34 blocks_used=3 running=5 waiting=3 free_blocks=497
[2026-09-27 16:27:55,735] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_05 prompt_tokens=38 blocks_used=3 running=6 waiting=2 free_blocks=494
[2026-09-27 16:27:55,827] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_06 prompt_tokens=36 blocks_used=3 running=7 waiting=1 free_blocks=491
[2026-09-27 16:27:55,935] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_07 prompt_tokens=39 blocks_used=3 running=8 waiting=0 free_blocks=488
[2026-09-27 16:27:56,080] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=3 finish_reason=eos released_blocks=3 running=7 waiting=0 free_blocks=491
[2026-09-27 16:27:56,182] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_06 generated=5 finish_reason=length released_blocks=3 running=6 waiting=0 free_blocks=494
[2026-09-27 16:27:56,839] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_01 generated=18 finish_reason=eos released_blocks=4 running=5 waiting=0 free_blocks=492
[2026-09-27 16:27:56,925] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_04 generated=20 finish_reason=length released_blocks=4 running=4 waiting=0 free_blocks=496
[2026-09-27 16:27:59,772] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_02 generated=80 finish_reason=length released_blocks=8 running=3 waiting=0 free_blocks=488
[2026-09-27 16:28:00,199] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_07 generated=90 finish_reason=length released_blocks=9 running=2 waiting=0 free_blocks=496
[2026-09-27 16:28:00,585] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_05 generated=100 finish_reason=length released_blocks=9 running=1 waiting=0 free_blocks=503
[2026-09-27 16:28:01,600] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_03 generated=120 finish_reason=length released_blocks=10 running=0 waiting=0 free_blocks=512
             8         21.8         67.6    3.10x       3.63

─────────────────────────────────────────────────────────────────
SCENARIO 4 — Queue pressure, 12 requests, batch cap=4
Expected: largest wall-time improvement
Short requests free slots immediately for waiting requests
─────────────────────────────────────────────────────────────────
[2026-09-27 16:28:01,600] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_00 max_tokens=10
[2026-09-27 16:28:01,970] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_00 tokens=8 time=0.368s blocks_used=3
[2026-09-27 16:28:01,971] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_01 max_tokens=20
[2026-09-27 16:28:02,813] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_01 tokens=20 time=0.843s blocks_used=4
[2026-09-27 16:28:02,815] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_02 max_tokens=40
[2026-09-27 16:28:04,415] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_02 tokens=40 time=1.601s blocks_used=5
[2026-09-27 16:28:04,415] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_03 max_tokens=10
[2026-09-27 16:28:04,804] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_03 tokens=8 time=0.389s blocks_used=3
[2026-09-27 16:28:04,806] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_04 max_tokens=80
[2026-09-27 16:28:08,424] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_04 tokens=80 time=3.623s blocks_used=8
[2026-09-27 16:28:08,424] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_05 max_tokens=30
[2026-09-27 16:28:09,720] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_05 tokens=30 time=1.290s blocks_used=5
[2026-09-27 16:28:09,720] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_06 max_tokens=50
[2026-09-27 16:28:11,826] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_06 tokens=50 time=2.104s blocks_used=6
[2026-09-27 16:28:11,826] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_07 max_tokens=15
[2026-09-27 16:28:12,554] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_07 tokens=15 time=0.728s blocks_used=4
[2026-09-27 16:28:12,554] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_08 max_tokens=20
[2026-09-27 16:28:13,412] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_08 tokens=20 time=0.859s blocks_used=4
[2026-09-27 16:28:13,415] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_09 max_tokens=60
[2026-09-27 16:28:15,988] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_09 tokens=60 time=2.572s blocks_used=6
[2026-09-27 16:28:15,988] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_10 max_tokens=10
[2026-09-27 16:28:16,474] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_10 tokens=10 time=0.492s blocks_used=4
[2026-09-27 16:28:16,474] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=req_11 max_tokens=45
[2026-09-27 16:28:18,531] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=req_11 tokens=45 time=2.050s blocks_used=5
[2026-09-27 16:28:18,532] forgeserve.scheduler.continuous_batching - INFO - ContinuousBatchScheduler initialised: max_batch_size=4 total_blocks=512 block_size=16
[2026-09-27 16:28:18,532] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_00 max_new_tokens=10 waiting=1
[2026-09-27 16:28:18,532] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_01 max_new_tokens=20 waiting=2
[2026-09-27 16:28:18,533] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_02 max_new_tokens=40 waiting=3
[2026-09-27 16:28:18,533] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_03 max_new_tokens=10 waiting=4
[2026-09-27 16:28:18,533] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_04 max_new_tokens=80 waiting=5
[2026-09-27 16:28:18,533] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_05 max_new_tokens=30 waiting=6
[2026-09-27 16:28:18,533] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_06 max_new_tokens=50 waiting=7
[2026-09-27 16:28:18,535] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_07 max_new_tokens=15 waiting=8
[2026-09-27 16:28:18,535] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_08 max_new_tokens=20 waiting=9
[2026-09-27 16:28:18,535] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_09 max_new_tokens=60 waiting=10
[2026-09-27 16:28:18,535] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_10 max_new_tokens=10 waiting=11
[2026-09-27 16:28:18,535] forgeserve.scheduler.continuous_batching - INFO - Request queued: id=req_11 max_new_tokens=45 waiting=12
[2026-09-27 16:28:18,629] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_00 prompt_tokens=36 blocks_used=3 running=1 waiting=11 free_blocks=509
[2026-09-27 16:28:18,721] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_01 prompt_tokens=34 blocks_used=3 running=2 waiting=10 free_blocks=506
[2026-09-27 16:28:18,821] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_02 prompt_tokens=35 blocks_used=3 running=3 waiting=9 free_blocks=503
[2026-09-27 16:28:18,935] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_03 prompt_tokens=36 blocks_used=3 running=4 waiting=8 free_blocks=500
[2026-09-27 16:28:19,269] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_00 generated=8 finish_reason=eos released_blocks=3 running=3 waiting=8 free_blocks=503
[2026-09-27 16:28:19,275] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_03 generated=8 finish_reason=eos released_blocks=3 running=2 waiting=8 free_blocks=506
[2026-09-27 16:28:19,365] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_04 prompt_tokens=34 blocks_used=3 running=3 waiting=7 free_blocks=503
[2026-09-27 16:28:19,459] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_05 prompt_tokens=36 blocks_used=3 running=4 waiting=6 free_blocks=500
[2026-09-27 16:28:19,844] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_01 generated=17 finish_reason=eos released_blocks=4 running=3 waiting=6 free_blocks=502
[2026-09-27 16:28:19,935] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_06 prompt_tokens=34 blocks_used=3 running=4 waiting=5 free_blocks=499
[2026-09-27 16:28:20,878] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_05 generated=30 finish_reason=length released_blocks=5 running=3 waiting=5 free_blocks=499
[2026-09-27 16:28:20,999] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_07 prompt_tokens=36 blocks_used=3 running=4 waiting=4 free_blocks=496
[2026-09-27 16:28:21,084] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_02 generated=40 finish_reason=length released_blocks=5 running=3 waiting=4 free_blocks=500
[2026-09-27 16:28:21,175] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_08 prompt_tokens=34 blocks_used=3 running=4 waiting=3 free_blocks=497
[2026-09-27 16:28:21,222] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_07 generated=3 finish_reason=eos released_blocks=3 running=3 waiting=3 free_blocks=500
[2026-09-27 16:28:21,316] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_09 prompt_tokens=34 blocks_used=3 running=4 waiting=2 free_blocks=497
[2026-09-27 16:28:22,189] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_08 generated=20 finish_reason=length released_blocks=4 running=3 waiting=2 free_blocks=497
[2026-09-27 16:28:22,295] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_10 prompt_tokens=40 blocks_used=3 running=4 waiting=1 free_blocks=494
[2026-09-27 16:28:22,594] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_06 generated=50 finish_reason=length released_blocks=6 running=3 waiting=1 free_blocks=499
[2026-09-27 16:28:22,685] forgeserve.scheduler.continuous_batching - INFO - Request admitted: id=req_11 prompt_tokens=35 blocks_used=3 running=4 waiting=0 free_blocks=496
[2026-09-27 16:28:22,815] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_10 generated=10 finish_reason=length released_blocks=4 running=3 waiting=0 free_blocks=499
[2026-09-27 16:28:23,737] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_04 generated=80 finish_reason=length released_blocks=8 running=2 waiting=0 free_blocks=502
[2026-09-27 16:28:24,255] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_09 generated=60 finish_reason=length released_blocks=6 running=1 waiting=0 free_blocks=507
[2026-09-27 16:28:24,690] forgeserve.scheduler.continuous_batching - INFO - Request finished: id=req_11 generated=45 finish_reason=length released_blocks=5 running=0 waiting=0 free_blocks=512

=================================================================
  SEQUENTIAL
  Requests : 12  |  Wall time: 16930 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 0.71
Throughput (tok/s)                                 22.8
-----------------------------------------------------------------
Mean latency (ms)                                1410.8
P50  latency (ms)                                1075.1
P95  latency (ms)                                3046.2
P99  latency (ms)                                3508.8
-----------------------------------------------------------------
Mean queue wait (ms)                                0.0
Max  queue wait (ms)                                0.0
-----------------------------------------------------------------
Total tokens generated                               386
=================================================================

=================================================================
  BATCHED FORWARD
  Requests : 12  |  Wall time: 6158 ms
=================================================================
Metric                                             Value
-----------------------------------------------------------------
Throughput (req/s)                                 1.95
Throughput (tok/s)                                 60.2
-----------------------------------------------------------------
Mean latency (ms)                                3290.2
P50  latency (ms)                                3174.2
P95  latency (ms)                                5919.8
P99  latency (ms)                                6107.6
-----------------------------------------------------------------
Mean queue wait (ms)                             1663.3
Max  queue wait (ms)                             4154.9
-----------------------------------------------------------------
Total tokens generated                               371
Total decode steps                                   112
Avg batch size / step                              3.31
=================================================================

======================================================================
  COMPARISON  Sequential  →  Batched Forward
======================================================================
Metric                               Sequential          Batched
----------------------------------------------------------------------
Wall time (ms)                           16930            6158
Throughput (tok/s)                        22.8            60.2
Throughput (req/s)                        0.71            1.95
Mean latency (ms)                       1410.8          3290.2
P99 latency (ms)                        3508.8          6107.6
Mean wait (ms)                             0.0          1663.3
Avg batch / step                            N/A            3.31
----------------------------------------------------------------------
Wall time speedup                         2.75x
Token throughput gain                     2.64x
Request throughput gain                   2.75x
Mean latency change                       2.33x
======================================================================

  Interpretation:
  ✅ Moderate batching gain: 2.64x
     Partial compute overlap. Try larger batch or longer sequences.
  ℹ  Individual latency 2.33x worse — expected trade-off.
     Each request waits for batch-mates. Total wall time still wins.


  Sequential per-request:
  ID            Tokens   Reason   Wait(ms)  Total(ms)    tok/s
  ------------------------------------------------------------
  req_00             8      eos        0.0      368.9     21.7
  req_01            20   length        0.0      843.5     23.7
  req_02            40   length        0.0     1601.9     25.0
  req_03             8      eos        0.0      389.5     20.5
  req_04            80   length        0.0     3624.4     22.1
  req_05            30   length        0.0     1290.9     23.2
  req_06            50   length        0.0     2105.1     23.8
  req_07            15   length        0.0      729.2     20.6
  req_08            20   length        0.0      859.4     23.3
  req_09            60   length        0.0     2573.2     23.3
  req_10            10   length        0.0      493.2     20.3
  req_11            45   length        0.0     2050.4     21.9

  Batched per-request:
  ID            Tokens   Reason   Wait(ms)  Total(ms)    tok/s
  ------------------------------------------------------------
  req_00             8      eos       99.2      741.8     10.8
  req_01            17      eos      189.9     1312.5     13.0
  req_02            40   length      287.9     2553.7     15.7
  req_03             8      eos      402.9      741.3     10.8
  req_04            80   length      832.1     5208.8     15.4
  req_05            30   length      929.8     2349.7     12.8
  req_06            50   length     1406.6     4060.8     12.3
  req_07             3      eos     2464.9     2689.2      1.1
  req_08            20   length     2644.8     3659.2      5.5
  req_09            60   length     2781.1     5727.6     10.5
  req_10            10   length     3765.0     4282.9      2.3
  req_11            45   length     4154.9     6154.6      7.3