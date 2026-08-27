======================================================================
ForgeServe PagedAttention Benchmark
======================================================================
Model       : Qwen/Qwen2.5-0.5B-Instruct
Block size  : 16
Num blocks  : 256
Warmup runs : 2
Bench runs  : 5
[2026-08-25 20:29:43,854] forgeserve.page_attention.block_manager - INFO - Pre-allocating 256 KV blocks (block_size=16). Total KV memory: 48.0 MB
[2026-08-25 20:29:44,020] forgeserve.page_attention.block_manager - INFO - BlockManager ready. 256 blocks available
[2026-08-25 20:29:44,020] forgeserve.model.paged_runtime - INFO - PagedRuntime initialized. Block pool: 256 blocks, block_size=16, memory=48.0 MB


>>> Scenario: single_request
Warming up with 2 runs...
[2026-08-25 20:29:44,021] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=7a1c7223 max_tokens=128
[2026-08-25 20:29:49,851] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=7a1c7223 tokens=128 time=5.838s blocks_used=11
[2026-08-25 20:29:49,851] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=6e743c74 max_tokens=128
[2026-08-25 20:29:54,815] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=6e743c74 tokens=128 time=4.955s blocks_used=11
Warmup complete.

======================================================================
Requests       : 1
Max new tokens : 128
Block size     : 16
Total blocks   : 256
======================================================================
[2026-08-25 20:29:54,819] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=54538cdb max_tokens=128
[2026-08-25 20:29:59,765] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=54538cdb tokens=128 time=4.944s blocks_used=11

Run 1/5
Generated tokens : 128
Total time       : 4945.23 ms
Throughput       : 25.88 tok/s
Peak memory      : 1018.47 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:29:59,767] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=856dc3fa max_tokens=128
[2026-08-25 20:30:04,897] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=856dc3fa tokens=128 time=5.134s blocks_used=11

Run 2/5
Generated tokens : 128
Total time       : 5135.34 ms
Throughput       : 24.93 tok/s
Peak memory      : 1018.47 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:30:04,897] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=232b26dd max_tokens=128
[2026-08-25 20:30:09,917] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=232b26dd tokens=128 time=5.012s blocks_used=11

Run 3/5
Generated tokens : 128
Total time       : 5013.25 ms
Throughput       : 25.53 tok/s
Peak memory      : 1018.47 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:30:09,917] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=4744a6a6 max_tokens=128
[2026-08-25 20:30:15,037] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=4744a6a6 tokens=128 time=5.117s blocks_used=11

Run 4/5
Generated tokens : 128
Total time       : 5118.16 ms
Throughput       : 25.01 tok/s
Peak memory      : 1018.47 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:30:15,037] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=c399a1a9 max_tokens=128
[2026-08-25 20:30:20,084] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=c399a1a9 tokens=128 time=5.045s blocks_used=11

Run 5/5
Generated tokens : 128
Total time       : 5046.39 ms
Throughput       : 25.36 tok/s
Peak memory      : 1018.47 MB
Free blocks      : 256
Block leak       : False


>>> Scenario: two_requests
Warming up with 2 runs...
[2026-08-25 20:30:20,084] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=e2c32b73 max_tokens=128
[2026-08-25 20:30:25,132] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=e2c32b73 tokens=128 time=5.052s blocks_used=11
[2026-08-25 20:30:25,132] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=f0084a1c max_tokens=128
[2026-08-25 20:30:30,221] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=f0084a1c tokens=128 time=5.084s blocks_used=11
Warmup complete.

======================================================================
Requests       : 2
Max new tokens : 128
Block size     : 16
Total blocks   : 256
======================================================================
[2026-08-25 20:30:30,221] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=f4b9881b max_tokens=128
[2026-08-25 20:30:35,434] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=f4b9881b tokens=128 time=5.211s blocks_used=11
[2026-08-25 20:30:35,436] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=55cde81c max_tokens=128
[2026-08-25 20:30:40,696] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=55cde81c tokens=128 time=5.260s blocks_used=11

Run 1/5
Generated tokens : 256
Total time       : 10472.81 ms
Throughput       : 24.44 tok/s
Peak memory      : 1019.67 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:30:40,696] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=98b7a3f0 max_tokens=128
[2026-08-25 20:30:45,952] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=98b7a3f0 tokens=128 time=5.255s blocks_used=11
[2026-08-25 20:30:45,952] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=7f6873b9 max_tokens=128
[2026-08-25 20:30:51,047] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=7f6873b9 tokens=128 time=5.096s blocks_used=11

Run 2/5
Generated tokens : 256
Total time       : 10352.92 ms
Throughput       : 24.73 tok/s
Peak memory      : 1019.67 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:30:51,047] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=55f26783 max_tokens=128
[2026-08-25 20:30:56,245] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=55f26783 tokens=128 time=5.194s blocks_used=11
[2026-08-25 20:30:56,245] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=210d1952 max_tokens=128
[2026-08-25 20:31:01,378] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=210d1952 tokens=128 time=5.132s blocks_used=11

Run 3/5
Generated tokens : 256
Total time       : 10327.53 ms
Throughput       : 24.79 tok/s
Peak memory      : 1019.67 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:31:01,379] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=558a0493 max_tokens=128
[2026-08-25 20:31:06,466] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=558a0493 tokens=128 time=5.088s blocks_used=11
[2026-08-25 20:31:06,466] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=49d8cb81 max_tokens=128
[2026-08-25 20:31:11,593] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=49d8cb81 tokens=128 time=5.126s blocks_used=11

Run 4/5
Generated tokens : 256
Total time       : 10215.60 ms
Throughput       : 25.06 tok/s
Peak memory      : 1019.67 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:31:11,595] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=ffa6508f max_tokens=128
[2026-08-25 20:31:16,751] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=ffa6508f tokens=128 time=5.156s blocks_used=11
[2026-08-25 20:31:16,751] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=a30bae3b max_tokens=128
[2026-08-25 20:31:21,868] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=a30bae3b tokens=128 time=5.117s blocks_used=11

Run 5/5
Generated tokens : 256
Total time       : 10274.98 ms
Throughput       : 24.91 tok/s
Peak memory      : 1019.67 MB
Free blocks      : 256
Block leak       : False


>>> Scenario: four_requests
Warming up with 2 runs...
[2026-08-25 20:31:21,868] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=5a3c3227 max_tokens=128
[2026-08-25 20:31:26,924] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=5a3c3227 tokens=128 time=5.061s blocks_used=11
[2026-08-25 20:31:26,924] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=02b4fd6d max_tokens=128
[2026-08-25 20:31:31,990] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=02b4fd6d tokens=128 time=5.057s blocks_used=11
Warmup complete.

======================================================================
Requests       : 4
Max new tokens : 128
Block size     : 16
Total blocks   : 256
======================================================================
[2026-08-25 20:31:31,990] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=fd73d0de max_tokens=128
[2026-08-25 20:31:37,809] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=fd73d0de tokens=128 time=5.819s blocks_used=11
[2026-08-25 20:31:37,809] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=b1b20299 max_tokens=128
[2026-08-25 20:31:43,372] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=b1b20299 tokens=128 time=5.570s blocks_used=11
[2026-08-25 20:31:43,382] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=1a8b219d max_tokens=128
[2026-08-25 20:31:49,253] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=1a8b219d tokens=128 time=5.878s blocks_used=11
[2026-08-25 20:31:49,253] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=4d6f7ddb max_tokens=128
[2026-08-25 20:31:55,164] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=4d6f7ddb tokens=128 time=5.910s blocks_used=11

Run 1/5
Generated tokens : 512
Total time       : 23181.28 ms
Throughput       : 22.09 tok/s
Peak memory      : 1017.87 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:31:55,164] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=3ca3eade max_tokens=128
[2026-08-25 20:32:01,144] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=3ca3eade tokens=128 time=5.975s blocks_used=11
[2026-08-25 20:32:01,144] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=df9dd597 max_tokens=128
[2026-08-25 20:32:06,814] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=df9dd597 tokens=128 time=5.669s blocks_used=11
[2026-08-25 20:32:06,814] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=02893e04 max_tokens=128
[2026-08-25 20:32:12,364] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=02893e04 tokens=128 time=5.546s blocks_used=11
[2026-08-25 20:32:12,364] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=f74c44f4 max_tokens=128
[2026-08-25 20:32:17,792] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=f74c44f4 tokens=128 time=5.432s blocks_used=11

Run 2/5
Generated tokens : 512
Total time       : 22626.02 ms
Throughput       : 22.63 tok/s
Peak memory      : 1017.87 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:32:17,800] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=afad5595 max_tokens=128
[2026-08-25 20:32:23,314] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=afad5595 tokens=128 time=5.514s blocks_used=11
[2026-08-25 20:32:23,314] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=5ef6eb18 max_tokens=128
[2026-08-25 20:32:28,787] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=5ef6eb18 tokens=128 time=5.472s blocks_used=11
[2026-08-25 20:32:28,787] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=8efd5261 max_tokens=128
[2026-08-25 20:32:34,012] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=8efd5261 tokens=128 time=5.225s blocks_used=11
[2026-08-25 20:32:34,012] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=d7d7367e max_tokens=128
[2026-08-25 20:32:39,379] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=d7d7367e tokens=128 time=5.373s blocks_used=11

Run 3/5
Generated tokens : 512
Total time       : 21587.72 ms
Throughput       : 23.72 tok/s
Peak memory      : 1017.87 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:32:39,389] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=bafd0177 max_tokens=128
[2026-08-25 20:32:44,770] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=bafd0177 tokens=128 time=5.390s blocks_used=11
[2026-08-25 20:32:44,770] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=554b6aa6 max_tokens=128
[2026-08-25 20:32:50,209] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=554b6aa6 tokens=128 time=5.430s blocks_used=11
[2026-08-25 20:32:50,209] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=62d253df max_tokens=128
[2026-08-25 20:32:55,587] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=62d253df tokens=128 time=5.377s blocks_used=11
[2026-08-25 20:32:55,587] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=9382e4a9 max_tokens=128
[2026-08-25 20:33:01,016] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=9382e4a9 tokens=128 time=5.428s blocks_used=11

Run 4/5
Generated tokens : 512
Total time       : 21629.62 ms
Throughput       : 23.67 tok/s
Peak memory      : 1017.87 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:33:01,016] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=28148a35 max_tokens=128
[2026-08-25 20:33:06,423] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=28148a35 tokens=128 time=5.410s blocks_used=11
[2026-08-25 20:33:06,423] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=b03e457b max_tokens=128
[2026-08-25 20:33:11,757] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=b03e457b tokens=128 time=5.334s blocks_used=11
[2026-08-25 20:33:11,757] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=c3a8daf8 max_tokens=128
[2026-08-25 20:33:17,396] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=c3a8daf8 tokens=128 time=5.636s blocks_used=11
[2026-08-25 20:33:17,396] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=85d05171 max_tokens=128
[2026-08-25 20:33:22,913] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=85d05171 tokens=128 time=5.511s blocks_used=11

Run 5/5
Generated tokens : 512
Total time       : 21894.99 ms
Throughput       : 23.38 tok/s
Peak memory      : 1017.87 MB
Free blocks      : 256
Block leak       : False


>>> Scenario: eight_requests
Warming up with 2 runs...
[2026-08-25 20:33:22,913] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=ece0f64f max_tokens=128
[2026-08-25 20:33:28,239] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=ece0f64f tokens=128 time=5.327s blocks_used=11
[2026-08-25 20:33:28,239] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=5aeee035 max_tokens=128
[2026-08-25 20:33:33,628] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=5aeee035 tokens=128 time=5.386s blocks_used=11
Warmup complete.

======================================================================
Requests       : 8
Max new tokens : 128
Block size     : 16
Total blocks   : 256
======================================================================
[2026-08-25 20:33:33,628] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=b4f56705 max_tokens=128
[2026-08-25 20:33:38,947] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=b4f56705 tokens=128 time=5.315s blocks_used=11
[2026-08-25 20:33:38,947] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=8015f68b max_tokens=128
[2026-08-25 20:33:44,098] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=8015f68b tokens=128 time=5.153s blocks_used=11
[2026-08-25 20:33:44,098] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=4fbd5ce3 max_tokens=128
[2026-08-25 20:33:49,267] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=4fbd5ce3 tokens=128 time=5.171s blocks_used=11
[2026-08-25 20:33:49,267] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=ef7fe872 max_tokens=128
[2026-08-25 20:33:54,571] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=ef7fe872 tokens=128 time=5.300s blocks_used=11
[2026-08-25 20:33:54,575] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=f7702ac7 max_tokens=128
[2026-08-25 20:34:00,250] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=f7702ac7 tokens=128 time=5.679s blocks_used=11
[2026-08-25 20:34:00,250] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=5eb69aa0 max_tokens=128
[2026-08-25 20:34:05,431] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=5eb69aa0 tokens=128 time=5.183s blocks_used=11
[2026-08-25 20:34:05,431] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=d9913a20 max_tokens=128
[2026-08-25 20:34:10,435] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=d9913a20 tokens=128 time=5.002s blocks_used=11
[2026-08-25 20:34:10,435] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=0b2b6dba max_tokens=128
[2026-08-25 20:34:15,494] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=0b2b6dba tokens=128 time=5.059s blocks_used=11

Run 1/5
Generated tokens : 1024
Total time       : 41869.70 ms
Throughput       : 24.46 tok/s
Peak memory      : 1018.18 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:34:15,494] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=f9331e43 max_tokens=128
[2026-08-25 20:34:20,517] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=f9331e43 tokens=128 time=5.023s blocks_used=11
[2026-08-25 20:34:20,517] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=6533259f max_tokens=128
[2026-08-25 20:34:25,538] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=6533259f tokens=128 time=5.013s blocks_used=11
[2026-08-25 20:34:25,538] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=9ac77d39 max_tokens=128
[2026-08-25 20:34:30,536] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=9ac77d39 tokens=128 time=4.998s blocks_used=11
[2026-08-25 20:34:30,536] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=663902fe max_tokens=128
[2026-08-25 20:34:35,563] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=663902fe tokens=128 time=5.024s blocks_used=11
[2026-08-25 20:34:35,563] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=8f296063 max_tokens=128
[2026-08-25 20:34:40,597] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=8f296063 tokens=128 time=5.036s blocks_used=11
[2026-08-25 20:34:40,597] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=88f387d4 max_tokens=128
[2026-08-25 20:34:45,617] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=88f387d4 tokens=128 time=5.021s blocks_used=11
[2026-08-25 20:34:45,617] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=709338a5 max_tokens=128
[2026-08-25 20:34:50,675] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=709338a5 tokens=128 time=5.058s blocks_used=11
[2026-08-25 20:34:50,675] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=6bbc1d35 max_tokens=128
[2026-08-25 20:34:55,686] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=6bbc1d35 tokens=128 time=5.006s blocks_used=11

Run 2/5
Generated tokens : 1024
Total time       : 40186.55 ms
Throughput       : 25.48 tok/s
Peak memory      : 1018.18 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:34:55,686] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=59ee2783 max_tokens=128
[2026-08-25 20:35:00,698] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=59ee2783 tokens=128 time=5.013s blocks_used=11
[2026-08-25 20:35:00,698] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=5482d1b9 max_tokens=128
[2026-08-25 20:35:05,702] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=5482d1b9 tokens=128 time=5.004s blocks_used=11
[2026-08-25 20:35:05,702] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=2628b5ce max_tokens=128
[2026-08-25 20:35:10,721] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=2628b5ce tokens=128 time=5.017s blocks_used=11
[2026-08-25 20:35:10,721] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=389d704d max_tokens=128
[2026-08-25 20:35:15,736] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=389d704d tokens=128 time=5.013s blocks_used=11
[2026-08-25 20:35:15,736] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=1147296e max_tokens=128
[2026-08-25 20:35:20,730] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=1147296e tokens=128 time=4.993s blocks_used=11
[2026-08-25 20:35:20,730] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=19e9e626 max_tokens=128
[2026-08-25 20:35:25,724] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=19e9e626 tokens=128 time=4.989s blocks_used=11
[2026-08-25 20:35:25,725] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=1eda265b max_tokens=128
[2026-08-25 20:35:30,737] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=1eda265b tokens=128 time=5.012s blocks_used=11
[2026-08-25 20:35:30,737] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=d2c0a571 max_tokens=128
[2026-08-25 20:35:35,937] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=d2c0a571 tokens=128 time=5.207s blocks_used=11

Run 3/5
Generated tokens : 1024
Total time       : 40255.44 ms
Throughput       : 25.44 tok/s
Peak memory      : 1018.18 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:35:35,947] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=e1526fb0 max_tokens=128
[2026-08-25 20:35:41,181] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=e1526fb0 tokens=128 time=5.237s blocks_used=11
[2026-08-25 20:35:41,181] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=df29716b max_tokens=128
[2026-08-25 20:35:46,448] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=df29716b tokens=128 time=5.263s blocks_used=11
[2026-08-25 20:35:46,448] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=c239de47 max_tokens=128
[2026-08-25 20:35:52,013] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=c239de47 tokens=128 time=5.571s blocks_used=11
[2026-08-25 20:35:52,013] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=0ef24835 max_tokens=128
[2026-08-25 20:35:57,710] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=0ef24835 tokens=128 time=5.689s blocks_used=11
[2026-08-25 20:35:57,710] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=7a8193a7 max_tokens=128
[2026-08-25 20:36:03,048] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=7a8193a7 tokens=128 time=5.342s blocks_used=11
[2026-08-25 20:36:03,048] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=861145e6 max_tokens=128
[2026-08-25 20:36:08,163] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=861145e6 tokens=128 time=5.117s blocks_used=11
[2026-08-25 20:36:08,163] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=9d776315 max_tokens=128
[2026-08-25 20:36:13,374] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=9d776315 tokens=128 time=5.203s blocks_used=11
[2026-08-25 20:36:13,374] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=3eca56c0 max_tokens=128
[2026-08-25 20:36:18,509] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=3eca56c0 tokens=128 time=5.140s blocks_used=11

Run 4/5
Generated tokens : 1024
Total time       : 42569.02 ms
Throughput       : 24.06 tok/s
Peak memory      : 1018.18 MB
Free blocks      : 256
Block leak       : False
[2026-08-25 20:36:18,509] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=d4173a58 max_tokens=128
[2026-08-25 20:36:23,596] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=d4173a58 tokens=128 time=5.080s blocks_used=11
[2026-08-25 20:36:23,596] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=c27dec4e max_tokens=128
[2026-08-25 20:36:28,691] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=c27dec4e tokens=128 time=5.098s blocks_used=11
[2026-08-25 20:36:28,691] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=a7746def max_tokens=128
[2026-08-25 20:36:33,846] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=a7746def tokens=128 time=5.150s blocks_used=11
[2026-08-25 20:36:33,846] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=43347b19 max_tokens=128
[2026-08-25 20:36:38,906] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=43347b19 tokens=128 time=5.062s blocks_used=11
[2026-08-25 20:36:38,906] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=31d5c059 max_tokens=128
[2026-08-25 20:36:44,311] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=31d5c059 tokens=128 time=5.403s blocks_used=11
[2026-08-25 20:36:44,311] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=cee9c30a max_tokens=128
[2026-08-25 20:36:49,698] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=cee9c30a tokens=128 time=5.383s blocks_used=11
[2026-08-25 20:36:49,698] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=cca15791 max_tokens=128
[2026-08-25 20:36:54,972] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=cca15791 tokens=128 time=5.281s blocks_used=11
[2026-08-25 20:36:54,972] forgeserve.engine.paged_generation - INFO - Starting paged generation: request=09f612c2 max_tokens=128
[2026-08-25 20:37:00,165] forgeserve.engine.paged_generation - INFO - Paged generation complete: request=09f612c2 tokens=128 time=5.191s blocks_used=11

Run 5/5
Generated tokens : 1024
Total time       : 41655.25 ms
Throughput       : 24.58 tok/s
Peak memory      : 1018.18 MB
Free blocks      : 256
Block leak       : False

======================================================================
PagedAttention benchmark completed successfully.
======================================================================

i think there is something wrong the number of blocks used across all request are the same what about you analyze it 