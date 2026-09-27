Title: Batched Continuous Decoding
Description: Multiple active requests are combined into a single batched forward pass at each decoding step.

Decode step 1:
  forward([req_1, req_2, req_3])

Decode step 2:
  forward([req_1, req_2, req_3])

# Phase 5B Benchmark — Batched Continuous Decoding

## Introduction

Phase 5B introduces true batched decode execution into ForgeServe.

The scheduler already supports multiple active requests, but the earlier implementation performed a separate model forward pass for each request:

```text
Request A → forward(A)
Request B → forward(B)
Request C → forward(C)
Request D → forward(D)
```

Phase 5B changes this to:

```text
Decode step 1:
    forward([A, B, C, D])

Decode step 2:
    forward([A, B, C, D])

...
```

Each scheduling step collects the next token from every active request and processes all active requests in one batched forward pass.

The purpose of this benchmark is to measure the effect of batched forward execution on:

- wall-clock completion time
- token throughput
- request throughput
- individual request latency
- queue waiting time
- effective batch size
- behavior under uniform and heterogeneous workloads

---

# Benchmark Configuration

| Setting | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| GPU | NVIDIA RTX 4070 |
| Dtype | bfloat16 |
| Attention | PyTorch SDPA |
| Total KV blocks | 512 |
| Block size | 16 tokens |
| KV pool | 96.0 MB |
| Maximum cached tokens | 8,192 |
| Sampling | Greedy |

The block pool is:

```text
512 blocks × 16 tokens/block
=
8,192 token slots
```

The KV block pool reserves approximately:

```text
96.0 MB
```

The benchmark compares:

```text
SEQUENTIAL
    one request → one forward pass at a time

vs.

BATCHED FORWARD
    all currently active requests
    → one batched forward pass per decode step
```

---

# What Changed in Phase 5B?

## Sequential Decode

With four active requests, the previous execution model effectively performs separate forward passes for each request.

```text
forward(A)
forward(B)
forward(C)
forward(D)
```

The model weights remain resident in VRAM after model loading. However, each request launches a separate forward computation and separately processes the model, creating repeated execution and compute overhead.

---

## Batched Decode

The new implementation collects the current token from each active request:

```text
[A_token]
[B_token]
[C_token]
[D_token]
```

and forms one batch:

```text
batch_tokens.shape = (4, 1)
```

The model then performs:

```text
forward([A, B, C, D])
```

in one call.

The output contains one next-token distribution for each request:

```text
logits[0] → Request A
logits[1] → Request B
logits[2] → Request C
logits[3] → Request D
```

The scheduler then updates each request independently.

---

# Scenario 1 — Uniform Lengths, N=4

## Workload

All four requests generated exactly:

```text
80 tokens/request
```

No early EOS termination occurred.

Therefore:

```text
Total generated tokens
=
4 × 80
=
320 tokens
```

Because every request had the same generation length, all four requests remained active for all 80 decode steps.

The measured average batch size was:

```text
4.00 requests/step
```

## Sequential Results

| Metric | Value |
|---|---:|
| Requests | 4 |
| Wall time | 15,433 ms |
| Throughput | 20.7 tok/s |
| Request throughput | 0.26 req/s |
| Mean latency | 3,858.2 ms |
| P50 latency | 3,895.6 ms |
| P95 latency | 4,124.5 ms |
| P99 latency | 4,139.4 ms |
| Mean queue wait | 0.0 ms |
| Total tokens | 320 |

## Batched Results

| Metric | Value |
|---|---:|
| Requests | 4 |
| Wall time | 4,307 ms |
| Throughput | 74.3 tok/s |
| Request throughput | 0.93 req/s |
| Mean latency | 4,305.3 ms |
| P50 latency | 4,305.3 ms |
| P95 latency | 4,305.8 ms |
| P99 latency | 4,305.9 ms |
| Mean queue wait | 319.2 ms |
| Total tokens | 320 |
| Decode steps | 80 |
| Average batch size | 4.00 |

## Improvement

### Wall time

```text
Sequential : 15,433 ms
Batched    :  4,307 ms
```

Measured speedup:

```text
3.58×
```

### Token throughput

```text
Sequential : 20.7 tok/s
Batched    : 74.3 tok/s
```

Measured gain:

```text
3.58×
```

### Request throughput

```text
Sequential : 0.26 req/s
Batched    : 0.93 req/s
```

Measured gain:

```text
3.58×
```

## Interpretation

This is the clearest demonstration of batched decoding in ForgeServe.

All four requests stay active for the complete 80-step decode. Instead of performing four independent forward passes per scheduling round, ForgeServe performs one forward pass containing all four requests.

The effective batch size is:

```text
4.00 requests/step
```

The measured result is:

```text
3.58× wall-time speedup
3.58× token-throughput gain
3.58× request-throughput gain
```

---

# Scenario 2 — Mixed Lengths, N=4

## Workload

The requested generation lengths were:

```text
10
30
80
120
```

However, some requests terminated early because of EOS.

Actual sequential generation:

```text
9
21
80
120
```

Total:

```text
230 tokens
```

Actual batched generation:

```text
2
17
80
120
```

Total:

```text
219 tokens
```

Therefore, the sequential and batched runs did not generate exactly the same number of tokens in this scenario.

This means throughput and wall-time comparisons should be interpreted with that difference in mind.

## Sequential Results

| Metric | Value |
|---|---:|
| Requests | 4 |
| Wall time | 9,797 ms |
| Throughput | 23.5 tok/s |
| Request throughput | 0.41 req/s |
| Mean latency | 2,449.2 ms |
| P50 latency | 2,113.0 ms |
| P95 latency | 4,874.3 ms |
| P99 latency | 5,089.0 ms |
| Mean queue wait | 0.0 ms |
| Total tokens | 230 |

## Batched Results

| Metric | Value |
|---|---:|
| Requests | 4 |
| Wall time | 5,406 ms |
| Throughput | 40.5 tok/s |
| Request throughput | 0.74 req/s |
| Mean latency | 2,747.4 ms |
| P50 latency | 2,516.7 ms |
| P95 latency | 5,173.0 ms |
| P99 latency | 5,358.1 ms |
| Mean queue wait | 276.9 ms |
| Total tokens | 219 |
| Decode steps | 120 |
| Average batch size | 1.82 |

## Improvement

Measured:

```text
Wall-time speedup       = 1.81×
Token-throughput gain   = 1.73×
Request-throughput gain = 1.81×
```

The average batch size was:

```text
1.82 requests/step
```

This is much lower than Scenario 1's:

```text
4.00 requests/step
```

because shorter requests finish earlier and leave the active batch.

## Interpretation

This is a more realistic continuous-batching workload.

Initially:

```text
[A, B, C, D]
```

are active.

When a short request finishes:

```text
[A, B, C, D]
        ↓
[B, C, D]
```

The effective batch size shrinks.

As a result, the GPU receives less parallel work later in the generation.

The benchmark measured:

```text
1.73× token-throughput gain
1.81× wall-time speedup
```

So batching still helps, but the benefit is smaller than the uniform workload.

## Individual Latency Trade-off

Sequential mean latency:

```text
2,449.2 ms
```

Batched mean latency:

```text
2,747.4 ms
```

The benchmark reports:

```text
Mean latency change = 1.12×
```

This demonstrates an important scheduling trade-off:

> Continuous batching improves aggregate system throughput and total completion time, but individual request latency can increase because requests participate in a shared decode schedule.

---

# Scenario 3 — Throughput Scaling

## Purpose

This scenario asks:

> Does throughput increase when more requests are decoded together?

## Results

| Concurrency | Sequential tok/s | Batched tok/s | Gain | Avg batch size |
|---:|---:|---:|---:|---:|
| 1 | 22.2 | 19.9 | 0.90× | 1.00 |
| 2 | 24.4 | 27.9 | 1.14× | 1.47 |
| 4 | 21.3 | 40.9 | 1.92× | 1.82 |
| 8 | 21.8 | 67.6 | 3.10× | 3.63 |

## Observation

The sequential implementation stays roughly flat:

```text
22.2
24.4
21.3
21.8 tok/s
```

As concurrency increases, the per-request forward passes do not combine into larger GPU operations.

The batched implementation shows much stronger scaling:

```text
19.9
27.9
40.9
67.6 tok/s
```

The largest measured gain in this scenario was:

```text
3.10×
```

at concurrency 8.

The effective batch size was:

```text
3.63 requests/step
```

rather than 8.00 because different requests terminated at different times.

## Important Insight

The number of submitted requests is not necessarily the same as the average batch size.

For example:

```text
8 submitted requests
```

does not imply:

```text
batch size = 8 for every decode step
```

because short requests can finish early.

The actual measured average was:

```text
3.63
```

This makes average batch size an important benchmark metric.

---

# Scenario 4 — Queue Pressure

## Workload

This scenario uses:

```text
12 requests
```

with a maximum batch size of:

```text
4
```

Therefore the scheduler can keep at most four active requests in one decode batch.

The workload contains requests with different lengths:

```text
10
20
40
10
80
30
50
15
20
60
10
45
```

Several short requests finish early and free slots for waiting requests.

This demonstrates the dynamic nature of continuous batching.

## Sequential Results

| Metric | Value |
|---|---:|
| Requests | 12 |
| Wall time | 16,930 ms |
| Throughput | 22.8 tok/s |
| Request throughput | 0.71 req/s |
| Mean latency | 1,410.8 ms |
| P50 latency | 1,075.1 ms |
| P95 latency | 3,046.2 ms |
| P99 latency | 3,508.8 ms |
| Mean queue wait | 0.0 ms |
| Total tokens | 386 |

## Batched Results

| Metric | Value |
|---|---:|
| Requests | 12 |
| Wall time | 6,158 ms |
| Throughput | 60.2 tok/s |
| Request throughput | 1.95 req/s |
| Mean latency | 3,290.2 ms |
| P50 latency | 3,174.2 ms |
| P95 latency | 5,919.8 ms |
| P99 latency | 6,107.6 ms |
| Mean queue wait | 1,663.3 ms |
| Max queue wait | 4,154.9 ms |
| Total tokens | 371 |
| Decode steps | 112 |
| Average batch size | 3.31 |

## Improvement

Measured:

```text
Wall-time speedup       = 2.75×
Token-throughput gain   = 2.64×
Request-throughput gain = 2.75×
```

The average batch size was:

```text
3.31
```

which is close to the configured maximum of:

```text
4
```

## Why Requests Enter and Leave the Batch

At the beginning, the scheduler admits:

```text
Request 0
Request 1
Request 2
Request 3
```

so:

```text
running = 4
waiting = 8
```

When short requests finish, they immediately release their KV blocks and active slots.

The scheduler can then admit waiting requests:

```text
short request finishes
        ↓
slot becomes available
        ↓
waiting request admitted
        ↓
new request joins the batch
```

This is the defining behavior of continuous batching.

The batch is therefore dynamic:

```text
[A B C D]
[A B C D]
[A B D E]
[B D E F]
[D E F G]
...
```

rather than remaining fixed for the entire workload.

## Latency Trade-off Under Queue Pressure

Sequential mean latency:

```text
1,410.8 ms
```

Batched mean latency:

```text
3,290.2 ms
```

The benchmark reports:

```text
Mean latency change = 2.33×
```

The batched system also introduces:

```text
Mean queue wait = 1,663.3 ms
Max queue wait  = 4,154.9 ms
```

This is expected from the configured batch cap and the workload ordering.

A request that arrives behind other requests may need to wait before entering the active batch.

Therefore, the system achieves:

```text
2.75× lower total wall time
2.64× higher token throughput
```

while individual requests may experience greater end-to-end latency.

---

# Cross-Scenario Summary

| Scenario | Requests | Batch Limit | Sequential tok/s | Batched tok/s | Throughput Gain | Wall-time Speedup | Avg Batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 4 | — | 20.7 | 74.3 | 3.58× | 3.58× | 4.00 |
| Mixed | 4 | — | 23.5 | 40.5 | 1.73× | 1.81× | 1.82 |
| Scaling | 8 | — | 21.8 | 67.6 | 3.10× | — | 3.63 |
| Queue pressure | 12 | 4 | 22.8 | 60.2 | 2.64× | 2.75× | 3.31 |

---

# Key Results

The strongest measured result was the uniform four-request workload:

```text
Sequential throughput : 20.7 tok/s
Batched throughput    : 74.3 tok/s

Throughput gain:
3.58×
```

Wall time decreased from:

```text
15,433 ms
```

to:

```text
4,307 ms
```

which is also:

```text
3.58× faster
```

The throughput scaling experiment also showed that batching continues to provide increasing throughput as concurrency rises:

```text
Concurrency = 1 → 19.9 tok/s
Concurrency = 2 → 27.9 tok/s
Concurrency = 4 → 40.9 tok/s
Concurrency = 8 → 67.6 tok/s
```

with the largest measured scaling gain reaching:

```text
3.10×
```

at concurrency 8.

The queue-pressure experiment reached:

```text
60.2 tok/s
```

with:

```text
2.64× token-throughput gain
2.75× wall-time speedup
```

---

# What We Learned

## 1. Batched forward execution reduces repeated model execution overhead

Multiple active requests can share one model forward pass.

Instead of:

```text
A → forward
B → forward
C → forward
D → forward
```

ForgeServe can perform:

```text
forward([A, B, C, D])
```

This produced the strongest gains for uniform workloads where all requests remained active together.

## 2. Average batch size matters

The number of submitted requests is not the same as the number of requests processed together at every step.

Measured average batch sizes:

```text
Uniform N=4       → 4.00
Mixed N=4         → 1.82
Scaling N=8       → 3.63
Queue pressure    → 3.31
```

Higher sustained batch occupancy generally corresponds to stronger batching gains in these experiments.

## 3. Continuous batching is dynamic

Requests leave the batch as soon as they finish.

Waiting requests can then enter:

```text
RUNNING
   ↓
FINISHED
   ↓
free slot + free KV blocks
   ↓
WAITING request admitted
   ↓
RUNNING
```

This allows the system to maintain useful GPU work as request lifetimes differ.

## 4. Throughput and latency are different objectives

The benchmark demonstrates an important serving trade-off.

For the 12-request queue-pressure scenario:

```text
Throughput:
22.8 → 60.2 tok/s
```

but:

```text
Mean latency:
1410.8 → 3290.2 ms
```

Therefore, continuous batching should not be evaluated using throughput alone.

A serving system needs to track both:

```text
system-level efficiency
        +
request-level latency
```

## 5. Maximum batch size is a scheduling constraint

The queue-pressure experiment used:

```text
max_batch_size = 4
```

Even though 12 requests were submitted, only up to four requests could be active in one decode batch.

As requests finished, new requests entered the batch.

This demonstrates why:

```text
number of submitted requests
```

and:

```text
batch size
```

are separate concepts.

---

# Important Benchmark Limitation

The mixed and queue-pressure workloads contain early EOS termination.

For Scenario 2:

```text
Sequential total tokens = 230
Batched total tokens    = 219
```

Therefore, those runs do not perform exactly the same amount of generation work.

The uniform scenario is the cleanest apples-to-apples comparison because every request generated exactly 80 tokens and both implementations produced the same total of:

```text
320 tokens
```

For future benchmarking, controlled termination would make heterogeneous workload comparisons even cleaner.

---

# What Phase 5B Demonstrates

Phase 5B demonstrates that ForgeServe can move from:

```text
multiple requests
        ↓
separate forward pass per request
```

to:

```text
multiple active requests
        ↓
one batched forward pass per decode step
```

The strongest measured result was:

```text
4 requests
80 tokens/request

20.7 tok/s
      ↓
74.3 tok/s

3.58× throughput gain
```

The 8-concurrency scaling experiment reached:

```text
67.6 tok/s
```

with a:

```text
3.10× throughput gain
```

The queue-pressure experiment reached:

```text
60.2 tok/s
```

with:

```text
2.64× token-throughput gain
2.75× wall-time speedup
```

---

# Conclusion

Phase 5B completes the transition from per-request decode execution to batched continuous decoding.

The scheduler maintains a dynamic set of active requests, and the runtime processes those requests together in a single forward pass at each decode step.

The benchmark results show substantial system-level gains:

```text
Uniform 4-request workload:
    3.58× throughput gain

8-request scaling:
    3.10× throughput gain

12-request queue pressure:
    2.64× throughput gain
```

At the same time, individual request latency can increase because requests may wait for admission or share decode steps with other active requests.

The main lesson from Phase 5B is:

> Continuous batching improves GPU utilization and system throughput by combining active requests into shared forward passes, but scheduler design must balance aggregate throughput against individual request latency.


