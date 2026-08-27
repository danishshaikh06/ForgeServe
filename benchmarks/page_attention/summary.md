# Phase 4 Benchmark — Paged KV Cache

## Introduction

Phase 4 introduces a paged KV-cache memory-management system into ForgeServe.

The goal of this benchmark is to validate that ForgeServe can:

* Allocate KV blocks correctly.
* Associate blocks with individual requests.
* Reuse blocks.
* Release blocks after generation.
* Avoid block leaks.
* Maintain a fixed GPU KV-memory pool.

This benchmark also helps us understand an important distinction:

> **Generated tokens across multiple requests are not the same thing as tokens stored simultaneously in memory.**

---

# Benchmark Configuration

| Setting                      | Value                      |
| ---------------------------- | -------------------------- |
| Model                        | Qwen/Qwen2.5-0.5B-Instruct |
| GPU                          | NVIDIA RTX 4070            |
| Block Size                   | 16 tokens                  |
| Total Blocks                 | 256                        |
| Warmup Runs                  | 2                          |
| Benchmark Runs               | 5                          |
| Maximum New Tokens / Request | 128                        |

The BlockManager pre-allocates 256 KV blocks.

The total KV memory reserved for the block pool is:

```text
48.0 MB
```

according to the runtime initialization output.

---

# How Blocks Are Allocated

Each block can hold:

```text
16 tokens
```

A request does not necessarily need exactly one block.

The required number of blocks is approximately:

```text
Number of Blocks
=
ceil(Total Sequence Tokens / Block Size)
```

where:

```text
Total Sequence Tokens
=
Prompt Tokens + Generated Tokens
```

For example, if a request has approximately:

```text
40 prompt tokens
128 generated tokens
```

then:

```text
40 + 128
=
168 total tokens
```

With:

```text
16 tokens / block
```

the request needs:

```text
ceil(168 / 16)
=
ceil(10.5)
=
11 blocks
```

Therefore:

> **11 blocks refers to one request.**

It does not mean that all requests together used only 11 blocks.

---

# Understanding the Eight-Request Scenario

The benchmark contains a scenario with:

```text
8 requests
```

and:

```text
128 generated tokens per request
```

Therefore the benchmark reports:

```text
8 × 128
=
1024 generated tokens
```

This does **not** mean that one request generated 1024 tokens.

Instead:

```text
Request 1 → 128 tokens → 11 blocks
Request 2 → 128 tokens → 11 blocks
Request 3 → 128 tokens → 11 blocks
Request 4 → 128 tokens → 11 blocks
Request 5 → 128 tokens → 11 blocks
Request 6 → 128 tokens → 11 blocks
Request 7 → 128 tokens → 11 blocks
Request 8 → 128 tokens → 11 blocks
```

So:

```text
Total generated tokens
=
8 × 128
=
1024 tokens
```

while:

```text
Blocks used per request
=
11
```

These are two different measurements.

---

# Why Doesn't the Eight-Request Scenario Use 88 Blocks?

The current benchmark processes the requests **sequentially**.

The lifecycle is:

```text
Request 1
    ↓
Allocate blocks
    ↓
Generate 128 tokens
    ↓
Free blocks
    ↓
Request 2
    ↓
Allocate blocks
    ↓
Generate 128 tokens
    ↓
Free blocks
    ↓
...
```

Therefore the block pool does not contain all eight request caches at the same time.

Conceptually:

```text
Time ────────────────────────────────────────>

R1: [ALLOCATE]──[GENERATE]──[FREE]

R2:                        [ALLOCATE]──[GENERATE]──[FREE]

R3:                                             [ALLOCATE]──...
```

At any one point in time, only one request is holding its blocks.

Therefore the pool can return to:

```text
256 free blocks
```

after every request.

## The benchmark output confirms this behavior.

# Total Generated Tokens vs Live Tokens

This is one of the most important concepts in this benchmark.

Suppose:

```text
8 requests
×
128 generated tokens
```

Then:

```text
Total generated tokens = 1024
```

But because the requests run sequentially, approximately only one request's sequence is resident at a time.

For example:

```text
Request 1
≈168 total live tokens
    ↓
free

Request 2
≈168 total live tokens
    ↓
free

...
```

Therefore:

```text
Total tokens generated over the experiment
    ≠
Maximum tokens simultaneously resident in memory
```

This distinction explains why the system can generate 1024 tokens in total without requiring enough memory to hold 1024 tokens at once.

---

# Eight Requests Do Not Mean Eight Concurrent Requests

The name:

```text
eight_requests
```

describes how many requests are executed during the scenario.

It does **not** mean that eight requests are simultaneously active.

The current benchmark is:

```text
Sequential requests
```

not:

```text
Concurrent requests
```

Therefore, the current benchmark should not be used to claim support for concurrent multi-user serving.

---

# Benchmark Results

## Single Request

Generated:

```text
128 tokens
```

Blocks used:

```text
11 blocks
```

Average throughput:

```text
≈25.35 tokens/sec
```

Peak GPU memory:

```text
≈1018 MB
```

Final free blocks:

```text
256
```

Block leak:

```text
False
```

## The output confirms that the allocated blocks were returned to the pool.

## Two Requests

Each request:

```text
128 generated tokens
11 blocks
```

Total generated tokens:

```text
256
```

Average throughput:

```text
≈24.79 tokens/sec
```

Peak GPU memory:

```text
≈1019 MB
```

Final free blocks:

```text
256
```

Block leak:

```text
False
```

## The requests are executed one after another, so the blocks of the first request are released before the second request begins.

## Four Requests

Each request:

```text
128 generated tokens
11 blocks
```

Total generated tokens:

```text
4 × 128
=
512
```

Average throughput:

```text
≈23.49 tokens/sec
```

Peak GPU memory:

```text
≈1018 MB
```

Final free blocks:

```text
256
```

Block leak:

```text
False
```

The benchmark therefore demonstrates repeated allocation and successful reuse of the same block pool.

---

## Eight Requests

Each request:

```text
128 generated tokens
11 blocks
```

Total generated tokens:

```text
8 × 128
=
1024
```

Average throughput:

```text
≈24.61 tokens/sec
```

Peak GPU memory:

```text
≈1018 MB
```

Final free blocks:

```text
256
```

Block leak:

```text
False
```

## The logs show that each request finishes before the next request begins, and the block pool is restored after the scenario.

# Summary Table

| Scenario | Requests | Tokens / Request | Total Generated Tokens | Blocks / Request | Peak Memory | Final Free Blocks |
| -------- | -------: | ---------------: | ---------------------: | ---------------: | ----------: | ----------------: |
| Single   |        1 |              128 |                    128 |               11 |    ~1018 MB |               256 |
| Two      |        2 |              128 |                    256 |               11 |    ~1019 MB |               256 |
| Four     |        4 |              128 |                    512 |               11 |    ~1018 MB |               256 |
| Eight    |        8 |              128 |                   1024 |               11 |    ~1018 MB |               256 |

The important point is:

```text
Blocks / Request = 11
```

while:

```text
Total Generated Tokens
=
Requests × Tokens per Request
```

For eight requests:

```text
8 × 128 = 1024
```

but the requests are sequential, so the 1024 tokens are not simultaneously resident.

---

# What Does This Benchmark Prove?

This benchmark successfully demonstrates:

```text
Request
   ↓
Allocate KV blocks
   ↓
Store KV data
   ↓
Generate
   ↓
Release blocks
   ↓
Return blocks to pool
```

It verifies:

* Correct block allocation.
* Correct request ownership.
* Correct block reuse.
* Correct block release.
* No block leakage.
* Stable preallocated KV memory.

The strongest correctness condition is:

```text
Free blocks before request
=
Free blocks after request
=
256
```

for the tested scenarios.

---

# What This Benchmark Does Not Prove

The benchmark does not yet measure:

* True concurrent requests.
* Multiple simultaneously resident KV caches.
* Continuous batching.
* Scheduler efficiency.
* Maximum concurrent request capacity.
* Memory fragmentation across simultaneously active requests.
* Throughput scaling under concurrent load.

These require a benchmark in which multiple requests remain active at the same time.

---

# The Next Paged KV Experiment

The next experiment will keep request caches resident simultaneously.

For example:

```text
Request 1 → 11 blocks
Request 2 → 11 blocks
Request 3 → 11 blocks
Request 4 → 11 blocks
```

If all four requests remain active:

```text
Total used blocks
=
11 + 11 + 11 + 11
=
44 blocks
```

With a pool of 256 blocks:

```text
Free blocks
=
256 - 44
=
212
```

This is fundamentally different from the current sequential benchmark.

The next benchmark will allow us to measure:

* Blocks used by each request.
* Total blocks used simultaneously.
* Free blocks remaining.
* KV memory currently in use.
* Internal block waste.
* Maximum number of resident requests.

---

# Conclusion

The current Phase 4 benchmark validates the basic correctness of ForgeServe's paged KV-cache allocator.

The most important results are:

```text
256 preallocated blocks
48 MB KV block pool
11 blocks per tested request
No block leaks
Correct block release and reuse
```

The eight-request scenario generated:

```text
1024 total tokens
```

but this does not represent 1024 simultaneously stored tokens. The eight requests were processed sequentially, so each request used its approximately 11 blocks and released them before the next request began.

Therefore:

> **Total generated tokens measures the amount of work performed during the benchmark, while blocks used measures memory belonging to a request at a particular point in time.**

