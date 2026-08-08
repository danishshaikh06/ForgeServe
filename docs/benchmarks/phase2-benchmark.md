# Phase 2 Benchmark — KV Cache

## Introduction

Phase 2 introduces **KV Cache** into ForgeServe.

The purpose of this benchmark is to compare the original Phase 1 generation approach with the KV Cache implementation.

The main question is:

> Does reusing previously computed Key and Value tensors improve autoregressive generation performance?

The benchmark compares two implementations:

* **Naive** — processes the complete sequence during every generation step.
* **KV Cache** — reuses previously computed Key and Value tensors during decoding.

---

# Benchmark Configuration

| Setting        | Value                               |
| -------------- | ----------------------------------- |
| Model          | Qwen/Qwen2.5-0.5B-Instruct          |
| Warmup Runs    | 2                                   |
| Benchmark Runs | 5                                   |
| GPU            | NVIDIA RTX 4070                     |
| KV Cache       | Enabled for KV Cache implementation |
| Sampling       | Greedy Decoding                     |

Four different scenarios were tested.

| Scenario | Prompt Tokens | Generated Tokens |
| -------- | ------------: | ---------------: |
| 1        |             4 |               50 |
| 2        |             4 |              200 |
| 3        |            17 |               50 |
| 4        |            17 |              200 |

---

# Why Use Multiple Scenarios?

Generation performance depends on both:

* Prompt length
* Number of generated tokens

A short prompt with a short response behaves differently from a long prompt with a long response.

For this reason, ForgeServe tests multiple combinations.

```text
                    Generated Tokens

                 50              200

Prompt 4       Scenario 1      Scenario 2

Prompt 17      Scenario 3      Scenario 4
```

This gives us a better understanding of how KV Cache behaves under different workloads.

---

# Benchmark Methodology

Each scenario was executed using the following process:

```text
Initialize Runtime
       │
       ▼
Load Model
       │
       ▼
Warmup × 2
       │
       ▼
Benchmark × 5
       │
       ▼
Calculate Metrics
```

Warmup runs are performed before measurement to reduce the effect of initialization and GPU startup overhead.

The final measurements are taken from the five benchmark runs.

---

# Results

The benchmark produced the following aggregate results:

| Metric          |       Result |
| --------------- | -----------: |
| Total Speedup   |    **1.13×** |
| TPOT Speedup    |    **1.14×** |
| Memory Overhead | **-12.9 MB** |

---

# Total Speedup

The KV Cache implementation achieved an overall speedup of:

```text
1.13×
```

This means the KV Cache implementation was approximately **13% faster** than the naive implementation for the benchmark workload.

The improvement comes from avoiding repeated computation of previously processed Key and Value tensors.

---

# TPOT Speedup

TPOT means:

> **Time Per Output Token**

It measures how much time is required to generate each output token during decoding.

The benchmark measured:

```text
TPOT Speedup = 1.14×
```

This indicates approximately a **14% improvement** in token generation efficiency.

TPOT is particularly important for autoregressive generation because users experience generation as a sequence of output tokens.

---

# Memory Result

The benchmark reported:

```text
Memory Overhead = -12.9 MB
```

The negative value means that the KV Cache implementation used approximately **12.9 MB less measured memory than the naive implementation under this benchmark setup**.

This result should not be interpreted as:

> "KV Cache always uses less memory."

In general, KV Cache requires memory to store Key and Value tensors.

The measured memory difference depends on exactly how memory was measured, when memory was sampled, model allocation behavior, and GPU allocator behavior.

Therefore, this result should be treated as a **measurement of this benchmark**, not as a general property of KV Cache.

---

# Phase 1 vs Phase 2

The main architectural difference is:

### Phase 1 — Naive

```text
Prompt
  │
  ▼
Forward entire sequence
  │
  ▼
Token
  │
  ▼
Append
  │
  ▼
Forward entire sequence again
  │
  ▼
Token
  │
  ▼
Repeat
```

Every iteration processes the complete sequence again.

---

### Phase 2 — KV Cache

```text
Prompt
  │
  ▼
Prefill
  │
  ▼
Create KV Cache
  │
  ▼
Decode one token
  │
  ▼
Update KV Cache
  │
  ▼
Decode next token
  │
  ▼
Repeat
```

Previously computed Key and Value tensors are reused.

---

# Why Is the Improvement Only 1.13×?

It is important not to expect a huge speedup from KV Cache in every situation.

KV Cache primarily reduces the redundant computation associated with processing previous tokens during decoding.

However, generation still requires:

* Transformer computation for the new token.
* Attention computation.
* Matrix multiplications.
* Sampling.
* Memory operations.
* Cache management.
* Python/framework overhead.

Additionally, the model used in this benchmark is relatively small:

```text
Qwen2.5-0.5B-Instruct
```

and the benchmark is running on a single RTX 4070.

Therefore, the measured improvement of **1.13×** is a useful real-world result rather than an indication that the implementation failed.

---

# What This Benchmark Proves

The benchmark demonstrates that ForgeServe's KV Cache implementation is functional and provides a measurable performance improvement over the naive implementation.

The important result is:

```text
Naive
  │
  │
  ├── repeated computation
  │
  ▼
KV Cache
  │
  │
  ├── reuse previous K/V
  │
  ▼
1.13× overall speedup
```

The benchmark therefore establishes Phase 2 as a new performance baseline.

---

# Benchmark Limitations

The current benchmark has several limitations.

### 1. Aggregate Results

The current benchmark output reports aggregate speedup rather than the individual latency and throughput values for each scenario.

Future benchmark versions should record:

* Mean latency
* Median latency
* Standard deviation
* Tokens per second
* TPOT
* Time to First Token (TTFT)
* Peak GPU memory

for every scenario.

---

### 2. Limited Workload

Only four prompt/generation combinations were tested.

Future benchmarks should include larger sequences and more workloads.

---

### 3. Single GPU

The benchmark uses a single RTX 4070.

Distributed inference is not part of Phase 2.

---

# Future Benchmark Format

As ForgeServe becomes more advanced, benchmark results should eventually look like:

| Phase   | Optimization   | Prompt | Generated | TTFT | TPOT | Tokens/sec |
| ------- | -------------- | -----: | --------: | ---: | ---: | ---------: |
| Phase 1 | Naive          |      4 |        50 |  TBD |  TBD |        TBD |
| Phase 2 | KV Cache       |      4 |        50 |  TBD |  TBD |        TBD |
| Phase 3 | FlashAttention |      4 |        50 |  TBD |  TBD |        TBD |
| Phase 4 | PagedAttention |      4 |        50 |  TBD |  TBD |        TBD |

This will allow ForgeServe to measure the effect of each optimization independently.

---

# Conclusion

Phase 2 successfully introduces KV Cache into ForgeServe.

Compared with the naive implementation, the benchmark measured:

```text
Overall Speedup     → 1.13×
TPOT Speedup        → 1.14×
Memory Difference   → -12.9 MB
```

The most important achievement of Phase 2 is not simply the performance improvement.

It is that ForgeServe now has a generation pipeline capable of **reusing previously computed attention state during autoregressive decoding**.

This provides the foundation for the next stages of the project.

The next major optimization is **FlashAttention**, which addresses a different part of the inference problem: the efficiency of the attention computation itself.
