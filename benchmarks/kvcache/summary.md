# Phase 2 Benchmark — KV Cache

## Introduction

Phase 2 compares two ForgeServe generation implementations:

* **Naive generation**
* **KV Cache generation**

The purpose of the benchmark is to measure whether reusing previously computed Key and Value tensors improves autoregressive decoding.

The benchmark is designed as a performance comparison, but it also helps us understand the cost and benefit of KV Cache.

---

# What Is Being Compared?

### Naive Generation

The naive implementation repeatedly processes the complete sequence.

```text
Prompt
  ↓
Forward entire sequence
  ↓
Generate token
  ↓
Append token
  ↓
Forward entire sequence again
  ↓
Generate next token
```

As the sequence grows, more previously processed tokens are recomputed.

---

### KV Cache Generation

The KV Cache implementation processes the prompt first and stores the Key and Value tensors.

```text
Prompt
  ↓
Prefill
  ↓
Store K/V
  ↓
Decode one token
  ↓
Reuse previous K/V
  ↓
Decode next token
```

This reduces redundant computation during decoding.

---

# Benchmark Configuration

| Setting              | Value                      |
| -------------------- | -------------------------- |
| Model                | Qwen/Qwen2.5-0.5B-Instruct |
| GPU                  | NVIDIA RTX 4070            |
| Warmup Runs          | 2                          |
| Benchmark Runs       | 5                          |
| Sampling             | Greedy                     |
| Naive Implementation | Yes                        |
| KV Cache             | Yes                        |

Four workloads were defined using two prompts and two requested generation lengths:

```text
Prompt A + 100 requested tokens
Prompt A + 300 requested tokens
Prompt B + 100 requested tokens
Prompt B + 300 requested tokens
```

---

# Important Benchmark Observation

The requested maximum generation lengths were not reached.

The model stopped early because it generated an End-of-Sequence (EOS) token.

The actual benchmark results were:

| Scenario                 | Actual Naive Tokens | Actual KV Cache Tokens |
| ------------------------ | ------------------: | ---------------------: |
| Prompt A / 100 requested |                  21 |                     21 |
| Prompt A / 300 requested |                  21 |                     21 |
| Prompt B / 100 requested |                  22 |                     22 |
| Prompt B / 300 requested |                  22 |                     22 |

Therefore, the benchmark should **not** be interpreted as a 100-token vs 300-token generation comparison.

It is an early-terminating generation comparison.

Future benchmarks should use controlled generation lengths when we want to study how KV Cache scales with decode length.

---

# Results

## Scenario 1

Actual generated tokens:

```text
21
```

| Metric      |      Naive |   KV Cache |
| ----------- | ---------: | ---------: |
| TTFT        |    37.8 ms |    42.6 ms |
| TPOT        |    40.3 ms |    34.5 ms |
| Total Time  |   843.7 ms |   733.0 ms |
| Throughput  | 25.0 tok/s | 28.7 tok/s |
| Peak Memory |   990.7 MB |   978.3 MB |

Results:

```text
Total speedup = 1.15×
TPOT speedup  = 1.17×
Memory delta  = -12.4 MB (memory delta = KV Cache memory − Naive memory)
```

---

## Scenario 2

Actual generated tokens:

```text
21
```

| Metric      |      Naive |   KV Cache |
| ----------- | ---------: | ---------: |
| TTFT        |    39.1 ms |    44.9 ms |
| TPOT        |    41.3 ms |    39.2 ms |
| Total Time  |   864.7 ms |   828.6 ms |
| Throughput  | 24.4 tok/s | 25.6 tok/s |
| Peak Memory |   990.7 MB |   978.3 MB |

Results:

```text
Total speedup = 1.04×
TPOT speedup  = 1.05×
Memory delta  = -12.4 MB
```

---

## Scenario 3

Actual generated tokens:

```text
22
```

| Metric      |      Naive |   KV Cache |
| ----------- | ---------: | ---------: |
| TTFT        |    38.7 ms |    38.6 ms |
| TPOT        |    37.9 ms |    32.9 ms |
| Total Time  |   835.7 ms |   729.9 ms |
| Throughput  | 26.3 tok/s | 30.2 tok/s |
| Peak Memory |  1000.0 MB |   987.1 MB |

Results:

```text
Total speedup = 1.14×
TPOT speedup  = 1.15×
Memory delta  = -12.9 MB
```

---

## Scenario 4

Actual generated tokens:

```text
22
```

| Metric      |      Naive |   KV Cache |
| ----------- | ---------: | ---------: |
| TTFT        |    38.4 ms |    38.7 ms |
| TPOT        |    38.0 ms |    33.4 ms |
| Total Time  |   837.3 ms |   739.3 ms |
| Throughput  | 26.3 tok/s | 29.8 tok/s |
| Peak Memory |  1000.0 MB |   987.1 MB |

Results:

```text
Total speedup = 1.13×
TPOT speedup  = 1.14×
Memory delta  = -12.9 MB
```

---

# What Does the Negative Memory Overhead Mean?

The benchmark calculates:

```text
KV Cache peak memory - Naive peak memory
```

For example:

```text
978.3 MB - 990.7 MB
= -12.4 MB
```

Therefore:

```text
Memory delta = -12.4 MB
```

means:

> During that benchmark run, the KV Cache implementation had approximately 12.4 MB lower measured peak GPU memory allocation than the naive implementation.

It does **not** mean that KV Cache itself requires negative memory.

---

# Why Can KV Cache Have Lower Peak Memory?

KV Cache stores Key and Value tensors, so the cache itself consumes memory.

However, the naive implementation repeatedly processes the growing sequence and can create larger temporary attention tensors during those forward passes.

The measured quantity is:

```text
torch.cuda.max_memory_allocated()
```

which represents peak GPU memory allocated during the measured execution.

It does not isolate the KV Cache tensor itself.

Therefore, the lower measured peak can result from the KV Cache implementation avoiding some of the temporary allocations created by repeatedly processing the full sequence.

The safest interpretation is:

> **The KV Cache implementation produced lower peak GPU memory allocation in this benchmark. This measurement represents total measured GPU allocation during generation and should not be interpreted as the raw memory footprint of the KV Cache alone.**

---

# What We Learned

The benchmark shows that KV Cache provided a measurable improvement on this workload.

Across the four scenarios:

```text
TPOT improvement:
approximately 5% to 17%

Total generation speedup:
approximately 1.04× to 1.15×

Peak memory:
approximately 12–13 MB lower in the KV Cache runs
```

The improvement is not identical across scenarios.

This is expected because inference performance depends on:

* Prompt length
* Actual generated length
* Model size
* GPU
* Kernel implementation
* Memory behavior

---

# Benchmark Limitation

The most important limitation is early EOS termination.

The benchmark requested up to 100 or 300 new tokens, but the model stopped after 21–22 tokens.

Therefore, this benchmark cannot demonstrate how KV Cache scales with long generation lengths.

A future benchmark should force controlled generation lengths or use a configuration that prevents early EOS termination when measuring scaling behavior.

---

# Conclusion

Phase 2 successfully demonstrated that KV Cache can improve autoregressive decoding performance in ForgeServe.

The measured results show:

```text
KV Cache
   ↓
Lower TPOT
   ↓
Higher throughput
   ↓
Lower measured peak GPU allocation
```

The most important achievement of Phase 2 is not the exact speedup number.

It is that ForgeServe now has a working KV Cache implementation that can be measured against a naive reference implementation.

This provides the performance baseline for future inference optimizations.
