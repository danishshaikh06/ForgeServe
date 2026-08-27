# Phase 3 Benchmark — SDPA and FlashAttention

## Introduction

Phase 3 investigates optimized attention execution in ForgeServe.

The baseline uses the Hugging Face **EAGER** attention implementation.

The optimized implementation uses **PyTorch SDPA (Scaled Dot Product Attention)**.

SDPA can dispatch to optimized CUDA attention implementations, including a FlashAttention backend when the operation and hardware satisfy the backend requirements.

For this reason, the current benchmark should be described as:

> **EAGER vs SDPA**

rather than simply:

> **EAGER vs FlashAttention**

The benchmark measures whether the optimized SDPA path improves attention-related inference performance.

---

# Benchmark Configuration

| Setting        | Value                 |
| -------------- | --------------------- |
| Model          | Qwen2.5-0.5B-Instruct |
| GPU            | NVIDIA RTX 4070       |
| Warmup Runs    | 2                     |
| Benchmark Runs | 5                     |
| Dtype          | bfloat16              |
| Baseline       | EAGER                 |
| Optimized Path | SDPA                  |
| Sampling       | Greedy                |

---

# Attention Implementations

## EAGER

The baseline attention implementation used by the model.

```text
EAGER
  ↓
Standard attention computation
```

---

## SDPA

The model uses:

```text
torch.nn.functional.scaled_dot_product_attention
```

PyTorch can select an optimized attention backend internally when the inputs satisfy the relevant requirements.

Therefore:

```text
SDPA
  ↓
Backend selection
  ↓
Optimized CUDA attention when eligible
```

The current benchmark confirms that SDPA was used, but it does not independently prove that the FlashAttention kernel was selected for every measured operation.

---

# Benchmark Results

## Scenario 1 — Short Prompt / 100 Requested Tokens

Actual prompt length:

```text
37 tokens
```

Results:

| Metric      |      EAGER |       SDPA |
| ----------- | ---------: | ---------: |
| TTFT        |    37.3 ms |    36.8 ms |
| TPOT        |    31.7 ms |    31.9 ms |
| Total Time  |  3172.8 ms |  3191.7 ms |
| Throughput  | 31.5 tok/s | 31.3 tok/s |
| Peak Memory |   980.7 MB |   981.8 MB |

Result:

```text
Total speedup = 0.99×
TPOT speedup  = 0.99×
```

There was effectively no performance improvement for this short workload.

---

## Scenario 2 — Medium Prompt / 200 Tokens

Actual prompt length:

```text
196 tokens
```

Results:

| Metric      |      EAGER |       SDPA |
| ----------- | ---------: | ---------: |
| TTFT        |    43.5 ms |    49.0 ms |
| TPOT        |    36.4 ms |    33.0 ms |
| Total Time  |  7296.6 ms |  6625.1 ms |
| Throughput  | 27.4 tok/s | 30.2 tok/s |
| Peak Memory |  1077.0 MB |  1077.0 MB |

Result:

```text
Total speedup = 1.10×
TPOT speedup  = 1.10×
```

SDPA provided a measurable performance improvement.

---

## Scenario 3 — Long Prompt / 50 Tokens

Actual prompt length:

```text
364 tokens
```

Results:

| Metric      |      EAGER |       SDPA |
| ----------- | ---------: | ---------: |
| TTFT        |    46.8 ms |    44.0 ms |
| TPOT        |    34.1 ms |    33.0 ms |
| Total Time  |  1715.4 ms |  1660.5 ms |
| Throughput  | 29.2 tok/s | 30.1 tok/s |
| Peak Memory |  1179.6 MB |  1179.6 MB |

Result:

```text
Total speedup = 1.03×
TPOT speedup  = 1.03×
```

Only a small performance improvement was observed.

---

## Scenario 4 — Long Prompt / 300 Tokens

Actual prompt length:

```text
364 tokens
```

Results:

| Metric      |      EAGER |       SDPA |
| ----------- | ---------: | ---------: |
| TTFT        |    51.8 ms |    45.4 ms |
| TPOT        |    36.5 ms |    33.0 ms |
| Total Time  | 10958.7 ms |  9926.1 ms |
| Throughput  | 27.4 tok/s | 30.2 tok/s |
| Peak Memory |  1179.6 MB |  1179.6 MB |

Result:

```text
Total speedup = 1.10×
TPOT speedup  = 1.10×
```

The longer workload shows a clear benefit from the SDPA path.

---

# Overall Observation

The benchmark shows that SDPA does not provide the same benefit for every workload.

Observed TPOT results:

```text
Short   → 0.99×
Medium  → 1.10×
Long    → 1.03×
Long    → 1.10×
```

The strongest improvements occurred for the medium and long-generation workloads.

---

# Memory Results

Peak GPU memory was effectively unchanged:

```text
Short:
980.7 → 981.8 MB

Medium:
1077.0 → 1077.0 MB

Long:
1179.6 → 1179.6 MB
```

Therefore, this benchmark does **not** demonstrate a measurable total-process memory reduction from SDPA.

This does not mean that optimized attention has no memory benefit.

The benchmark measures overall peak GPU allocation, which also includes:

* Model weights
* Other runtime allocations
* Temporary tensors
* Allocator behavior

It therefore does not isolate attention's intermediate memory usage.

---

# Important Interpretation

The current implementation uses:

```text
attn_implementation = "sdpa"
```

This means ForgeServe is using PyTorch's SDPA path.

Our current logging reports that FlashAttention conditions are eligible.

However:

> **Eligibility is not proof of actual FlashAttention kernel execution.**

PyTorch SDPA can select different internal backends depending on the operation and hardware.

Therefore, the current benchmark should be interpreted as:

> **EAGER vs optimized SDPA execution**

rather than as a definitive:

> **EAGER vs FlashAttention kernel**

---

# What We Learned

The benchmark demonstrates three important points.

### 1. Optimized attention can improve inference performance

The strongest measured result was approximately:

```text
1.10× TPOT speedup
```

for the medium and long-generation workloads.

---

### 2. Workload size matters

Very short workloads showed almost no benefit.

This is expected because the overhead of the overall model execution can dominate when the attention workload is small.

---

### 3. Total GPU memory is not enough to evaluate attention optimization

Peak process memory remained almost unchanged.

A more specialized benchmark is required to isolate:

* Attention temporary memory
* Attention execution time
* GPU kernel choice
* Memory traffic

---

# Next Verification Step

Before calling this a definitive FlashAttention benchmark, ForgeServe should verify which SDPA backend is actually executing.

A useful experiment is to compare:

```text
EAGER
   ↓
SDPA automatic backend
   ↓
SDPA forced FlashAttention backend
```

This will allow us to determine whether the automatic SDPA path is actually selecting the FlashAttention backend for the measured workloads.

---

# Conclusion

Phase 3 successfully integrates PyTorch SDPA into ForgeServe and demonstrates measurable performance improvements for several workloads.

The strongest observed result was approximately:

```text
10% lower TPOT
```

However, the current benchmark does not independently prove that FlashAttention itself executed for every measurement.

The next step is therefore **backend verification**, followed by a controlled comparison of EAGER, automatic SDPA, and explicitly selected FlashAttention execution.
