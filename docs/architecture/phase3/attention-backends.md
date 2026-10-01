# Attention Backends

## Overview

ForgeServe supports configurable attention implementations so the inference engine can compare a reference attention path against an optimized PyTorch path.

The current comparison is:

```text
EAGER
  ↓
Standard attention path

SDPA
  ↓
Scaled Dot Product Attention
  ↓
PyTorch backend selection
  ↓
Optimized CUDA attention when eligible
```

ForgeServe describes the benchmark as **EAGER vs SDPA**, not simply **EAGER vs FlashAttention**. Selecting `attn_implementation="sdpa"` confirms the SDPA path, but does not independently prove that the FlashAttention kernel executed for every operation.

## Why Attention Backends Matter

Attention is a major component of transformer inference. Making the backend configurable lets ForgeServe compare the same model and workload under different execution paths.

## EAGER

EAGER is the baseline/reference implementation. It provides a straightforward point of comparison for optimization experiments.

## SDPA

SDPA uses PyTorch's scaled dot product attention implementation and can select an optimized backend internally when the operation and hardware satisfy its requirements.

Therefore:

```text
SDPA
  ↓
backend selection
  ↓
optimized CUDA implementation when eligible
```

is the correct mental model.

## Phase 3 Results

| Workload | EAGER TPOT | SDPA TPOT | Speedup |
|---|---:|---:|---:|
| 37-token prompt / 100 requested | 31.7 ms | 31.9 ms | 0.99× |
| 196-token prompt / 200 tokens | 36.4 ms | 33.0 ms | 1.10× |
| 364-token prompt / 50 tokens | 34.1 ms | 33.0 ms | 1.03× |
| 364-token prompt / 300 tokens | 36.5 ms | 33.0 ms | 1.10× |

The strongest measured improvement was **1.10× lower TPOT** on the medium and long-generation workloads.

Peak process GPU memory did not materially change in these measurements.

## Engineering Lesson

Optimization should be validated experimentally. The measured benefit depends on workload size and generation length, so ForgeServe does not assume that an optimized attention path is always faster.
