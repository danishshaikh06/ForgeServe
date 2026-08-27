# Understanding Negative Memory Overhead

The Phase 2 benchmark reports a metric called **Memory Overhead**.

It is calculated as:

```text
Memory Overhead
=
KV Cache Peak Memory
-
Naive Peak Memory
```

For example, suppose the benchmark reports:

```text
Naive Peak Memory     = 1154.9 MB
KV Cache Peak Memory  = 986.3 MB
```

Then:

```text
Memory Overhead
=
986.3 - 1154.9

=
-168.6 MB
```

The negative sign does **not** mean that the KV Cache uses negative memory.

It means:

> The KV Cache execution reached a lower peak GPU allocation than the Naive execution.

---

# Why Can the Value Be Negative?

The important point is that the benchmark is measuring:

```text
Peak GPU Memory Allocated During the Run
```

It is **not** measuring only the memory occupied by the KV Cache.

The measured memory can include:

```text
Model weights
+
Attention temporary tensors
+
Intermediate tensors
+
KV Cache
+
Other runtime allocations
```

Therefore, the comparison is:

```text
                         Peak GPU allocation

Naive
┌─────────────────────────────────────────────┐
│ Model                                       │
│ Temporary tensors                           │
│ Attention computation                       │
│ Other allocations                           │
└─────────────────────────────────────────────┘
                 ↑
             Peak = N


KV Cache
┌─────────────────────────────────────────────┐
│ Model                                       │
│ KV Cache                                    │
│ Smaller temporary allocations               │
│ Other allocations                           │
└─────────────────────────────────────────────┘
                 ↑
             Peak = K
```

If:

```text
K < N
```

then:

```text
K - N < 0
```

and the reported memory overhead becomes negative.

---

# Mathematical Interpretation

Let:

```text
M_naive = peak GPU memory allocated by the naive run

M_kv = peak GPU memory allocated by the KV-cache run
```

The benchmark calculates:

```text
ΔM = M_kv - M_naive
```

There are three possible outcomes.

### Negative

```text
ΔM < 0
```

This means:

```text
M_kv < M_naive
```

The KV-cache run reached a lower peak allocation.

---

### Zero

```text
ΔM = 0
```

This means:

```text
M_kv = M_naive
```

Both executions reached approximately the same peak allocation.

---

### Positive

```text
ΔM > 0
```

This means:

```text
M_kv > M_naive
```

The KV-cache execution reached a higher peak allocation.

---

# Why Does the Naive Implementation Often Need More Temporary Memory?

The naive implementation repeatedly processes the entire growing sequence.

If the sequence lengths during generation are:

```text
S, S+1, S+2, ..., S+N
```

then every generation step performs attention over an increasingly large sequence.

Conceptually:

```text
Attention(S)
Attention(S+1)
Attention(S+2)
...
Attention(S+N)
```

This can create larger intermediate tensors as the sequence grows.

The KV-cache implementation instead processes:

```text
Prefill(S)
```

followed by decode steps that process only the newly generated token while reusing cached Key and Value tensors.

Therefore, the transient memory requirements of the decode steps can remain much more stable.

---

# Why Can KV Cache Stay Approximately Flat?

Suppose the total peak GPU allocation observed by the benchmark is:

```text
986 MB
```

for both:

```text
100 generated tokens
```

and:

```text
250 generated tokens
```

This does **not** mean the KV Cache stayed the same size.

The KV Cache itself grows with sequence length.

The important distinction is:

```text
KV Cache Size
        ↑
        grows

Peak Total GPU Allocation
        ↑
        may remain approximately unchanged
```

The additional KV memory may fit within memory that is already part of the process's allocated memory footprint, while other temporary allocations determine the overall peak.

Therefore:

```text
Peak GPU Memory ≠ KV Cache Size
```

---

# Example

Imagine:

```text
Model memory = 800 MB
```

At 100 generated tokens:

```text
KV Cache = 20 MB
Temporary memory = 100 MB

Total peak ≈ 920 MB
```

At 250 generated tokens:

```text
KV Cache = 50 MB
Temporary memory = 70 MB

Total peak ≈ 920 MB
```

The KV Cache became larger:

```text
20 MB → 50 MB
```

but the total peak stayed approximately the same because the temporary memory requirements changed.

This is why a flat `max_memory_allocated()` value does not imply a flat KV-cache size.

---

# What the Negative Number Actually Tells Us

Suppose:

```text
Naive = 1154.9 MB
KV Cache = 986.3 MB
```

Then:

```text
ΔM = 986.3 - 1154.9
   = -168.6 MB
```

The correct interpretation is:

> During this benchmark run, the KV-cache execution reached a peak GPU allocation approximately 168.6 MB lower than the naive execution.

It does **not** mean:

```text
KV Cache requires 168.6 MB less memory
```

and it does **not** mean:

```text
KV Cache itself consumes -168.6 MB
```

The metric is a **difference between two total peak-memory measurements**.

---

# Important Note About the GPU Allocator

PyTorch uses a caching memory allocator.

Memory behavior therefore involves more than simply:

```text
allocate → use → release
```

Some memory can remain managed by the allocator so that future operations can reuse it.

Because of this, process-level memory measurements can hide the exact size of individual objects such as the KV Cache.

For this reason, `max_memory_allocated()` is best interpreted as a **peak execution-memory metric**, not as a direct measurement of KV-cache capacity.

---

# The Key Formula to Remember

```text
Memory Overhead
=
Peak Memory(KV Cache)
-
Peak Memory(Naive)
```

Therefore:

```text
Negative → KV run had lower peak allocation

Zero     → Both reached approximately the same peak

Positive → KV run had higher peak allocation
```

The sign simply tells us **which execution reached the larger peak**.

It does not describe the size of the KV Cache itself.

---

# Why This Matters for ForgeServe

This distinction becomes especially important as ForgeServe evolves.

Later phases will measure memory differently:

```text
Phase 2
Peak execution memory

        ↓

Paged KV Cache
Active KV blocks

        ↓

Continuous batching
Per-request KV memory

        ↓

Production serving
GPU memory utilization
```

As the system becomes more sophisticated, we will measure the specific resource we are interested in instead of relying only on total process memory.
