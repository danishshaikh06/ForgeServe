# Adaptive Batching

## Why Fixed Batch Size Is Only a Baseline

A fixed configuration might say:

```python
max_batch_size = 4
```

This is easy to implement, but it assumes that four requests are always the right amount of work.

That is not necessarily true.

## Compute Capacity

A GPU has finite compute and memory resources.

A larger batch generally increases the amount of work in one forward pass.

At some point, increasing the batch further may stop producing proportional throughput gains.

Therefore we want:

\[
\boxed{
\text{largest useful batch}
}
\]

not merely:

\[
\boxed{
\text{largest possible batch}
}
\]

## Memory Capacity

KV cache growth is another constraint.

For request \(i\):

\[
B_i
=
\left\lceil
\frac{T_i}{b}
\right\rceil
\]

Total used blocks:

\[
B_{used}
=
\sum_i B_i
\]

The scheduler must keep:

\[
B_{used}\leq B_{total}
\]

## Resource-Aware Admission

A future scheduler can think in terms of a budget:

\[
C_{current}+C_{new}\leq C_{max}
\]

and:

\[
M_{current}+M_{new}\leq M_{max}
\]

where:

```text
C → estimated compute cost
M → memory cost
```

A request is admitted only when both constraints are safe.

## Why Request Count Alone Is Not Enough

Consider:

```text
A → 100-token context
B → 100-token context
C → 4000-token context
D → 4000-token context
```

All four are requests, but they do not necessarily represent the same memory or compute cost.

Therefore:

```text
batch size = 4
```

does not mean:

```text
cost = 4 identical units
```

## Benchmark-Driven Adaptation

ForgeServe can use benchmark data to understand the relationship:

\[
batch\ size
\rightarrow
execution\ time
\]

and:

\[
batch\ size
\rightarrow
throughput
\]

For example, measured throughput can be evaluated at batch sizes 1, 2, 4, and 8.

The scheduler can then learn where additional batching remains useful.

## Important Caveat

A benchmark on one model and GPU does not automatically generalize to every model and GPU.

The useful batch size depends on:

```text
model architecture
model size
KV dimensions
sequence lengths
dtype
GPU
attention backend
memory capacity
workload
```

## Future ForgeServe Direction

A future adaptive scheduler could choose batch size based on:

```text
current active requests
available KV blocks
sequence lengths
configured batch cap
measured GPU behavior
```

That would turn the current fixed scheduler into a resource-aware scheduler.

## Mental Model

Start with:

```text
fixed batch size
```

then evolve toward:

```text
adaptive batch size
```

then, if needed:

```text
predictive resource-aware scheduling
```

The fixed scheduler is the baseline. Adaptive scheduling is an optimization built on top of a correct baseline.
