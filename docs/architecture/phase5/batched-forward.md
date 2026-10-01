# Batched Forward Execution

## Overview

Phase 5B changes decode execution from a separate forward pass per request to a shared forward pass for all active requests in the current batch.

Before:

```text
forward(A)
forward(B)
forward(C)
forward(D)
```

After:

```text
forward([A, B, C, D])
```

The goal is one model invocation per decode step for the active batch.

## Why This Helps

The model weights are loaded into GPU memory during model initialization and remain resident. Separate request execution still causes repeated forward computation and separate GPU work.

For a linear layer:

\[
Y=XW^T
\]

separate requests perform multiple operations:

\[
Y_1=X_1W^T,\quad Y_2=X_2W^T,\quad \ldots
\]

Batching combines the inputs:

\[
X=\begin{bmatrix}X_1\\X_2\\\vdots\\X_N\end{bmatrix}
\]

and performs the operation as one larger matrix computation:

\[
\boxed{Y=XW^T}
\]

## Batched Decode Input

Each active request contributes one current token.

For `N` active requests:

```text
batch_tokens.shape = (N, 1)
```

Example:

```text
A → 1532
B → 291
C → 8912
D → 42
```

becomes:

```text
[
  [1532],
  [291],
  [8912],
  [42],
]
```

## Output Mapping

The model returns logits for every request row:

```text
logits.shape = (N, 1, vocab_size)
```

Taking the last position gives:

```text
(N, vocab_size)
```

The batch ordering must remain stable:

```text
row 0 → Request A
row 1 → Request B
row 2 → Request C
row 3 → Request D
```

Each row is sampled independently and mapped back to the corresponding `RequestState`.

## Variable Sequence Lengths

Requests can have different sequence lengths. A dense batched representation needs a common sequence dimension, usually based on:

\[
L_{max}=\max(T_1,\ldots,T_N)
\]

With left padding:

```text
A → [PAD ... PAD A A A]
B → [B B B B B B B B]
```

The attention mask identifies real positions and padding.

## Logical Position vs Physical Index

If a request has 12 real tokens, its next token has logical position 12 because positions are zero-based.

After left padding, the same new token can occupy the final physical position of the dense padded representation.

Therefore:

```text
logical position != physical padded index
```

This distinction is important for positional information and KV extraction.

## Paged KV to Batched Cache

Each request retains its own block table:

```text
A → [B7, B3, B1]
B → [B5, B2]
```

The current model interface needs one batched cache representation. The practical bridge is:

```text
per-request paged KV
      ↓
gather to contiguous KV
      ↓
left-pad to common length
      ↓
combine into batch
      ↓
one model forward
```

A future kernel-level paged-attention implementation could remove the intermediate gather.

## Allocate Before Forward

Before the batched decode step, every request must have storage available for the new KV values.

Correct ordering:

```text
check block requirement
        ↓
allocate required blocks
        ↓
build batch
        ↓
one forward pass
        ↓
extract new K/V
        ↓
write K/V into request blocks
```

The rule is:

> Never perform GPU work that cannot be stored safely.

If a request crosses a block boundary and no block is allocated, the model can produce new K/V values but the runtime has nowhere to store them.

## New KV Extraction

With right-aligned decode, the newest token occupies the final sequence position. For batch index `i`:

```python
new_k = key_cache[layer_idx][i, :, -1, :]
new_v = value_cache[layer_idx][i, :, -1, :]
```

The `-1` is the physical final index, not the token's logical position.

## Phase 5B Results

### Uniform 4-request workload

Each request generated 80 tokens.

```text
Sequential throughput: 20.7 tok/s
Batched throughput:    74.3 tok/s
```

Measured gain:

```text
3.58×
```

Wall time:

```text
15,433 ms → 4,307 ms
```

Average batch size:

```text
4.00 requests/step
```

### Mixed 4-request workload

```text
Sequential throughput: 23.5 tok/s
Batched throughput:    40.5 tok/s
```

Token-throughput gain:

```text
1.73×
```

Average batch size:

```text
1.82
```

Shorter requests ended earlier, reducing effective batch size.

### Concurrency scaling

Measured batched throughput:

```text
Concurrency 1 → 19.9 tok/s
Concurrency 2 → 27.9 tok/s
Concurrency 4 → 40.9 tok/s
Concurrency 8 → 67.6 tok/s
```

At concurrency 8, the measured gain was:

```text
3.10×
```

with an average batch size of 3.63 requests/step.

### Queue pressure

The benchmark used 12 requests with:

```text
max_batch_size = 4
```

Measured:

```text
Sequential: 22.8 tok/s
Batched:    60.2 tok/s
```

Token-throughput gain:

```text
2.64×
```

Wall-time speedup:

```text
2.75×
```

Average batch size:

```text
3.31
```

However:

```text
Sequential mean latency: 1410.8 ms
Batched mean latency:    3290.2 ms
Mean queue wait:         1663.3 ms
```

This demonstrates the throughput/latency trade-off in queue-heavy workloads.

## Current Scope

Phase 5B establishes batched forward execution. The current implementation may still gather paged KV storage into a cache representation compatible with the model's existing attention path.

Future contributors can work on reducing queue waiting, adaptive batch sizing, more advanced scheduling policies, or kernel-level paged attention.
