# Batched Forward Pass

## From Separate Forwards to One Forward

Before batched decoding, four active requests might execute:

```text
forward(A)
forward(B)
forward(C)
forward(D)
```

The batched version performs:

```text
forward([A, B, C, D])
```

once per decode step.

## Why the GPU Benefits

The model weights are loaded into GPU memory when the model is initialized and normally remain resident there.

The important difference is not repeatedly copying weights from CPU to GPU.

Instead, separate forward passes cause the GPU to perform separate model operations.

For a linear layer:

\[
Y=XW^T
\]

Separate requests:

\[
Y_1=X_1W^T
\]

\[
Y_2=X_2W^T
\]

\[
Y_3=X_3W^T
\]

A batched version stacks the inputs:

\[
X=
\begin{bmatrix}
X_1\\
X_2\\
X_3
\end{bmatrix}
\]

and computes:

\[
\boxed{
Y=XW^T
}
\]

This creates a larger matrix operation and allows the GPU to process the requests together.

## Decode Input Shape

In autoregressive decode, each active request contributes one current token.

For \(N\) requests:

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

## Output Shape

The model produces one output sequence per batch row.

Conceptually:

```text
logits.shape = (N, 1, vocab_size)
```

The final token position gives:

```text
next_logits.shape = (N, vocab_size)
```

Then:

```text
row 0 → request A
row 1 → request B
row 2 → request C
row 3 → request D
```

The row-to-request mapping must never change accidentally.

## The Decode Cycle

A batched decode step is:

```text
1. Identify active requests
2. Check KV block requirements
3. Allocate blocks if needed
4. Collect one token from each request
5. Build batch metadata
6. Run one model forward
7. Sample one token per request
8. Update each request's KV state
9. Detect completed requests
10. Release completed requests
```

## Why This Is Different from Just Calling the Model with More Text

The requests can have different lengths.

Example:

```text
A → 12 cached tokens
B → 20 cached tokens
C → 8 cached tokens
```

So the runtime must combine:

- current input tokens
- attention information
- position information
- each request's KV state

into a representation the model can process as a batch.

That is the difficult part of batched inference.

## ForgeServe's Practical Path

The current paged KV architecture can conceptually follow:

```text
paged KV per request
       ↓
gather
       ↓
contiguous padded representation
       ↓
batched cache
       ↓
one model forward
```

A future kernel-level paged-attention implementation could avoid some of this gather work.

## Key Insight

The main transformation is:

```text
N requests
   ↓
N forward calls
```

becomes:

```text
N requests
   ↓
1 batched forward call
```

while preserving independent state for every request.
