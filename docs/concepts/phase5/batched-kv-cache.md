# Batched KV Cache

## The Problem

Each request has its own KV cache.

For example:

```text
Request A → 6 cached tokens
Request B → 13 cached tokens
```

Their cache lengths are different.

But a normal dense batched tensor is rectangular.

So the runtime needs to transform the per-request caches into a common batched representation.

## Per-Request KV

Conceptually, one request has:

\[
K_i,V_i
\]

For layer \(l\), a dense KV tensor can be represented conceptually as:

\[
K_l
\in
\mathbb{R}^{N\times H_{kv}\times L\times D}
\]

for a batch of \(N\) requests with common sequence dimension \(L\).

## Left Padding

Let:

\[
L_{max}=\max(T_1,T_2,\ldots,T_N)
\]

For each request:

```text
request KV
    ↓
gather blocks
    ↓
contiguous KV
    ↓
left-pad to Lmax
```

Example:

```text
A → [PAD PAD PAD PAD PAD PAD PAD A0 A1 A2 A3 A4 A5]
B → [B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12]
```

## Stacking

Once each request has the same logical tensor shape, they can be combined along a batch dimension.

Conceptually:

```text
A padded KV ┐
B padded KV ├──→ batched KV
C padded KV │
D padded KV ┘
```

The exact dimension ordering depends on the model/cache API.

## Why the Current `gather_padded()` Exists

A request's KV blocks are stored in the global paged pool.

They may be physically scattered:

```text
Request A → B7, B3, B1
```

The runtime first gathers them:

```text
B7 + B3 + B1
      ↓
contiguous request KV
```

Then pads to the target batch length.

So:

\[
\boxed{
\text{paged KV}
\rightarrow
\text{contiguous padded KV}
}
\]

is a preparation step.

It does not by itself create the batch.

## `gather_as_dynamic_cache()`

A different helper can gather one request and place its K/V into a Hugging Face `DynamicCache`.

Conceptually:

```text
paged KV
   ↓
contiguous KV
   ↓
DynamicCache
```

The important difference is the output type.

## Batched Cache Challenge

Suppose:

```text
A → 6 tokens
B → 13 tokens
```

You cannot simply pass:

```text
cache_A
cache_B
```

as two unrelated caches to one model invocation.

The model needs a single batch-level cache representation compatible with its attention implementation.

## True Paged Attention

A more advanced architecture would avoid gathering into a dense cache.

Instead:

```text
block pool
+
block tables
+
sequence lengths
        ↓
paged attention kernel
```

The attention kernel directly uses each request's blocks.

ForgeServe's current implementation is better described as paged KV storage and block management, with gathering used to interface with the existing attention path.

## Key Insight

The scheduler problem is:

```text
many request states
```

The cache problem is:

```text
make those states appear as one batch
```

The runtime must solve both without mixing the KV data of different requests.
