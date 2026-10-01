# Paged KV Storage

## Overview

Phase 4 introduces a paged KV-cache memory-management system. Instead of giving every request one large contiguous KV allocation, ForgeServe divides KV storage into fixed-size blocks managed by a global pool.

```text
Global KV pool
┌────┬────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │ ...│
└────┴────┴────┴────┴────┴────┘
```

Each request owns a logical list of blocks:

```text
Request A → [B7, B3, B1]
Request B → [B5, B2]
```

The physical blocks do not need to be adjacent.

## Configuration

Phase 4 used:

| Setting | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| GPU | NVIDIA RTX 4070 |
| Block size | 16 tokens |
| Total blocks | 256 |
| KV pool | 48.0 MB |
| Max new tokens/request | 128 |

The pool represents:

```text
256 × 16 = 4096 token slots
```

## Block Requirement

For a request with `T` total sequence tokens:

\[
B = \left\lceil\frac{T}{\text{block size}}\right\rceil
\]

For example:

```text
Prompt      = 40 tokens
Generated   = 128 tokens
Total       = 168 tokens
Block size  = 16
```

Therefore:

\[
B = \left\lceil\frac{168}{16}\right\rceil = 11
\]

## Allocation and Ownership

The `BlockManager` allocates block IDs from the free pool and records request ownership. When a request finishes, its blocks are returned to the pool for reuse.

The core pool invariant is:

\[
\boxed{B_{used}+B_{free}=B_{total}}
\]

For the Phase 4 configuration:

\[
B_{used}+B_{free}=256
\]

## What Phase 4 Proved

The benchmark validated correct allocation, request ownership, reuse, release, stable preallocated KV memory, and leak-free cleanup. Across the tested scenarios the final pool returned to **256 free blocks**.

The eight-request scenario generated **1,024 total tokens**, but the requests were sequential, so those tokens were not simultaneously resident. Each request used about 11 blocks and released them before the next request began.

## Current Scope

The implementation should be described as **paged KV-cache storage and block management**, not kernel-level PagedAttention. The current path may gather paged blocks into a contiguous cache representation before model attention executes.

A future kernel-level paged-attention implementation could consume block tables directly.
