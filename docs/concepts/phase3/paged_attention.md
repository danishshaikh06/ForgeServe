# Phase 3 Concept: PagedAttention & Dynamic Memory Management

## Overview

In traditional LLM inference engines (such as naive PyTorch implementations), memory management for Key-Value (KV) caching relies on **contiguous pre-allocation**. When a sequence starts generating, memory is allocated assuming the maximum possible context length (e.g. 2048 or 4096 tokens). 

This leads to massive **memory fragmentation** and **waste**:
* **Internal Fragmentation**: Reserved slots for tokens that are never actually generated.
* **External Fragmentation**: Virtual memory chunks of varying sizes allocated and deallocated as requests complete.
* **Reservation Waste**: Pre-allocating maximum sequence memory limits the batch size severely.

**PagedAttention** solves this problem by borrowing concepts from operating system virtual memory management (paging).

---

## Key Principles of PagedAttention

### 1. Logical vs. Physical Memory Blocks

Instead of storing KV cache vectors contiguously in memory, PagedAttention divides the KV cache into fixed-size **blocks** (e.g., 16 tokens per block).

```text
Logical KV Cache (Sequence 1):
[ Block 0 (Tokens 0-15) ] -> [ Block 1 (Tokens 16-31) ] -> [ Block 2 (Tokens 32-47) ]

Physical GPU Memory Blocks:
[ Phys Block 42 ]   [ Phys Block 7 ]   [ Phys Block 105 ]   [ Phys Block 12 ]
```

* **Logical Blocks**: Sequential blocks of KV tensors for a specific request.
* **Physical Blocks**: Non-contiguous slots allocated dynamically on the GPU memory pool.

### 2. Block Tables

Each sequence maintains a **Block Table** that maps its logical blocks to physical GPU blocks:

| Logical Block Index | Physical Block ID | Status |
| :--- | :--- | :--- |
| Block 0 | Physical Block 14 | Filled |
| Block 1 | Physical Block 89 | Filled |
| Block 2 | Physical Block 3 | Active (Tokens 32-38 populated) |

When a sequence generates a new token that exceeds its current physical block capacity, the engine allocates a single new physical block from the free memory pool.

---

## Memory Efficiency Comparison

```text
Static Contiguous Cache:
[ Token 0..15 | Token 16..31 | Unused (Allocated to Max Context) ......... ]
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                             80%+ Wasted

Paged KV Cache:
[ Phys Block 14 ] -> [ Phys Block 89 ] -> [ Phys Block 3 (Only current block allocated) ]
                                          Zero wasted pre-allocated memory!
```

---

## Benefits of PagedAttention

1. **Near-Zero Memory Waste**: Memory is allocated incrementally block by block. Unused capacity is at most `Block Size - 1` tokens per sequence.
2. **Increased Batch Size**: Up to 2x - 4x higher throughput by packing significantly more concurrent sequences into GPU memory.
3. **Flexible Copy-on-Write Sharing**: Enables fast parallel sampling (beam search, parallel decoding) where multiple sequences share the exact same prompt prefix physical blocks until they diverge.

---

## Summary

PagedAttention decouples the logical sequence representation from physical GPU memory placement. It is the cornerstone for high-throughput LLM serving.
