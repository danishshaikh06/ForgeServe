# Paged KV Cache & PagedAttention — Complete Notes & Concepts

## 1. Big Picture

The primary challenge in high-throughput LLM inference is:

> **How do we store and reuse the Key-Value (KV) cache efficiently during LLM inference?**

The Paged KV Cache system is organized into three core components:

```text
KVBlock
   ↓
actual KV-cache storage

BlockManager
   ↓
manages/allocates the KV blocks

PagedKVCache
   ↓
manages the blocks for one request and writes/gathers KV data
```

### Mental Model & Core Responsibilities:

```text
KVBlock      → "What is the storage?"
BlockManager → "Who gets the storage?"
PagedKVCache → "How do I use the storage for this request?"
```

---

## 2. What is the KV Cache?

During Transformer attention, every layer produces three primary projection vectors:
* **Query (Q)**
* **Key (K)**
* **Value (V)**

For autoregressive text generation, we do not want to recompute the Key and Value vectors for all previous tokens every time we generate a single new token.

Instead, we **cache K and V** from previous tokens.

### Example Sequence:
Input sequence:
```text
T0 T1 T2 T3
```

After processing them through the model, we store:
```text
Layer 0 → K/V for T0 T1 T2 T3
Layer 1 → K/V for T0 T1 T2 T3
Layer 2 → K/V for T0 T1 T2 T3
...
```

When a new token `T4` arrives, the model reuses `K/V for T0 T1 T2 T3` instead of recomputing them from scratch.

---

## 3. Why Do We Store KV for Every Layer?

A common misconception is: *"The last layer contains the most information, so why not just store the last layer's KV?"*

The answer is: **Each Transformer attention layer operates independently and requires its own historical K/V data.**

Suppose a model has 3 layers:
```text
Layer 0 → K0, V0
Layer 1 → K1, V1
Layer 2 → K2, V2
```

When token `T4` is processed:
1. `T4` enters **Layer 0** $\rightarrow$ Layer 0 requires previous `K0, V0`.
2. `T4` enters **Layer 1** $\rightarrow$ Layer 1 requires previous `K1, V1`.
3. `T4` enters **Layer 2** $\rightarrow$ Layer 2 requires previous `K2, V2`.

> **Key Takeaway:** The last layer's K/V cannot replace earlier layers' K/V. Each layer has its own attention operation and must maintain its own K/V cache history.

---

## 4. Physical Structure of a `KVBlock`

One physical `KVBlock` represents **one physical piece of KV-cache memory**.

A single physical block contains Key and Value data for **all Transformer layers** for a fixed chunk of token positions (`block_size`).

```text
Block 7
│
├── Layer 0
│    ├── K → 16 tokens
│    └── V → 16 tokens
│
├── Layer 1
│    ├── K → 16 tokens
│    └── V → 16 tokens
│
├── Layer 2
│    ├── K → 16 tokens
│    └── V → 16 tokens
│
└── Layer 3
     ├── K → 16 tokens
     └── V → 16 tokens
```

> **Key Rule:** A block represents a chunk of tokens, and for that chunk it stores K/V data for every Transformer layer.

For `block_size = 16`:
* **Block 0** $\rightarrow$ tokens 0–15, for **ALL** layers
* **Block 1** $\rightarrow$ tokens 16–31, for **ALL** layers
* **Block 2** $\rightarrow$ tokens 32–47, for **ALL** layers

---

## 5. Tensor Shapes within a `KVBlock`

For a configuration of:
```python
num_layers = 4
num_heads  = 8
block_size = 16
head_dim   = 64
```

The tensor shapes for a single physical block are:
```python
K_shape = (num_layers, num_heads, block_size, head_dim) # (4, 8, 16, 64)
V_shape = (num_layers, num_heads, block_size, head_dim) # (4, 8, 16, 64)
```

Dimension Breakdown:
* `num_layers` (4): Total number of Transformer layers.
* `num_heads` (8): Total Key-Value attention heads.
* `block_size` (16): Total token capacity inside this physical block.
* `head_dim` (64): Dimension size of each attention head.

---

## 6. What is `block_size` and `num_filled`?

### `block_size`
Defines **how many token positions one physical block can hold**.

For `block_size = 4`:
```text
Block 3
┌────┬────┬────┬────┐
│ T0 │ T1 │ T2 │ T3 │
└────┴────┴────┴────┘
```

### `num_filled`
Tracks **how many token positions in a specific block are currently populated**.

Progression example for `block_size = 4`:
```text
Initially:     [   ][   ][   ][   ] -> num_filled = 0
After T4:      [T4 ][   ][   ][   ] -> num_filled = 1
After T5:      [T4 ][T5 ][   ][   ] -> num_filled = 2
After T6 & T7: [T4 ][T5 ][T6 ][T7 ] -> num_filled = 4 (Block FULL)
```

---

## 7. Pre-Allocation & Block Tracking (`_pool`, `_free_stack`, `_owned`)

Blocks are created **before a request's prefill phase**. When `BlockManager` initializes, it pre-allocates all physical GPU memory tensors.

### Tracking Structures:
1. **`_pool`**: Keeps references to all physical `KVBlock` objects created in GPU memory.
   ```text
   _pool = [Block 0, Block 1, Block 2, Block 3, Block 4]
   ```
2. **`_free_stack`**: Tracks which block IDs are currently available for allocation.
   ```text
   _free_stack = [0, 1, 2, 3, 4]  # Stack push/pop operates in O(1) time
   ```
3. **`_owned`**: Maps request IDs to their assigned physical block IDs.
   ```python
   _owned = {
       "request_A": [4, 3],
       "request_B": [1]
   }
   ```

> **Creating a block = allocating its GPU KV memory up-front.**

---

## 8. Complete Request Lifecycle (Prefill to Decode)

```text
                         USER TEXT
                            │
                            ↓
                        Tokenizer
                            │
                            ↓
                   [T0 T1 T2 T3 T4]
                            │
                            ↓
                  Know sequence length (5)
                            │
                            ↓
                 Calculate blocks needed (ceil(5/4) = 2 blocks)
                            │
                            ↓
                    BlockManager
                            │
                     allocate blocks
                            │
             ┌──────────────┴──────────────┐
             ↓                             ↓
         KVBlock                         KVBlock
         Block 4                         Block 3
         (EMPTY)                         (EMPTY)
             │                             │
             └──────────────┬──────────────┘
                            ↓
                         PREFILL
                            │
                            ↓
                     Transformer
                            │
          ┌─────────────────┼────────────────┐
          ↓                 ↓                ↓
       Layer 0           Layer 1          Layer 2
          │                 │                │
        K/V               K/V              K/V
          │                 │                │
          └─────────────────┼────────────────┘
                            ↓
                    past_key_values
                            │
                            ↓
                       write_token()
                            │
                            ↓
                     Physical blocks
                            │
             ┌──────────────┴──────────────┐
             ↓                             ↓
          Block 4                        Block 3
       T0 T1 T2 T3                         T4
                            │
                            ↓
                         DECODE
                            │
                       New token T5
                            │
                            ↓
                  Reuse previous KV + compute new K/V
                            │
                            ↓
                       write_token()
                            │
                            ↓
                       Block 3
                    T4 T5 T6 ...
```

---

## 9. Core Mental Model Summary

```text
                 KV CACHE SYSTEM

KVBlock
   │
   │  "I am the actual physical GPU memory."
   ↓
┌───────────────────────────────┐
│ Layer 0 → K/V                 │
│ Layer 1 → K/V                 │
│ Layer 2 → K/V                 │
│ ...                           │
│ Layer N → K/V                 │
│ for block_size tokens         │
└───────────────────────────────┘

BlockManager
   │
   │ "I manage all physical blocks on the device."
   ↓
[Block 0][Block 1][Block 2][Block 3]...

PagedKVCache
   │
   │ "I manage block mappings and KV reads/writes for Request A."
   ↓
Request A → [Block 4, Block 3, Block 2]
```

### The Golden Rule:
> **`KVBlock` provides physical KV storage, `BlockManager` manages and allocates those physical blocks across requests, and `PagedKVCache` maps a request's logical token sequence onto those blocks while handling per-layer K/V reads and writes.**
