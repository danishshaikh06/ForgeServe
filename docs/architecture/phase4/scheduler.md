# Phase 4 Architecture: Iteration-Level Scheduler & Batched Decode Engine

## Overview

This document details the architectural specification for continuous batching, batched decode tensor operations, left-padded attention masks, dynamic block pre-allocation, and output distribution in ForgeServe.

---

## 1. Batched Decode Tensor Layouts

During a single batched decode step with $N$ active requests:

### Token Input Tensor
Each active request provides exactly one new input token:
$$\boxed{batch\_tokens.shape = (N, 1)}$$

```python
batch_tokens = torch.stack([req.last_token for req in active_requests], dim=0) # (N, 1)
```

### Logits & Sampling Output
```text
outputs.logits.shape = (N, 1, vocab_size)
next_logits = outputs.logits[:, -1, :]  # Shape: (N, vocab_size)
next_tokens = torch.argmax(next_logits, dim=-1) # Shape: (N, 1)
```

$$\boxed{batch[i] \leftrightarrow request_i \text{ (Order Invariant across all operations)}}$$

---

## 2. Resolving the 3 Core Batched Decode Questions

### Question 1: How are variable sequence lengths and Position IDs handled?
Suppose Request A has length 6 (`[tok0..tok5]`) and Request B has length 13 (`[tok0..tok12]`).
We pad the sequence to $L_{max} = 13$ using **Left-Padding**:

```text
Request A mask (padded): [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  (7 zeros, 6 ones + 1 new)
Request B mask (padded): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  (13 ones + 1 new)
```

The model automatically computes Position IDs by counting valid `1` entries in the attention mask:
* **Request A**: 7 zeros then 7 ones $\rightarrow$ Position ID = $7 - 1 = 6$ ✅
* **Request B**: 13 ones $\rightarrow$ Position ID = $13$ ✅

> **Takeaway:** The left-padded attention mask handles position IDs automatically without manual intervention.

---

### Question 2: Which position in the output KV contains the new token?
After a forward pass, the model returns updated KV tensors of shape $(N, \text{num\_heads}, L_{max} + 1, \text{head\_dim})$.

Because tokens are appended at the end of the sequence dimension, **the new token is ALWAYS at index `-1`**:
```python
new_kv_token = output_kv[i, :, -1, :]  # Shape: (num_heads, head_dim)
```
This slice is written directly into request $i$'s active physical GPU block slot.

---

### Question 3: Why MUST blocks be allocated BEFORE the forward pass?
```text
CRITICAL FAULT IF ALLOCATED AFTER:
1. Run batched forward pass on GPU (expensive CUDA execution completes).
2. Attempt to write KV to block -> Block is FULL!
3. BlockManager out of blocks -> KVCacheOutOfMemoryError!
4. Results in corrupted GPU state, wasted execution, and unrecoverable crash.

CORRECT ORDERING (PRE-ALLOCATION BEFORE FORWARD):
1. Inspect all requests in batch prior to execution.
2. If request i will cross a block boundary (num_filled == block_size), allocate NEW block now.
3. If allocation fails -> defer request/step before touching GPU.
4. Run forward pass -> All KV block writes are GUARANTEED to succeed!
```

$$\boxed{\text{Rule: Never run GPU work if you cannot store the output.}}$$

---

## 3. Paged KV Storage vs. Dense Batched Cache (Stage 1 vs Stage 2)

```text
STAGE 1 (Current Implementation):
Per-Request Paged KV -> gather_padded() -> Dense Batched Tensor -> PyTorch Model Forward

STAGE 2 (Future True Paged Attention):
Per-Request Paged KV -> Custom Triton/CUDA PagedAttention Kernel (No gathering!)
```

### `gather_padded()` Mathematics & Tensor Operations
For a request $r$ with physical blocks $[K_{r,1}, K_{r,2}, \dots, K_{r,B}]$:

1. **Extract Valid Slots**:
   Slice filled slots from each block: $K_{r,b} = \text{block.k\_cache}[:, :, :\text{num\_filled}, :]$
2. **Concatenate Blocks**:
   $$K_r^{full} = \operatorname{concat}_{seq}(K_{r,1}, K_{r,2}, \dots, K_{r,B}) \quad \text{Shape: } (\text{layers}, \text{heads}, S_r, \text{head\_dim})$$
3. **Left-Pad to $L_{max}$**:
   Create padding zeros tensor $Z \in \mathbb{R}^{\text{layers} \times \text{heads} \times (L_{max} - S_r) \times \text{head\_dim}}$:
   $$K_r^{padded} = \operatorname{concat}_{seq}(Z, K_r^{full})$$

---

## 4. The 7 Correctness Invariants

To guarantee system correctness, the runtime enforces:

1. **Pool Conservation**: $B_{used} + B_{free} = N$ (Always 256).
2. **Request Ownership**: $B_i = |\text{owned\_blocks}_i|$.
3. **No Double Ownership**: $\text{owned}(R_i) \cap \text{owned}(R_j) = \emptyset \quad \forall i \neq j$.
4. **Clean Release**: $R_i = \text{FINISHED} \implies B_i = 0$.
5. **Exact Block Count**: $B_i = \lceil T_i / b \rceil$.
6. **Capacity Bounds**: $B_{used} \le N$.
7. **Order Stability**: $\text{batch}[i] \leftrightarrow \text{request}_i$ throughout token collection, forward pass, and output distribution.

---

## 5. System Pseudocode Loop

```python
while engine_is_running:
    # 1. Admit waiting requests if free blocks exist
    while waiting_queue and block_manager.free_blocks >= waiting_queue[0].initial_blocks:
        req = waiting_queue.pop(0)
        block_manager.allocate(req.id, req.initial_blocks)
        running_queue.append(req)

    if not running_queue:
        continue

    # 2. Check & Pre-allocate blocks BEFORE forward pass
    for req in running_queue:
        if req.needs_new_block():
            block_manager.allocate(req.id, 1)

    # 3. Gather padded KV tensors & build batch inputs
    batch_tokens = torch.stack([req.get_last_token() for req in running_queue], dim=0) # (N, 1)
    batch_kv = gather_padded_batch(running_queue) # Layer-by-layer padded KV tensors

    # 4. ONE batched model forward pass
    outputs = model(input_ids=batch_tokens, past_key_values=batch_kv)

    # 5. Extract next tokens & write back KV
    next_logits = outputs.logits[:, -1, :] # (N, vocab_size)
    next_tokens = torch.argmax(next_logits, dim=-1)

    for i, req in enumerate(running_queue):
        req.append_token(next_tokens[i])
        
        # Extract new KV at index -1
        new_k = outputs.past_key_values.extract(layer=..., batch_idx=i, pos=-1)
        req.write_kv_token(new_k)

        # 6. Check completion & release blocks
        if req.is_finished():
            block_manager.free(req.id)
            running_queue.remove(req)
```

---

## 6. Architecture Responsibilities

$$\boxed{\text{Scheduler = WHEN / WHO}} \quad \text{Selects active batch, handles queues & completion.}$$
$$\boxed{\text{BlockManager = WHERE}} \quad \text{Manages GPU physical block pools \& ownership.}$$
$$\boxed{\text{PagedRuntime = HOW}} \quad \text{Performs tensor gathering, left-padding, and PyTorch model execution.}$$
