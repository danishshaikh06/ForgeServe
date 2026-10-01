# Phase 3 Architecture: KVBlock, BlockManager & PagedKVCache

## Overview

This architecture document details the technical implementation, tensor indexing, data structures, and per-request binding mechanisms powering Paged KV Caching in ForgeServe.

---

## 1. Class Hierarchy & System Bridge

The Paged KV Cache architecture consists of three decoupled components:

```text
+-------------------------------------------------------------------------+
|                              BlockManager                               |
|  - _pool: List[KVBlock]         -> Keeps all physical blocks            |
|  - _free_stack: List[int]       -> Free block ID stack (O(1) pop/push)   |
|  - _owned: Dict[str, List[int]] -> Maps request_id to physical block IDs|
+-------------------------------------------------------------------------+
                                     |
                                 Allocates
                                     v
+-------------------------------------------------------------------------+
|                                KVBlock                                  |
|  - block_id: int               - num_filled: int                        |
|  - block_size: int             - k_cache: torch.Tensor                  |
|  - num_layers, num_heads, head_dim                                      |
|  - Shape: (num_layers, num_heads, block_size, head_dim)                |
+-------------------------------------------------------------------------+
                                     ^
                                 References
                                     |
+-------------------------------------------------------------------------+
|                              PagedKVCache                               |
|  - block_table: List[KVBlock]  -> Logical-to-Physical mapping           |
|  - seq_len: int                -> Total request token count             |
|  - write_token(...)            -> Writes model KV output to blocks      |
|  - gather()                    -> Reconstructs logical KV for model HF  |
+-------------------------------------------------------------------------+
```

---

## 2. `KVBlock` Technical Specification

### Definition & Attributes
```python
class KVBlock:
    def __init__(
        self,
        block_id: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: str = "cuda"
    ):
        self.block_id = block_id
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.num_filled = 0

        # Pre-allocated GPU Tensors
        shape = (num_layers, num_heads, block_size, head_dim)
        self.k_cache = torch.zeros(shape, device=device)
        self.v_cache = torch.zeros(shape, device=device)

    def reset(self) -> None:
        """Resets block filled counter before reallocation."""
        self.num_filled = 0
```

### Physical Tensor Layout
```text
K Tensor Shape = (num_layers, num_heads, block_size, head_dim)
V Tensor Shape = (num_layers, num_heads, block_size, head_dim)
```

For example (`num_layers=4`, `num_heads=8`, `block_size=16`, `head_dim=64`):
* `k_cache.shape` = `torch.Size([4, 8, 16, 64])`
* `v_cache.shape` = `torch.Size([4, 8, 16, 64])`

---

## 3. `BlockManager` Technical Specification

### Core Data Structures:
* **`self._pool: List[KVBlock]`**: Contains references to all pre-created `KVBlock` instances on the GPU.
* **`self._free_stack: List[int]`**: Tracks IDs of unallocated physical blocks. Initialized as `[0, 1, 2, ..., num_blocks - 1]`.
* **`self._owned: Dict[str, List[int]]`**: Dictionary tracking which request ID owns which physical block IDs.

### Allocation & Lifecycle Methods:

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, num_layers: int, num_heads: int, head_dim: int):
        self._pool = [
            KVBlock(i, block_size, num_layers, num_heads, head_dim)
            for i in range(num_blocks)
        ]
        self._free_stack = list(range(num_blocks))
        self._owned = collections.defaultdict(list)

    def allocate(self, request_id: str, num_blocks: int = 1) -> List[KVBlock]:
        if len(self._free_stack) < num_blocks:
            raise RuntimeError("GPU Memory Out of Blocks")

        allocated = []
        for _ in range(num_blocks):
            block_id = self._free_stack.pop() # O(1) free block retrieval
            block = self._pool[block_id]
            block.reset() # Reset old metadata prior to new request ownership
            allocated.append(block)

        # Track ownership
        self._owned[request_id].extend(b.block_id for b in allocated)
        return allocated

    def free(self, request_id: str) -> None:
        """Returns all physical blocks owned by request_id back to _free_stack."""
        block_ids = self._owned.pop(request_id, [])
        for block_id in block_ids:
            self._pool[block_id].reset()
            self._free_stack.append(block_id)
```

---

## 4. `PagedKVCache` (Per-Request Layer)

### Block Table Mapping
The `PagedKVCache` object maintains a per-request `block_table`.

```text
Logical View:   [ Block 0 (T0..T3) ] -> [ Block 1 (T4..T7) ] -> [ Block 2 (T8..T11) ]
Physical View:  [ Phys Block 4     ] -> [ Phys Block 3     ] -> [ Phys Block 2      ]
```

### `write_token()` Step Logic
Extracts single token K/V slices from model output logits and writes them into the active physical block across all Transformer layers.

```python
def write_token(self, past_key_values, token_position: int) -> None:
    # 1. Determine logical and physical block index
    block_index = token_position // self.block_size
    current_block = self.block_table[block_index]
    slot_offset = current_block.num_filled

    # 2. Iterate through all layers
    for layer_idx, (k, v) in enumerate(past_key_values):
        # Extract token's K/V tensor: k shape is (1, num_heads, seq_len, head_dim)
        # Slicing at token_position gives shape: (num_heads, head_dim)
        k_token = k[0, :, token_position, :]
        v_token = v[0, :, token_position, :]

        # Write directly into physical GPU block tensor slice
        current_block.k_cache[layer_idx, :, slot_offset, :] = k_token
        current_block.v_cache[layer_idx, :, slot_offset, :] = v_token

    # 3. Update block and request metadata
    current_block.num_filled += 1
    self.seq_len += 1
```

### Distinction: `num_filled` vs `seq_len`
* **`num_filled`**: Tokens currently stored inside a **specific physical block** ($0 \le \text{num\_filled} \le \text{block\_size}$).
* **`seq_len`**: Total cumulative tokens generated for the **entire request**.

---

## 5. `gather()` Reconstruction Bridge

When interfacing with models or kernels that expect contiguous `past_key_values` tensors, `gather()` concatenates physical blocks in logical order:

```python
def gather(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Gathers scattered physical blocks into a contiguous KV cache representation."""
    gathered_past_kv = []
    
    for layer_idx in range(self.num_layers):
        layer_k_list = []
        layer_v_list = []
        
        for block in self.block_table:
            # Extract filled token slots for this layer
            # Shape: (num_heads, num_filled, head_dim)
            k_slice = block.k_cache[layer_idx, :, :block.num_filled, :]
            v_slice = block.v_cache[layer_idx, :, :block.num_filled, :]
            
            layer_k_list.append(k_slice)
            layer_v_list.append(v_slice)
            
        # Concatenate along sequence dimension (dim=1) -> (1, num_heads, seq_len, head_dim)
        full_k = torch.cat(layer_k_list, dim=1).unsqueeze(0)
        full_v = torch.cat(layer_v_list, dim=1).unsqueeze(0)
        gathered_past_kv.append((full_k, full_v))
        
    return gathered_past_kv
```

---

## 6. Summary: Component Method Responsibilities

| Class | Method | Primary Responsibility |
| :--- | :--- | :--- |
| `KVBlock` | `reset()` | Clears `num_filled` counter prior to block reuse. |
| `BlockManager` | `allocate(req_id, n)` | Pops `n` block IDs from `_free_stack`, resets them, and registers ownership in `_owned`. |
| `BlockManager` | `free(req_id)` | Returns all block IDs owned by `req_id` to `_free_stack`. |
| `PagedKVCache` | `write_token(...)` | Computes block offset, iterates across layers, and writes `k[0, :, pos, :]` into block slots. |
| `PagedKVCache` | `gather()` | Assembles non-contiguous GPU block slices into contiguous logical `past_key_values`. |
