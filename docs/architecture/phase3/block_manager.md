# Phase 3 Architecture: Block Manager & KV Memory Allocator

## Overview

The `BlockManager` (and underlying `BlockAllocator`) handles allocation, tracking, and freeing of fixed-size physical memory blocks in GPU memory for PagedAttention.

---

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                            BlockManager                               |
|                                                                       |
|  +------------------------+        +-------------------------------+  |
|  |     Free Block Pool    |        |       Sequence Block Tables   |  |
|  | [B0, B1, B2, B3, ...]  |        | Seq 1: [B0 -> P12, B1 -> P4]  |  |
|  +------------------------+        | Seq 2: [B0 -> P99]            |  |
|                                    +-------------------------------+  |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                          GPU Memory Pool                              |
|  +-------------------+ +-------------------+ +-------------------+    |
|  | Physical Block P12| | Physical Block P4 | | Physical Block P99|    |
|  | [16 x K, 16 x V]  | | [16 x K, 16 x V]  | | [16 x K, 16 x V]  |    |
|  +-------------------+ +-------------------+ +-------------------+    |
+-----------------------------------------------------------------------+
```

---

## Core Technical Components

### 1. `PhysicalTokenBlock`
Represents a single physical memory block on the GPU device.

* **Attributes**:
  * `block_number`: Unique identifier for the block (0 to `num_blocks - 1`).
  * `block_size`: Number of token slots per block (typically 16 or 32).
  * `ref_count`: Reference counter to track sequence sharing (used for prefix sharing & beam search).

### 2. `BlockAllocator`
Manages the free block queue and allocations.

* **Primary Operations**:
  * `allocate()`: Pops a free physical block ID from the free pool. Throws GPU Out-Of-Memory if empty.
  * `free(block_id)`: Decrements `ref_count`. If `ref_count == 0`, returns the block ID back to the free pool.

### 3. `BlockTable`
Maintains the mapping of sequence logical token indices to physical block IDs.

```python
class BlockTable:
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.physical_blocks: List[PhysicalTokenBlock] = []

    def get_physical_block_id(self, logical_idx: int) -> int:
        block_offset = logical_idx // self.block_size
        return self.physical_blocks[block_offset].block_number
```

---

## Memory Calculation Formula

The total number of physical blocks available on the GPU is computed as:

$$\text{Num Blocks} = \left\lfloor \frac{\text{Total Dedicated KV Memory (Bytes)}}{2 \times \text{Num Layers} \times \text{Num Heads} \times \text{Head Dim} \times \text{Block Size} \times \text{Element Size (Bytes)}} \right\rfloor$$

---

## Execution Flow: Sequence Allocation & Growth

1. **Prompt Prefill Phase**:
   * Calculate required logical blocks: $\lceil \text{Prompt Length} / \text{Block Size} \rceil$.
   * `BlockAllocator.allocate()` allocates the exact number of blocks.

2. **Autoregressive Generation Step**:
   * Token index $N$ is generated.
   * If $N \pmod{\text{Block Size}} == 0$, allocate a new physical block and append to `BlockTable`.
   * Otherwise, populate the next slot inside the current active physical block.

3. **Request Cleanup**:
   * On request completion, all physical blocks associated with the sequence are freed back to the free pool.
