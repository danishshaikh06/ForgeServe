"""
KVBlock: The fundamental unit of PagedAttention memory management.

A KVBlock owns a fixed slice of pre-allocated GPU memory.
It stores Keys and Values for up to block_size tokens.
It does not know which request it belongs to — the BlockManager
and PagedKVCache manage that relationship.

Design decision:
    We store K and V for ALL layers in one block object.
    Alternative: one block per layer.
    We chose all-layers because:
        - Simpler block table (one entry covers all layers)
        - Allocation is per-token not per-layer
        - Matches vLLM's design
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KVBlock:
    """
    Fixed-size unit of KV cache storage.

    Attributes:
        block_id:    Unique identifier for this block in the pool.
        block_size:  Maximum tokens this block can store.
        num_layers:  Number of transformer layers.
        num_heads:   Number of attention heads per layer.
        head_dim:    Dimension of each attention head.
        device:      GPU device this block lives on.
        num_filled:  How many token slots are currently occupied.
        k_cache:     Key tensor. Shape: (num_layers, num_heads, block_size, head_dim)
        v_cache:     Value tensor. Shape: (num_layers, num_heads, block_size, head_dim)
    """
    block_id: int
    block_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    device: str
    num_filled: int = field(default=0, init=True)

    # KV storage tensors - allocated once, resued accross requests
    k_cache: torch.Tensor = field(init=False)
    v_cache: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        """
        Pre-allocate GPU memory for this block.
        Shape:(num_layers, num_heads, block_size, head_dim).
        The K and V cache are stored in the same block as per block size but in seperate tensors.
        """
        shape = (self.num_layers, self.num_heads, self.block_size, self.head_dim)
        self.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=self.device)

    @property
    def is_full(self) -> bool:
        """
        True when no more tokens can be written to this block
        """
        return self.num_filled >= self.block_size

    @property
    def num_free_slot(self) -> int:
        """
        How many token slot remain in this block
        """
        return self.block_size - self.num_filled

    def write_token(
            self,
            layer_idx: int,
            k: torch.Tensor,
            v: torch.Tensor,
    ) -> None:
        """
        Write one token's K and V into the next free slot.
        Args:
            layer_idx: Which transformer layer these kv tensors are from
            k: Key tensor for this token, Shape: (num_heads, head_dim)
            v: Value tensor for this token, Shape: (num_heads, head_dim)
        """
        if self.is_full:
            raise RuntimeError(
                f"Block {self.block_id} is full ({self.block_size} tokens)."
                "Allocate a new block before writing"
            )
        slot = self.num_filled
         # In layer idx, at token position slot, write the K and v values for all heads.
        self.k_cache[layer_idx, :, slot, : ] = k
        self.v_cache[layer_idx, :, slot, : ] = v

    def increment_filled(self) -> None:
        """
        Advance the fill pointer after writing all layers for one token.
        Called once per token after write_token for all layers.
        """
        self.num_filled+=1

    def reset(self) -> None:
        """
        Reset this block for reuse by a new request.
        Does not zero the memory — the fill pointer guards stale data.
        """
        self.num_filled = 0


