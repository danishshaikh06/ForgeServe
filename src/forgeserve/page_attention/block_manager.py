"""
BlockManager: Global Allocator For KV Block Pool 

Responsibilites:
        - Pre-allocate all blocks at startup (zero cudaMalloc during inference)
        - Maintain free block stack (O(1) allocation and pop/free)
        - Track which request owns which block 
        - Raise OOMException when pool is exausted 

Design: pre-allocation at startup 
    - All GPU memory for KV cache is allocated here, once.
    - Allocation during inference = O(1) pointer operation from free stack.
    - No cudaMalloc during live inference = No latency Spikes 
"""
from __future__ import annotations

import torch 

from forgeserve.page_attention.block import KVBlock
from forgeserve.page_attention.exception import (
    KVCacheOutOfMemoryError,
    KVCacheBlockNotOwnedError,
)
from forgeserve.logger import get_logger

logger = get_logger(__name__)

class BlockManager:
    """
    Manages a fixed pool of KVBlocks pre-allocated on GPU 

    Args:
        num_blocks: Total blocks in pool 
        block_size: Token per block (vLLM default: 16)
        num_layers: Transformer layer count
        num_heads: Attention heads per layer
        head_dim: Dimensions per attention head
        device: GPU device string 
    """
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int, 
        head_dim: int, 
        device: str = "cuda",
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        logger.info(
            "Pre-allocating %d KV blocks (block_size=%d). "
            "Total KV memory: %.1f MB", num_blocks, block_size,
            self._total_memory_mb(num_blocks, block_size, num_layers, num_heads,head_dim),
        )

        # pre-allocate all blocks - this is the only cuda malloc 
        # keeps track of all the KVBlock objects that have been created.
        self._pool: list[KVBlock] = [
            KVBlock(
                block_id= i,
                block_size=block_size,
                num_layers=num_layers,
                num_heads = num_heads,
                head_dim=head_dim,
                device=device,
            )
            for i in range(num_blocks)
        ]

        #Free block stack - O(1) push and pop 
        #Stack means recently freed blocks reused first (cache warm)
        self._free_stack: list[int] = list(range(num_blocks))

        # Keeps the track of request_id along with block_ids
        self._owned: dict[str, list[int]] = {} 

        logger.info(
            "BlockManager ready. %d blocks available",
            self.num_free_blocks,
        )

    def allocate(
            self,
            request_id: str,
            num_blocks: int = 1,
    ) -> list[KVBlock]:
        """
        Allocate blocks from the free pool for a request.

        Args:
            request_id: Unique identifier for the request.
            num_blocks: How many blocks to allocate.

        Returns:
            List of allocated KVBlock objects.

        Raises:
            KVCacheOutOfMemoryError: If pool is exhausted.
        """
        if len(self._free_stack) < num_blocks:
            raise KVCacheOutOfMemoryError(
                f"Cannot allocate {num_blocks} blocks for request '{request_id}'. "
                f"Only {self.num_free_blocks} blocks available. "
                f"Request must wait for blocks to be freed."
            )

        allocated = []
        for _ in range(num_blocks):
            block_id = self._free_stack.pop()
            block = self._pool[block_id]
            block.reset()
            allocated.append(block)

        #Track the ownership of the block
        if request_id not in self._owned:
            self._owned[request_id] = []
        self._owned[request_id].extend(b.block_id for b in allocated)

        logger.debug(
            "Allocated %d block(s) to request '%s'. Free: %d",
            num_blocks, request_id, self.num_free_blocks,
        )

        return allocated

    def free(
            self, 
            request_id: str,
            ) -> None:
        """
        Return all blocks owned by a request to the free pool.

        Args:
            request_id: The request whose blocks should be freed.

        Raises:
            KVCacheBlockNotOwnedError: If request_id is unknown.
        """
        if request_id not in self._owned:
            raise KVCacheBlockNotOwnedError(
                 f"Request '{request_id}' has no allocated blocks to free."
            )
        # Append the free blocks back in the stack 
        block_ids = self._owned.pop(request_id)
        for block_id in block_ids:
            self._free_stack.append(block_id)

        logger.debug(
            "Freed %d block(s) from request '%s'. Free: %d",
            len(block_ids), request_id, self.num_free_blocks,
        )

    @property
    def num_free_blocks(self) -> int:
        """
        Return: Number of free blocks
        """
        return len(self._free_stack)

    @property
    def num_used_blocks(self) -> int:
        """ 
        Returns: How many blocks hvae been allocated
        """
        return self.num_blocks - self.num_free_blocks 

   
    def can_allocate(self, num_blocks: int = 1) -> bool:
        """Check if allocation is possible without raising."""
        return len(self._free_stack) >= num_blocks

    @staticmethod
    def _total_memory_mb(
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ) -> float:
        """
        Calculate the total GPU memory required for all KV-cache blocks.

        Each block stores both the Key (K) and Value (V) caches with shape:
            (num_layers, num_heads, block_size, head_dim)

        Memory is calculated assuming each K/V element uses bfloat16,
        which requires 2 bytes per element.

        Returns:
            Total KV-cache memory in MiB.
        """
        bytes_per_block = (
            2 # K and V 
            * num_layers
            * num_heads
            * block_size
            * head_dim
            * 2 # bfloat16 = 2bytes -> 1 byte = 8 bits 
        )
        return (num_blocks * bytes_per_block) / (1024 ** 2) 
    
    @classmethod
    def from_model_config(
        cls,
        num_blocks: int,
        block_size: int,
        model,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> BlockManager:
        """
        Construct BlockManager from a loaded HuggingFace model config.
        Automatically reads num_layers, num_heads, head_dim.

        Args:
            num_blocks: How many blocks to pre-allocate.
            block_size: Tokens per block.
            model:      Loaded HuggingFace model.
            device:     GPU device string.
        """

        config = model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_key_value_heads # GQA-aware
        head_dim = config.hidden_size // config.num_attention_heads 

        return cls(
            num_blocks = num_blocks,
            block_size = block_size,
            num_layers = num_layers,
            num_heads = num_heads,
            head_dim = head_dim,
            device = device,
        )
