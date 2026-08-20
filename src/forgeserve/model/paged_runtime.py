"""
PagedRuntime: Extends Runtime with PagedAttention KV management.

Key difference from Runtime:
    Phase 2 Runtime:     stores full past_key_values as contiguous tuple
    PagedRuntime:        stores KV in fixed blocks via BlockManager
                         gathers blocks before each forward pass
                         extracts only new token after each decode step

The external generation API is unchanged.
Only KV storage strategy changes internally.
"""
from __future__ import annotations

import uuid
import torch 

from forgeserve.model.runtime import Runtime
from forgeserve.model.types import AttentionImplementation
from forgeserve.page_attention.block_manager import BlockManager
from forgeserve.page_attention.paged_cache import PagedKVCache
from forgeserve.page_attention.exception import KVCacheOutOfMemoryError
from forgeserve.logger import get_logger

logger = get_logger(__name__)

class PagedRuntime(Runtime):
    """
    Runtime with PagedAttention KV cache management.

    Args:
        model_name:    HuggingFace model identifier.
        block_manager: Pre-initialized BlockManager with GPU block pool.
        attention:     Attention backend (EAGER or SDPA).
    """
    def __init__(
            self,
            model_name: str,
            block_manager: BlockManager,
            attention: AttentionImplementation.SDPA,
    ) -> None:
        super().__init__(model_name=model_name, attention=attention)
        self.block_manager = block_manager

        logger.info(
            "PagedRuntime initialized. Block pool: %d blocks, "
            "block_size=%d, memory=%.1f MB",
            block_manager.num_blocks,
            block_manager.block_size,
            block_manager._total_memory_mb(
                block_manager.num_blocks,
                block_manager.block_size,
                block_manager.num_layers,
                block_manager.num_heads,
                block_manager.head_dim,
            ),
        )

    def paged_prefill(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            request_id: str | None = None,
    ) -> tuple[torch.Tensor, PagedKVCache]:
        """
        Prefill phase with paged KV storage.

        Steps:
            1. Allocate enough blocks for the full prompt
            2. Run forward pass (same as Phase 2 prefill)
            3. Write each prompt token's KV into blocks
            4. Return logits and PagedKVCache

        Args:
            input_ids:     Prompt token ids. Shape: (1, prompt_len)
            attention_mask: Attention mask. Shape: (1, prompt_len)
            request_id:    Optional request identifier. Auto-generated if None.

        Returns:
            logits:       Shape (1, vocab_size)
            paged_cache:  PagedKVCache with all prompt tokens stored in blocks
        """
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]

        prompt_len = input_ids.shape[1]

        # How many blocks do we need for this prompt?
        # Ceiling division: 13 tokens with block_size=16 needs 1 block
        # 17 tokens with block_size=16 needs 2 blocks
        num_blocks_needed = (prompt_len + self.block_manager.block_size - 1) // self.block_manager.block_size

        logger.debug(
            "Paged prefill: request=%s prompt_len=%d blocks_needed=%d",
            request_id, prompt_len, num_blocks_needed,
        )

        # Allocate blocks — raises KVCacheOutOfMemoryError if pool exhausted
        blocks = self.block_manager.allocate(request_id, num_blocks_needed)

        #create page cache 
        paged_cache = PagedKVCache(
            request_id = request_id,
            initial_blocks= blocks,
            block_size=self.block_manager.block_size,
            num_layers=self.block_manager.num_layers,
        )

        #forward pass
        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

        assert output.logits is not None
        assert output.past_key_values is not None

        # Write each prompt token's KV into blocks
        # past_key_values contains KV for all prompt tokens
        # We write them one by one into the block structure
        for token_pos in range(prompt_len):
            # write down the key and value cache for each layer in the blocks 
            paged_cache.write_token(
                past_key_values=output.past_key_values,
                token_position=token_pos,
            )
        # Shape [batch_size, sequence_length, vocab_size]
        logits = output.logits[:, -1, :]

        logger.debug(
            "Paged prefill complete: request=%s seq_len=%d blocks_used=%d",
            request_id, paged_cache.seq_len, len(paged_cache.block_table),
        )

        return logits, paged_cache

    def paged_decode_step(
            self,
            token_id: torch.Tensor,
            attention_mask: torch.Tensor,
            paged_cache: PagedKVCache,
    ) -> tuple[torch.Tensor, PagedKVCache]:
        """
        Decode one token using paged KV cache.

        Steps:
            1. Check if current block is full — allocate new one if needed
            2. Gather blocks into contiguous past_key_values
            3. Forward pass with single new token
            4. Extract ONLY the last token's KV from output
            5. Write that new token into current block
            6. Return logits and updated paged_cache

        Args:
            token_id:      New token. Shape: (1, 1)
            attention_mask: Full sequence mask. Shape: (1, seq_len + 1)
            paged_cache:   Current paged KV cache for this request.

        Returns:
            logits:       Shape (1, vocab_size)
            paged_cache:  Same object, updated with new token's KV
        """
        current_block = paged_cache.block_table[-1]
        if current_block.is_full:
            if not self.block_manager.can_allocate(1):
                raise KVCacheOutOfMemoryError(
                    f"Request '{paged_cache.request_id}': block pool exhausted "
                    f"during decode at position {paged_cache.seq_len}. "
                    f"Request must wait for blocks to be freed."
                )
            new_blocks = self.block_manager.allocate(paged_cache.request_id, 1)
            paged_cache.append_block(new_blocks[0])
            logger.debug(
                "Request '%s': allocated new block at seq_len=%d",
                paged_cache.request_id, paged_cache.seq_len,
            )

        # Gather blocks (scattered in physical block location)-> contigous past keys and values 
        #This is the bridge betwwen paged storage and hugging face api 
        gathered_kv = paged_cache.gather()

        logger.debug(
            "Decode step: request=%s position=%d blocks=%d",
            paged_cache.request_id,
            paged_cache.seq_len,
            len(paged_cache.block_table),
        )

        # Forward pass with sigle new token 
        output = self.forward(
            input_ids=token_id,
            attention_mask=attention_mask,
            past_key_values = gathered_kv,
            use_cache = True 
        )

        # Extract last token Key and Value vector and write to block 
        # # output.past_key_values has shape (1, num_heads, seq_len+1, head_dim)
        # We only want position -1: the new token we just decoded
        # Everything else is already in our blocks
        new_token_kv = self._extract_last_token_kv(output.past_key_values)
        self._write_token_kv_to_cache(new_token_kv, paged_cache)

        logits = output.logits[:, -1, :]

        return logits, paged_cache

    def free_request(
            self,
            request_id:str
    ) -> None:
        """
        Release all blocks held by a completed request.
        Must be called after generation completes.
        Failure to call this leaks GPU memory.
        """
        self.block_manager.free(request_id)
        logger.debug("Freed blocks for request '%s'", request_id)

    def _extract_last_token_kv(
            self,
            past_key_values: tuple,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract the last token's K and V from all layers.

        Args:
            past_key_values: Full KV from forward pass.
                             K shape per layer: (1, num_heads, seq_len, head_dim)

        Returns:
            List of (k_token, v_token) per layer.
            k_token shape: (num_heads, head_dim)
        """
        result = []
        for k,v in past_key_values:
             # Extract position -1: the newly decoded token
            k_new = k[0, :, -1, :] #  (num_heads, head_dim)
            v_new = v[0, :, -1, :] #  (num_heads, head_dim)
            result.append((k_new, v_new))
        return result 

    def _write_tokne_kv_to_cache(
            self,
            token_kv: list[tuple[torch.Tensor, torch.Tensor]],
            paged_cache: PagedKVCache,
    ) -> None:
        """
        Write one token's KV (all layers) into the current block.

        Args:
            token_kv:    List of (k, v) per layer. k shape: (num_heads, head_dim)
            paged_cache: Target paged cache.
        """
        current_block = paged_cache.block_table[-1]

        for layer_idx, (k_token,v_token) in enumerate(token_kv):
            current_block.write_token(layer_idx, k_token, v_token)

        current_block.increment_filled()
        paged_cache.seq_len+=1

    






