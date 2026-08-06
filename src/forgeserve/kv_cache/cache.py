from __future__ import annotations

from dataclasses import dataclass

import torch

from forgeserve.kv_cache.exception import KVCacheNotInitializedError

from transformers.cache_utils import Cache

# HuggingFace past_key_values type alias
# tuple of (K, V) per layer -> it refers to transformer layers
# each K or V: (batch_size, num_heads, seq_len, head_dim)
PastKeyValues = Cache

@dataclass
class KVCache:
    """
    Wraps HuggingFace past_key_values with a clean interface.

    Responsibilities:
        - Store K and V tensors from each attention layer
        - Track sequence length seen so far
        - Provide clean access for the runtime

    This is intentionally a thin wrapper.
    Phase 4 (PagedAttention) will replace this with block-based management.
    """
    past_key_values: PastKeyValues
    seq_len: int # total tokens processed so far (prompt + generated)

    @classmethod
    def from_prefill(
        cls,
        past_key_values: PastKeyValues,
        prompt_len: int,
    )-> KVCache:
        """
        Create a KVCache from the result of a prefill forward pass.

        Args:
            past_key_values: The past_key_values returned by the model.
            prompt_len: Number of tokens in the original prompt.
        """
        return cls(
            past_key_values=past_key_values,
            seq_len = prompt_len,
        )

    def update(self, new_past_key_values: PastKeyValues)-> KVCache:
        """
        Return a new KVCache with updated tensors after a decode step.

        We return a new object rather than mutating in place.
        Immutability makes bugs easier to catch during development.

        In Phase 4, this will become in-place block updates.
        """
        return KVCache(
            past_key_values = new_past_key_values,
            seq_len = self.seq_len + 1
        )

    @property
    def num_layers(self) -> int:
        """Number of attention layers in the cache."""
        return len(self.past_key_values)

    def validate(self) -> None:
        """
        Sanity check cache structure.
        Useful during development to catch shape bugs early.
        """
        for layer_idx, (k,v) in enumerate(self.past_key_values):
            if k.shape != v.shape:
                raise KVCacheNotInitializedError(
                    f'Layer {layer_idx}: K shape {k.shape} != V shape {v.shape}'
                )
            if k.shape[2] != self.seq_len:
                raise KVCacheNotInitializedError(
                    f'Layer {layer_idx}: cache seq_len {k.shape[2]}'
                    f'!= tracked seq_len {self.seq_len}'
                )


