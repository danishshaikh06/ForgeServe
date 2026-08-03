from abc import ABC, abstractmethod

import torch


class Sampler(ABC):
    """Abstract interface for sampling strategies."""

    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Select the next token from logits.

        Args:
            logits: Tensor of shape (batch_size, vocab_size)

        Returns:
            Tensor containing token ids.
        """
        raise NotImplementedError
