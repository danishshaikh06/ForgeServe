import torch

from forgeserve.logger import get_logger

from forgeserve.sampler.base import Sampler
from forgeserve.sampler.exception import InvalidLogitsShapeError

logger = get_logger(__name__)


class GreedySampler(Sampler):
    """
    Greedy sampling strategy.

    Selects the token with the maximum logit score.
    """

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:
                Tensor of shape (batch_size, vocab_size)

        Returns:
            Tensor containing selected token ids.
        """

        if logits.ndim != 2:
            raise InvalidLogitsShapeError(
                f"Expected logits with shape (batch_size, vocab_size). "
                f"Received {tuple(logits.shape)}."
            )

        logger.debug(
            "Selecting next token using greedy decoding. Shape=%s",
            tuple(logits.shape),
        )

        token_ids = torch.argmax(logits, dim=-1)

        logger.debug("Greedy sampling completed.")

        return token_ids