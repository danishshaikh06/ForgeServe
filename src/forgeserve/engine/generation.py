from __future__ import annotations

import time

import torch

from forgeserve.engine.exception import GenerationException
from forgeserve.engine.response import GenerationResponse
from forgeserve.logger import get_logger
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.base import Sampler
from forgeserve.engine.config import GenerationConfig

logger = get_logger(__name__)


class GenerationEngine:
    """
    Orchestrates autoregressive text generation.
    """

    def __init__(
        self,
        runtime: Runtime,
        sampler: Sampler,
    ) -> None:

        self.runtime = runtime
        self.sampler = sampler

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResponse:

        logger.info("Starting text generation")

        start = time.perf_counter()

        try:
            encoded = self.runtime.tokenize(prompt, "You are an ai inference model created by danish ")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            generated = 0

            for _ in range(config.max_new_tokens):
                output = self.runtime.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                assert output.logits is not None
                logits = output.logits[:, -1, :]

                next_token = self.sampler.sample(logits)

                input_ids, attention_mask = self._append_token(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    next_token=next_token,
                )

                generated += 1

                if self._should_stop(next_token):
                    logger.debug("EOS Token Encountered")
                    break

            text = self.runtime.decode(input_ids[0])
            end = time.perf_counter()

            time_taken = end - start

            logger.info("Generation completed in %.3f seconds", time_taken)

            return GenerationResponse(
                text=text,
                generated_tokens=generated,
                finish_reason="eos" if self._should_stop(next_token) else "length",
            )

        except Exception as exc:
            logger.exception("Generation Failed")
            raise GenerationException("Text generation failed") from exc

    def _append_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        next_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append the generated token to the current sequence.
        """

        next_token = next_token.unsqueeze(-1)  # shape was (1,dim) -> (1,1,dim)

        input_ids = torch.cat(
            (input_ids, next_token),
            dim=-1,
        )

        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=-1,
        )

        return input_ids, attention_mask

    def _should_stop(
        self,
        token: torch.Tensor,
    ) -> bool:
        """
        Stop generation when the EOS token is produced.
        """

        eos = self.runtime.tokenizer.eos_token_id

        return bool((token == eos).all())
