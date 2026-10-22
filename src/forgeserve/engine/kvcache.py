
from __future__ import annotations

import time

import torch

from forgeserve.engine.config import GenerationConfig
from forgeserve.engine.exception import GenerationException
from forgeserve.engine.response import GenerationResponse
from forgeserve.logger import get_logger
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.base import Sampler

logger = get_logger(__name__)


class KVCacheGenerationEngine:
    """Generate text autoregressively using a KV cache."""

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
        """Generate a continuation for a single input prompt."""

        logger.info("Starting text generation using KV cache")
        start = time.perf_counter()

        if config.max_new_tokens <= 0:
            logger.warning(
                "max_new_tokens is %d; returning empty response",
                config.max_new_tokens,
            )
            return GenerationResponse(
                text="",
                generated_tokens=0,
                finish_reason="length",
            )

        try:
            encoded = self.runtime.tokenize(
                prompt,
                config.system_prompt,
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # This implementation deliberately supports one sequence.
            if input_ids.shape[0] != 1:
                raise ValueError(
                    "KVCacheGenerationEngine currently supports "
                    "batch size 1 only"
                )

            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask and input_ids must have identical shapes"
                )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            self.runtime.log_gpu_memory("Before prefill")

            # Process the prompt once.
            logits, cache = self.runtime.prefill(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            self.runtime.log_gpu_memory("After prefill")

            generated_ids: list[int] = []
            generated = 0
            is_eos = False

            for step in range(config.max_new_tokens):
                # Sample one next token.
                next_token = self.sampler.sample(logits)

                if next_token.ndim != 1 or next_token.shape[0] != 1:
                    raise ValueError(
                        "Expected sampler output with shape (1,), "
                        f"got {tuple(next_token.shape)}"
                    )

                generated += 1

                # Count EOS, but do not include it in the response text.
                if self._should_stop(next_token):
                    is_eos = True
                    logger.debug("EOS token encountered")
                    break

                generated_ids.append(int(next_token[0].item()))

                # Do not run another forward pass if the token budget
                # has been exhausted.
                if step == config.max_new_tokens - 1:
                    break

                # The cache currently contains all tokens processed so far.
                # Extend the mask for the token about to be processed.
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=-1,
                )

                # Process only the newly sampled token.
                logits, cache = self.runtime.decode_step(
                    token_id=next_token.unsqueeze(-1),
                    attention_mask=attention_mask,
                    cache=cache,
                )

                if generated % 10 == 0:
                    self.runtime.log_gpu_memory(
                        f"After {generated} generated tokens"
                    )

            # Decode only the continuation, not the original prompt.
            text = self.runtime.decode(generated_ids)

            elapsed = time.perf_counter() - start

            logger.info(
                "Generation completed in %.3f seconds; "
                "generated_tokens=%d; finish_reason=%s",
                elapsed,
                generated,
                "eos" if is_eos else "length",
            )

            return GenerationResponse(
                text=text,
                generated_tokens=generated,
                finish_reason="eos" if is_eos else "length",
            )

        except Exception as exc:
            logger.exception("Generation failed")
            raise GenerationException(
                "Text generation failed"
            ) from exc


    def _should_stop(
        self,
        token: torch.Tensor,
    ) -> bool:
        """Return True when the single generated sequence emits EOS."""

        eos_token_id = self.runtime.tokenizer.eos_token_id

        if eos_token_id is None:
            return False

        if isinstance(eos_token_id, int):
            eos_ids = [eos_token_id]
        else:
            eos_ids = list(eos_token_id)

        if not eos_ids:
            return False

        eos_tensor = torch.tensor(
            eos_ids,
            dtype=token.dtype,
            device=token.device,
        )

        return bool(torch.isin(token, eos_tensor).all())
