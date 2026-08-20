"""
PagedGenerationEngine: Generation using PagedAttention KV cache.

External API is identical to GenerationEngine.
Internally uses PagedRuntime for block-based KV storage.

Key addition: request_id parameter for block tracking.
Every request must have a unique ID so blocks can be freed
when generation completes.
"""
from __future__ import annotations

import time 
import uuid

import torch 

from forgeserve.engine.exception import GenerationException
from forgeserve.engine.response import GenerationResponse
from forgeserve.engine.config import GenerationConfig
from forgeserve.page_attention.exception import KVCacheOutOfMemoryError
from forgeserve.logger import get_logger
from forgeserve.model.paged_runtime import PagedRuntime
from forgeserve.sampler.base import Sampler

logger = get_logger(__name__)

class PagedGenerationEngine:
    """
    Generation engine using PagedAttention for KV cache management.

    Args:
        runtime: PagedRuntime with block manager initialized.
        sampler: Sampling strategy.
    """
    def __init__(
            self,
            runtime: PagedRuntime,
            sampler: Sampler,
    ) -> None:
        self.runtime = runtime
        self.sampler = sampler 

    def generate(
            self,
            prompt: str,
            config: GenerationConfig,
            request_id: str | None = None,
    ) -> GenerationResponse:
        """
        Generate text using paged KV cache.

        Args:
            prompt:     Input text.
            config:     Generation configuration.
            request_id: Optional unique ID. Auto-generated if None.
                        Used to track and free KV blocks after generation.

        Returns:
            GenerationResponse with text, token count, finish reason.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]

        logger.info(
            "Starting paged generation: request=%s max_tokens=%d",
            request_id, config.max_new_tokens,
        )

        start = time.perf_counter()

        try:
            encoded = self.runtime.tokenize(
                text= prompt,
                system_prompt=config.system_prompt
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            try:
                # prefill 
                logits, paged_cache = self.runtime.paged_prefill(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    request_id=request_id,
                )

            except KVCacheOutOfMemoryError as e:
                logger.warning(
                    "OOM during prefill for request '%s': %s", request_id, e
                )
                raise

            generated = 0 
            is_eos = False

            # Decode loop
            for _ in range (config.max_new_tokens):
                next_token = self.sampler.sample(logits)
                generated+=1

                is_eos = self._should_stop(next_token)

                if is_eos:
                    logger.debug(
                        "EOS at request='%s' position=%d",
                        request_id, generated,
                    )
                    break 

                input_ids, attention_mask = self._append_token(
                    input_ids, attention_mask, next_token
                )

                try:
                    logits, paged_cache = self.runtime.paged_decode_step(
                        token_id=next_token.unsqueeze(-1),
                        attention_mask=attention_mask,
                        paged_cache=paged_cache,
                    )
                except KVCacheOutOfMemoryError as e:
                    logger.warning(
                        "OOM during decode at position %d for request '%s': %s",
                        generated, request_id, e,
                    )
                    raise

            text = self.runtime.decode(input_ids[0])
            elapsed = time.perf_counter() - start

            logger.info(
                "Paged generation complete: request=%s tokens=%d "
                "time=%.3fs blocks_used=%d",
                request_id, generated, elapsed,
                len(paged_cache.block_table),
            )

            return GenerationResponse(
                text = text,
                generated_tokens=generated,
                finish_reason="eos" if is_eos else "length"
            )

        except Exception as exc:
            logger.exception(
                "Generation failed for request '%s'", request_id
            )
            raise GenerationException(
                f"Paged generation failed for request '{request_id}'"
            ) from exc

        finally:
            # Always Free the blocks, even if generation fails that is why it is in this block 
            # Failure to free = permanent gpu memory leak
            try:
                self.runtime.free_request(request_id)
                logger.debug(
                    "Blocks freed for request '%s'", request_id
                )
            except Exception:
                logger.warning(
                    "Failed to free blocks for request '%s'. "
                    "Memory may be leaked.", request_id
                )

    def _append_token(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            next_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append the generated token to the current sequence.
        """
        next_token = next_token.unsqueeze(-1)
        input_ids = torch.cat((input_ids, next_token), dim = -1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (attention_mask.size(0), 1),
                    dtype= attention_mask.dtype,
                    device= attention_mask.device,
                ),
            ),
            dim = -1 
        )

        return input_ids, attention_mask

    def _should_stop(self, token: torch.Tensor) -> bool:
        """
        Stop generation when the EOS token is produced.
        """
        eos = self.runtime.tokenizer.eos_token_id
        return bool((token == eos).all())





