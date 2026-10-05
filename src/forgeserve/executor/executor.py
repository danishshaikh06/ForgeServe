from __future__ import annotations

import torch

from forgeserve.logger import get_logger
from forgeserve.scheduler.request import RequestState
from forgeserve.model.paged_runtime import PagedRuntime
from forgeserve.sampler.greedy import GreedySampler
from forgeserve.kv_cache.exception import KVCacheOutOfMemoryError


logger = get_logger(__name__)

class Executor:
    """
    Defines the execution boundary between the scheduler/engine
    and the model runtime.
    """
    def __init__(
            self,
            runtime: PagedRuntime,
            sampler: GreedySampler,
    ):
        self.runtime = runtime
        self.sampler = sampler

    #Prefill
    def prefill(self,request: RequestState) -> None:
        """
        Tokenize the prompt and run the prefill forward pass.

        After this method returns:
        * ``request.input_ids`` and ``request.attention_mask`` hold
            the prompt tensors on the correct device.
        * ``request.paged_cache`` holds the populated ``PagedKVCache``.
        * ``request.logits`` holds the logits for the first decode step.
        * ``request.prompt_tokens`` is set.

        The request is not yet marked RUNNING.  That happens in
        ``admit_requests`` after this method returns successfully.

        Parameters
        ----------
        request:
            The request to prefill.  Must be in WAITING status.
        """
        encoded = self.runtime.tokenize(
            text=request.prompt,
            system_prompt=None,
        )

        input_ids: torch.Tensor = encoded["input_ids"].to(self.runtime.loader.device)
        attention_mask: torch.Tensor = encoded["attention_mask"].to(self.runtime.loader.device)

        logits, paged_cache = self.runtime.paged_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            request_id=request.request_id,
        )

        request.input_ids = input_ids
        request.attention_mask = attention_mask
        request.paged_cache = paged_cache
        request.logits = logits
        request.prompt_tokens = input_ids.shape[1]

        logger.debug(
            "Prefill complete: id=%s prompt_tokens=%d blocks=%d",
            request.request_id,
            request.prompt_tokens,
            request.blocks_used,
        )

    def decode_batch(
        self,
        active_request:list[RequestState],
    ) -> None:
        """
        Decode one token for all active requests in a single GPU forward pass.

        This replaces the sequential _decode_one loop.
        All requests share one forward pass — weights loaded once for all.

        Args:
            active_requests: All currently RUNNING RequestState objects.
        """
        if not active_request:
            return

        # Sample next token for all requests
        # we need the last_token_id set on each request before batched_decode_step
        # Because batched_decode_step reads it to build the token batch tensor
        # Sample from current logits FIRST, Then run forward for the NEXT step
        for req in active_request:
            logits = req.logits #(1,vocab_size)
            next_token = self.sampler.sample(logits) # (1,)
            req.last_token_id = int(next_token.item())

        try:
            results = self.runtime.batched_decode_step(active_request)
        except KVCacheOutOfMemoryError as e:
            logger.warning(
                "OOM during batched decode: %s. "
                "Finishing affected requests.", e
            )

            #mark all req as oom-teminated
            for req in active_request:
                req.finish_reason = "oom"
                req.max_new_tokens = req.generated_tokens
            return

        # Update each request with its results
        for req, (logits,paged_cache) in zip(active_request, results, strict=True):
            req.logits = logits
            req.paged_cache = paged_cache
            req.generated_tokens+=1

            logger.debug(
                "Batched token decoded: id=%s token=%d generated=%d blocks=%d",
                req.request_id,
                req.last_token_id,
                req.generated_tokens,
                req.blocks_used,
            )

    def free_request(self, request_id: str) -> None:
        """
        Release all KV blocks owned by a request.
        """

        self.runtime.free_request(request_id)
