from __future__ import annotations

import torch

from forgeserve.logger import get_logger
from forgeserve.scheduler.request import RequestState
from forgeserve.model.paged_runtime import PagedRuntime
from forgeserve.sampler.greedy import GreedySampler
from forgeserve.page_attention.exception import KVCacheOutOfMemoryError


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
        active_request: list[RequestState],
    ) -> None:
        """
        Generate one token for every active request.

        Optimization:
        Skip the next model forward pass for requests that
        have reached EOS or max_new_tokens.

        Scheduler remains responsible for finishing requests
        and releasing KV blocks.
        """

        if not active_request:
            return

        eos_id = self.runtime.tokenizer.eos_token_id

        # Only these requests need the next forward pass.
        continuing_requests: list[RequestState] = []

        # Sample the next token.
        for req in active_request:

            if req.logits is None:
                raise RuntimeError(
                    f"Request '{req.request_id}' has no logits."
                )

            next_token = self.sampler.sample(req.logits)

            req.last_token_id = int(next_token.item())
            req.generated_tokens += 1

            # First token is available immediately after sampling.
            if req.generated_tokens == 1:
                req.mark_first_token()

            # Check whether another forward is needed.
            eos_reached = (
                eos_id is not None
                and req.last_token_id == eos_id
            )

            length_reached = (
                req.generated_tokens >= req.max_new_tokens
            )

            if not (eos_reached or length_reached):
                continuing_requests.append(req)

            logger.debug(
                "Token sampled: id=%s token=%d generated=%d/%d "
                "eos=%s length_reached=%s",
                req.request_id,
                req.last_token_id,
                req.generated_tokens,
                req.max_new_tokens,
                eos_reached,
                length_reached,
            )

        skipped = len(active_request) - len(continuing_requests)

        logger.debug(
            "Decode filtering: active=%d continuing=%d skipped=%d",
            len(active_request),
            len(continuing_requests),
            skipped,
        )

        #  Nothing left to forward.
        if not continuing_requests:
            logger.debug(
                "Skipping GPU forward: all requests reached "
                "EOS or generation limit."
            )
            return

        # Forward only continuing requests.
        try:
            results = self.runtime.batched_decode_step(
                continuing_requests
            )

        except KVCacheOutOfMemoryError as exc:
            logger.warning(
                "KV-cache OOM during batched decode: %s. "
                "Terminating %d continuing requests.",
                exc,
                len(continuing_requests),
            )

            for req in continuing_requests:
                req.finish_reason = "oom"

            # Scheduler will finalize these requests and free KV.
            return

        # Update logits and caches.
        # Token counts were already updated during sampling.
        for req, (logits, paged_cache) in zip(
            continuing_requests,
            results,
            strict=True,
        ):
            req.logits = logits
            req.paged_cache = paged_cache

            logger.debug(
                "Next-token logits ready: id=%s generated=%d "
                "blocks=%d",
                req.request_id,
                req.generated_tokens,
                req.blocks_used,
            )
            
    def free_request(self, request_id: str) -> None:
        """
        Release all KV blocks owned by a request.
        """

        self.runtime.free_request(request_id)
