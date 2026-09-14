"""
Request lifecycle state for the continuous-batching scheduler.

Each generation request is represented by a single ``RequestState``
object.  The scheduler creates it on submission, updates it as the
request moves through WAITING → RUNNING → FINISHED, and reads it
during every decode step.

Design principles
-----------------
* ``RequestState`` is a **pure data container**.  It owns no scheduling
  logic, no sampling logic, and no GPU operations.  All decisions are
  made by ``ContinuousBatchScheduler``.
* Status transitions happen only through explicit methods so callers
  cannot set an illegal state directly.
* Timestamps are recorded automatically inside the transition methods
  so benchmarks never need to instrument the scheduler separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass,field
from enum import Enum,auto
from typing import TYPE_CHECKING
import torch 

if TYPE_CHECKING:  # only check during mypy checks dosent work at runtime 
    from forgeserve.page_attention.paged_cache import PagedKVCache

class RequestStatus(Enum):
     """
    Legal states for a generation request.

    Transitions
    -----------
    WAITING → RUNNING   when the scheduler admits the request and
                        prefill completes successfully.
    RUNNING → FINISHED  when EOS is produced or ``max_new_tokens``
                        is reached.
    WAITING → CANCELLED request was removed from the queue before
    RUNNING → CANCELLED admission, or aborted during decode.
    """
     WAITING = auto()
     RUNNING = auto()
     FINISHED = auto()
     CANCELLED = auto()

@dataclass
class RequestState:
    """
    All runtime state for one active generation request.

    The scheduler is the sole owner of this object. No other
    component should mutate it directly.

    *Parameters
    request_id:
        Unique identifier supplied by the caller.  Used as the key
        for KV-block ownership in ``BlockManager``.
    prompt:
        Raw text prompt.  Stored so the scheduler can re-tokenize
        if needed (e.g. after a preemption in future phases).
    max_new_tokens:
        Hard upper bound on generated tokens.  Generation always
        stops here even if EOS has not appeared.

    Mutable fields (set by the scheduler during the request lifetime)
    -----------------------------------------------------------------
    input_ids, attention_mask:
        Tensors on the target device. ``attention_mask`` is extended
        by one column after every decode step so the model always sees
        the correct full-sequence mask.
    paged_cache:
        The ``PagedKVCache`` holding this request's KV blocks.
        ``None`` before prefill and after the request is finished.
    logits:
        Raw logits from the most recent forward pass.
        Shape ``(1, vocab_size)``.  ``None`` until after prefill.
    last_token_id:
        The integer token id produced by the most recent decode step.
        Used by the scheduler for EOS detection.
    prompt_tokens:
        Number of tokens in the tokenised prompt.  Set once after
        prefill; used for block-count bookkeeping and metrics.
    generated_tokens:
        Running count of tokens produced so far.  Incremented by
        ``_decode_one`` on every step.
    finish_reason:
        ``"eos"`` if an EOS token was produced, ``"length"`` if
        ``max_new_tokens`` was exhausted.  Empty string until the
        request reaches FINISHED.

    Timestamp fields (set automatically by transition methods)
    ----------------------------------------------------------
    created_at:   perf_counter when the object was constructed.
    started_at:   perf_counter when ``mark_running`` was called.
    finished_at:  perf_counter when ``mark_finished`` was called.

    These three timestamps give you TTFT and total-latency for free
    without any external instrumentation.
    """
    #Params
    request_id: str
    prompt: str
    max_new_tokens: int

    # Set by the scheduler during prefill 
    input_ids: torch.Tensor | None = None 
    attention_mask: torch.Tensor | None = None 
    paged_cache: PagedKVCache | None = None 
    logits: torch.Tensor | None = None 
    last_token_id: int | None = None 

    #used to calculate no of blocks 
    prompt_tokens: int = 0 
    generated_tokens: int = 0 

    status: RequestStatus= RequestStatus.WAITING
    finish_reason: str =""

    created_at: float = field(default_factory=time.perf_counter)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def total_tokens(self) -> int:
        """
        Total tokens currently represented by this request.

        Equals prompt tokens plus all tokens generated so far.
        Useful for block-count estimates and logging.
        """
        return self.prompt_tokens + self. generated_tokens

    @property
    def is_finished(self) -> bool:
        """``True`` once the request has reached FINISHED or CANCELLED."""
        return self.status in (
            RequestStatus.FINISHED,
            RequestStatus.CANCELLED,
        )

    @property
    def blocks_used(self) -> int:
        """
        Number of KV blocks currently allocated to this request.

        The ``PagedKVCache`` is the single source of truth for block
        ownership.  The scheduler never tracks this independently to
        avoid the two-source-of-truth problem.

        Returns 0 before prefill and after the cache is released.
        """
        if self.paged_cache is None:
            return 0 
        return len(self.paged_cache.block_table)

    @property
    def time_to_first_token_ms(self) -> float | None:
        """
        Elapsed milliseconds from submission to first token.

        Returns ``None`` if the request has not yet been admitted.
        """
        if self.started_at is None:
            return 0
        return (self.created_at - self.started_at) * 1_000 

    @property
    def total_latency_ms(self) -> int:
        """
        Elapsed milliseconds from submission to completion.

        Returns ``None`` if the request has not yet finished.
        """
        if self.finished_at is None:
            return 0
        return (self.finished_at - self.created_at) * 1_000

    #status 
    def mark_running(self) -> None:
        """
        Transition WAITING → RUNNING.

        Records ``started_at`` for TTFT measurement.

        Raises
        ------
        RuntimeError
            If the request is not currently WAITING.
        """
        if self.status != RequestStatus.WAITING:
            raise RuntimeError(
                f"Cannot mark request '{self.request_id}' as RUNNING "
                f"from status {self.status.name}."
            )

        self.status = RequestStatus.RUNNING
        self.started_at = time.perf_counter()

    def mark_finished(self) -> None:
        """
        Transition RUNNING → FINISHED.

        Records ``finished_at`` for total-latency measurement.

        Raises
        ------
        RuntimeError
            If the request is not currently RUNNING.
        """
        if self.status != RequestStatus.RUNNING:
            raise RuntimeError(
                f"Cannot mark request '{self.request_id}' as FINISHED "
                f"from status {self.status.name}."
            )
        self.status = RequestStatus.FINISHED
        self.finished_at = time.perf_counter()

    def mark_cancelled(self) -> None:
        """
        Transition any non-terminal state → CANCELLED.

        Records ``finished_at`` so latency metrics remain consistent
        even for cancelled requests.
        """
        if self.status in (
            RequestStatus.CANCELLED,
            RequestStatus.FINISHED
        ):
            raise RuntimeError(
                f"Cannot cancel request '{self.request_id}' "
                f"that is already {self.status.name}."
            )
        self.status = RequestStatus.CANCELLED
        self.finished_at = time.perf_counter()

    def __repr__(self) -> str:
        return(
            f"RequestState("
            f"id={self.request_id!r}, "
            f"status={self.status.name}, "
            f"prompt_tokens={self.prompt_tokens}, "
            f"generated={self.generated_tokens}/{self.max_new_tokens}, "
            f"blocks={self.blocks_used}"
            f")"
        )
        
         



     


