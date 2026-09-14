"""
Continuous-batching scheduler for ForgeServe Phase 5.

What is continuous batching?
-----------------------------
In naive batching, the server waits until a fixed batch of requests
arrives, processes them all together, and only then accepts new work.
Every slot in the batch is held for the full generation length of the
*longest* request.  Short requests finish early but their GPU slot
sits idle until the batch completes.  This wastes throughput.

Continuous batching (also called *iteration-level scheduling*) fixes
this by operating at the **token** level instead of the request level:

    1. Every *step*, every running request decodes exactly one token.
    2. As soon as a request finishes, its KV blocks are freed.
    3. Freed blocks immediately allow a waiting request to be admitted.
    4. The batch composition changes every step — it is *continuous*.

This means short requests vacate their slot immediately, long requests
coexist with newly admitted short ones, and GPU utilisation stays
high throughout.

Scheduler design
----------------
This implementation is intentionally minimal:

* **FIFO admission** — requests are admitted in arrival order.
* **No preemption** — a running request is never evicted.
* **No priority** — all requests are equal.
* **No batching heuristic** — batch size is bounded only by the KV
  block pool and an optional ``max_batch_size`` cap.

These omissions are deliberate.  Each one is a separate concern
(preemption, priority scheduling, dynamic batching) that belongs in a
later phase.  The goal here is to make the continuous-batching loop
itself as clear as possible.

Key invariant
-------------
At the end of every ``step()`` call:

    * Every request that was RUNNING at the start of the step has
      consumed exactly one token of decode budget.
    * Any request that reached EOS or ``max_new_tokens`` during that
      step has had its KV blocks freed and its status set to FINISHED.
    * The waiting queue has been re-evaluated: requests that can now
      be admitted (because blocks were freed) have been prefilled and
      moved to RUNNING.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch 

from forgeserve.logger import get_logger
from forgeserve.model.paged_runtime import PagedRuntime
from forgeserve.sampler import Sampler
from forgeserve.scheduler.request import RequestState,RequestStatus
from forgeserve.page_attention.block_manager import BlockManager
if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

class ContinuousBatching:
    """
    Iteration-level continuous-batching scheduler.

    Parameters
    ----------
    runtime:
        A fully initialised ``PagedRuntime`` with an attached
        ``BlockManager``.  The scheduler calls ``paged_prefill``,
        ``paged_decode_step``, and ``free_request`` on it.
    sampler:
        Token-sampling strategy.  Injected rather than hardcoded so
        the caller can swap greedy for top-p without touching the
        scheduler.
    max_batch_size:
        Optional hard cap on simultaneous running requests.
        ``None`` means the only limit is the KV block pool.

    Attributes
    ----------
    _waiting:
        FIFO deque of requests that have been submitted but not yet
        admitted.  Only the front of the queue is ever inspected for
        admission; if the front cannot be admitted (insufficient
        blocks), admission stops entirely to preserve FIFO order and
        prevent starvation.
    _running:
        Dict mapping ``request_id`` → ``RequestState`` for every
        currently active request.  Dict iteration order is insertion
        order in CPython 3.7+, which gives deterministic decode order.
    """

    def __init__(
            self,
            runtime: PagedRuntime,
            sampler: Sampler,
            max_batch_size: int | None = None,
    ) -> None:
        self.runtime = runtime
        self.sampler = sampler 
        self.max_batch_size = max_batch_size

        self._waiting: deque[RequestState] = deque()
        self._running: dict[str, RequestState] = {}

        logger.info(
            "ContinuousBatchScheduler initialised: "
            "max_batch_size=%s total_blocks=%d block_size=%d",
            max_batch_size,
            self.block_manager.num_blocks,
            self.block_manager.block_size,
        )

    @property
    def waiting(self) -> tuple[RequestState, ...]:
        """Snapshot of the waiting queue (oldest first)."""
        return tuple(self._waiting)

    @property
    def running(self) -> tuple[RequestState, ...]:
        """Snapshot of currently running requests."""
        return tuple(self._running.values())

    @property
    def num_waiting(self) -> int:
        """Number of requests waiting for KV blocks."""
        return len(self._waiting)

    @property 
    def num_running(self) -> int:
        """Number of requests currently decoding."""
        return len(self._running)

    @property
    def block_manager(self, ) -> int:
        """Convenience accessor for the runtime's block manager."""
        return self.runtime._require_block_manager()

    #add request in the waiting queue 
    def add_request(
            self,
            request_id: str,
            prompt: str,
            max_new_tokens: int
    ) -> RequestState:
        """
        Submit a new request to the waiting queue.

        No KV blocks are allocated here.  Allocation happens inside
        _prefill when the request reaches the front of the queue
        and sufficient blocks are available.

        Parameters
        ----------
        request_id:
            Caller-supplied unique identifier.  Must not already exist
            in the waiting queue or the running set.
        prompt:
            Raw text to condition generation on.
        max_new_tokens:
            Maximum tokens to generate. Must be positive.

        Returns
        -------
        RequestState
            The newly created state object.  The caller may hold a
            reference to monitor progress, but must not mutate it.

        Raises
        ------
        ValueError
            If ``request_id`` is already present or ``max_new_tokens``
            is not positive.
        """
        if request_id in self._running:
            raise ValueError(
                f"Request '{request_id}' is already running."
            )

        if any(r.request_id == request_id for r in self._waiting):
            raise ValueError(
                f"Request '{request_id}' is already in the waiting queue."
            )

        if max_new_tokens<= 0:
            raise ValueError(
                f"max_new_tokens must be > 0, got {max_new_tokens}."
            )
        
        request = RequestState(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        self._waiting.append(request)

        logger.info(
            "Request queued: id=%s max_new_tokens=%d "
            "waiting=%d",
            request_id,
            max_new_tokens,
            self.num_waiting,
        )

    #add request in the running dictionary from waiting queue 
    def admit_request(self) -> int:
        """
        Promote waiting requests to RUNNING as blocks permit.

        Admission is **FIFO with a hard front-of-queue block**.
        If the oldest waiting request cannot be admitted (not enough
        free blocks), admission stops immediately.  We never skip
        ahead to a shorter request because doing so would starve long
        requests indefinitely — a classic head-of-line blocking problem
        in reverse.

        Each admitted request is prefilled synchronously before the
        next candidate is inspected.  This means the block count is
        updated after every admission, giving an accurate picture of
        available capacity.

        Returns
        -------
        int
            Number of requests newly admitted this call.
        """
        admitted = 0 

        while self._waiting:
            if(
                self.max_batch_size is not None 
                and self.num_running >= self.max_batch_size
            ):
                break 

            candidate = self._waiting[0]

            if not self._can_admit(candidate):
                # Front of queue cannot be admitted.
                # Stop — do not skip to smaller requests (prevents starvation).
                break

            self._waiting.popleft()
            self._prefill(candidate)
            candidate.mark_running()
            self._running[candidate.request_id] = candidate
            admitted+=1

            logger.info(
                "Request admitted: id=%s prompt_tokens=%d "
                "blocks_used=%d running=%d waiting=%d "
                "free_blocks=%d",
                candidate.request_id,
                candidate.prompt_tokens,
                candidate.blocks_used,
                self.num_running,
                self.num_waiting,
                self.block_manager.num_free_blocks,
            )

        return admitted   

    def _can_admit(self, request: RequestState) -> bool:
        """
        Return ``True`` if the block pool can service this request's
        prompt without exceeding capacity.

        We reserve only the prompt KV footprint here.  Decode blocks
        are allocated on demand inside ``paged_decode_step`` as
        generation crosses block boundaries.  This minimises upfront
        reservation and avoids over-provisioning for short generations.

        Parameters
        ----------
        request:
            The candidate request at the front of the waiting queue.
        """
        prompt_tokens = self._count_prompt_tokens(request.prompt)
        blocks_needed = self._blocks_required(prompt_tokens)
        return self.block_manager.allocate(blocks_needed)

