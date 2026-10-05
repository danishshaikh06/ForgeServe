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

from forgeserve.logger import get_logger
from forgeserve.model.paged_runtime import PagedRuntime
from forgeserve.sampler.greedy import GreedySampler
from forgeserve.scheduler.request import RequestState
from forgeserve.executor.executor import Executor

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

class ContinuousBatchScheduler:
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
            executor: Executor,
            max_batch_size: int | None = None,
    ) -> None:
        self.runtime = runtime
        self.executor = executor
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
    def block_manager(self):
        """Convenience accessor for the runtime's block manager."""
        return self.runtime.block_manager

    #add request in the waiting queue
    def add_request(
            self,
            request_id: str,
            prompt: str,
            max_new_tokens: int
    ) -> RequestState:
        """
        Submit a new request to the waiting queue.

        No KV blocks are allocated here. Allocation happens inside
        _prefill when the request reaches the front of the queue
        and sufficient blocks are available.

        Parameters
        ----------
        request_id:
            Caller-supplied unique identifier. Must not already exist
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

        return request

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
            self.executor.prefill(candidate)
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
        return self.block_manager.can_allocate(blocks_needed)

    #step
    def step(self) -> int:
        """
        Execute one continuous-batching scheduling step.
        Now uses batched forward pass instead of sequential loop.

        The step proceeds in four phases:

        1. **Admit** — move waiting requests to RUNNING if blocks allow.
        2. **Decode** — advance every running request by exactly one token.
        3. **Finish** — release blocks for requests that completed this step.
        4. **Re-admit** — newly freed blocks may allow waiting requests in.

        Phase 4 is the defining property of continuous batching: freed
        slots are reused *within the same scheduling step*, not deferred
        to the next one.

        Returns
        -------
        int
            Number of requests that participated in the decode phase.
            Zero means nothing is running and nothing could be admitted.
        """
        self.admit_request()

        if not self._running:
            return 0

        active = list(self._running.values())

        logger.debug(
            "Decode step: batch_size=%d free_blocks=%d",
            len(active),
            self.block_manager.num_free_blocks,
        )

        # ONE call handles all requests in one GPU forward pass
        self.executor.decode_batch(active)

        self._finish_completed_requests()

        # Re-admit immediately: blocks freed above may unblock the queue.
        self.admit_request()

        return len(active)

    #comletion
    def _finish_completed_requests(self) -> int:
        """
        Identify and release all requests that finished this step.

        A request is finished when either:
        * The most recently decoded token is the EOS token, or
        * ``generated_tokens`` has reached ``max_new_tokens``.

        Blocks are freed immediately so they can be reassigned to
        waiting requests within the same scheduling step.

        Returns
        -------
        int
            Number of requests finished and released this call.
        """

        completed = 0

        for _request_id,request in list(self._running.items()):
            eos_reached = self._is_eos(request)
            length_reached = request.generated_tokens>=request.max_new_tokens
            oom_reached = request.finish_reason == "oom"

            if not (eos_reached or length_reached or oom_reached):
                continue

            if not oom_reached:
                request.finish_reason = "eos" if eos_reached else "length"

            self._finish_request(request)
            completed+=1

        return completed

    def _finish_request(self, request:RequestState) -> None:
        """
        Release all KV blocks owned by a completed request and
        remove it from the running set.

        The ``finally``-style guarantee: blocks are always freed even if
        the request ended abnormally.  Failure to free blocks here is a
        permanent GPU memory leak — the pool shrinks with every request
        and the system eventually deadlocks.

        Parameters
        ----------
        request:
            A RUNNING request that has just finished generation.
        """
        blocks_released = request.blocks_used

        self.runtime.free_request(request.request_id)

        request.paged_cache = None
        request.logits = None
        request.mark_finished()

        del self._running[request.request_id]

        logger.info(
            "Request finished: id=%s generated=%d "
            "finish_reason=%s released_blocks=%d "
            "running=%d waiting=%d free_blocks=%d",
            request.request_id,
            request.generated_tokens,
            request.finish_reason,
            blocks_released,
            self.num_running,
            self.num_waiting,
            self.block_manager.num_free_blocks,
        )

    def _is_eos(self, request:RequestState) -> bool:
        """
        Return ``True`` if the last generated token is the EOS token.

        Parameters
        ----------
        request:
            The request to check.  If ``last_token_id`` is ``None``
            (before any decode step), returns ``False``.
        """
        if request.last_token_id is None:
            return False
        eos_id = self.runtime.tokenizer.eos_token_id
        return request.last_token_id == eos_id

    #helpers
    def _count_prompt_tokens(self, prompt: str) -> int:
        """
        Tokenize ``prompt`` and return the token count.

        Used only for the pre-admission block estimate.  The actual
        tokenization during prefill is done by ``_prefill`` which
        stores the tensors on the request.

        Parameters
        ----------
        prompt:
            Raw text prompt.
        """

        encoded = self.runtime.tokenize(text=prompt, system_prompt=None)
        return int(encoded["input_ids"].shape[1])

    def _blocks_required(self, tokens:int) -> int:
        """
        Return the number of KV blocks needed to hold ``tokens`` tokens.

        Uses ceiling division so a partial block is counted as a full one.

        Parameters
        ----------
        tokens:
            Number of tokens to store.
        """
        block_size = self.block_manager.block_size
        return (tokens + block_size - 1) // block_size

    #introspection
    def snapshot(self) -> dict[str,int]:
        """
        Return a point-in-time snapshot of scheduler and pool state.

        Intended for benchmarks, dashboards, and structured logging.
        All values are consistent with each other (no race conditions
        because this is single-threaded).

        Returns
        -------
        dict with keys:
            waiting       — requests in the waiting queue
            running       — requests currently decoding
            used_blocks   — KV blocks currently allocated
            free_blocks   — KV blocks available for allocation
            total_blocks  — sum of used and free (constant)
        """
        return{
            "waiting": self.num_waiting,
            "running": self.num_running,
            "used_blocks": self.block_manager.num_used_blocks,
            "free_blocks": self.block_manager.num_free_blocks,
            "total_blocks": self.block_manager.num_blocks,
        }

    def __repr__(self) -> str:
        return (
            f"ContinuousBatchScheduler("
            f"running={self.num_running}, "
            f"waiting={self.num_waiting}, "
            f"free_blocks={self.block_manager.num_free_blocks}/"
            f"{self.block_manager.num_blocks})"
        )













