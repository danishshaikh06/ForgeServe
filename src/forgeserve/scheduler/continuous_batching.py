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