# Phase 4 Architecture: Iteration-Level Scheduler

## Overview

The `Scheduler` manages sequence queues, sequence lifecycles, block allocation budgets, and selects which requests are executed during each iteration step of the engine.

---

## State Architecture & Queues

```
+-------------------------------------------------------------------+
|                           Scheduler                               |
|                                                                   |
|   +-----------------------+             +----------------------+  |
|   |     Waiting Queue     |             |    Running Queue     |  |
|   | [Seq A, Seq B, ...]   |             |  [Seq C, Seq D]      |  |
|   +-----------------------+             +----------------------+  |
|               |                                    |              |
|               v (Allocate Blocks)                  v              |
|   +------------------------------------------------------------+  |
|   |                       Swapped Queue                        |  |
|   |                 [Seq E (Preempted to CPU)]                 |  |
|   +------------------------------------------------------------+  |
+-------------------------------------------------------------------+
```

### Sequence States

1. **`WAITING`**: Request submitted by client, awaiting prompt prefill block allocation.
2. **`RUNNING`**: Active in GPU memory, currently generating tokens iteration by iteration.
3. **`SWAPPED`**: Preempted due to GPU memory pressure; blocks temporarily moved to CPU RAM.
4. **`FINISHED`**: Execution completed (EOS token reached or `max_tokens` limit hit).

---

## Scheduling Algorithm (Step Loop)

At each step, `Scheduler.schedule()` executes the following logic:

```python
def schedule(self) -> SchedulerOutputs:
    # 1. First: Try to schedule RUNNING sequences (Decode step)
    running_scheduled = []
    for seq in self.running:
        if self.block_manager.can_append_slot(seq):
            self.block_manager.append_slot(seq)
            running_scheduled.append(seq)
        else:
            # Preemption required if out of memory
            self._preempt(seq)

    # 2. Second: If memory budget permits, schedule WAITING sequences (Prefill step)
    budget = self.max_num_batched_tokens - num_running_tokens
    while self.waiting:
        seq = self.waiting[0]
        if self.block_manager.can_allocate_prompt(seq) and seq.num_tokens <= budget:
            self.waiting.pop(0)
            self.block_manager.allocate_prompt(seq)
            self.running.append(seq)
            budget -= seq.num_tokens
        else:
            break

    return SchedulerOutputs(scheduled_sequences=running_scheduled + self.running)
```

---

## Preemption Strategies

When GPU memory becomes full during a decode step:

1. **Recomputation Policy**: Free all physical blocks assigned to the sequence and put it back into `WAITING` state. When resumed, its prompt and tokens are recomputed.
2. **Swap Policy**: Move sequence KV cache blocks from GPU device memory to CPU host RAM. When GPU memory frees up, swap the blocks back.

---

## Data Structures

* `SequenceGroup`: Group of sequences sharing the same prompt (e.g. for parallel beam search).
* `Sequence`: Individual text generation stream containing prompt tokens, generated tokens, status, and assigned `BlockTable`.
* `SchedulerOutputs`: Decision payload containing sequences selected for prefill/decode in the current iteration step.
