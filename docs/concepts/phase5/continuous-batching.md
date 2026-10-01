# Continuous Batching

## The Problem

Suppose requests arrive at the same time:

```text
Request A
Request B
Request C
Request D
```

A sequential inference engine does:

```text
A → generate everything → finish
B → generate everything → finish
C → generate everything → finish
D → generate everything → finish
```

The GPU processes one request's decode work at a time.

Continuous batching changes the unit of scheduling.

Instead of scheduling a request for its entire lifetime, the scheduler repeatedly chooses the currently active requests for the next decode step.

```text
step 1 → [A, B, C, D]
step 2 → [A, B, C, D]
step 3 → [A, B, D]
step 4 → [A, B, D, E]
...
```

Requests can leave when they finish, and waiting requests can enter without waiting for the whole original group to complete.

## The Core Definition

Let:

\[
\mathcal{B}_t
\]

be the set of requests processed at decode step \(t\).

Continuous batching allows:

\[
\boxed{
\mathcal{B}_t \neq \mathcal{B}_{t+1}
}
\]

when requests finish or new requests are admitted.

This is the main difference from a fixed static batch.

## Request Lifecycle

A simple request state machine is:

```text
WAITING
   ↓
RUNNING
   ↓
FINISHED
```

The request owns its KV blocks while it is RUNNING.

When it finishes:

```text
RUNNING
   ↓
FINISHED
   ↓
release KV blocks
```

## Example

Suppose:

```text
A → 10 generated tokens
B → 100 generated tokens
C → 20 generated tokens
D → 50 generated tokens
```

The initial batch is:

```text
[A, B, C, D]
```

After ten decode steps, A finishes:

```text
[B, C, D]
```

After twenty decode steps, C finishes:

```text
[B, D]
```

A waiting request E can then enter:

```text
[B, D, E]
```

The long-running requests never had to restart because another request finished.

## Scheduler Responsibilities

The scheduler decides:

- which requests are waiting
- which requests are running
- which requests may enter
- which requests participate in the next decode step
- which requests have finished
- when their KV blocks are released

A useful architectural split is:

```text
Scheduler
    → who runs and when

Runtime
    → how the model executes

BlockManager
    → where KV memory lives
```

## Continuous Batching Is Not Automatically Parallel Execution

There are two separate ideas:

```text
multiple requests are active
```

and:

```textone model forward processes multiple requests
```

The first is request scheduling.

The second is batched forward execution.

ForgeServe Phase 5B combines both.

## Why It Improves Throughput

If four active requests each need one decode token, sequential execution performs four separate forward calls:

```text
forward(A)
forward(B)
forward(C)
forward(D)
```

Batched execution performs:

```text
forward([A, B, C, D])
```

This lets the GPU process a larger operation and share model execution across requests.

## The Important Trade-off

Continuous batching is primarily useful for improving system-level efficiency.

It does not guarantee that every individual request has lower end-to-end latency.

A request can wait for admission:

```text
arrival
  ↓
WAITING
  ↓
admission
  ↓
RUNNING
```

It can also spend longer in a shared schedule.

Therefore ForgeServe benchmarks both:

\[
\text{throughput}
\]

and:

\[
\text{latency}
\]

## Mental Model

Think of continuous batching as a moving group:

```text
time →

[A B C D]
[A B C D]
[A B   D]
[A B   D E]
[  B   D E]
[  B     E F]
...
```

The group changes continuously while generation is in progress.
