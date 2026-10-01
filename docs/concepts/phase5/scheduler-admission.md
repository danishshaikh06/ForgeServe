# Scheduler Admission and Resource Limits

## What Is Admission?

Admission is the decision:

> Can this waiting request become RUNNING now?

A request is not admitted simply because it exists.

The scheduler must check whether the system has enough capacity.

## Simplest Admission Rule

Suppose the KV pool contains:

\[
N
\]

blocks.

Current used blocks:

\[
B_{used}
\]

Then:

\[
B_{free}=N-B_{used}
\]

If a new request needs \(B_{new}\) initial blocks, admit it only when:

\[
\boxed{
B_{new}\leq B_{free}
}
\]

## Example

Suppose:

```text
256 total blocks
200 currently used
```

Then:

\[
B_{free}=256-200=56
\]

A new request needs:

```text
12 blocks
```

Since:

\[
12\leq56
\]

it can be admitted.

## Memory vs Compute

Admission is not only about memory.

A scheduler may also have a maximum forward batch size:

\[
B_{max}
\]

Therefore a request can be blocked by either:

```text
memory capacity
```

or:

```text
batch/compute capacity
```

## Two Separate Limits

Think of:

\[
\boxed{
\text{KV capacity}
}
\]

and:

\[
\boxed{
\text{compute/batch capacity}
}
\]

For example:

```text
512 KV blocks
max forward batch = 8
```

The server may keep more than eight requests resident while executing at most eight in one model forward.

## FIFO Admission

A simple scheduler can maintain:

```text
waiting queue:
R1 → R2 → R3 → R4 → ...
```

and admit requests in arrival order.

This is easy to understand and provides a useful baseline.

## Why More Sophisticated Policies Exist

FIFO does not optimize every objective.

A short request behind long-running requests can wait a long time.

Possible future strategies include:

```text
FIFO
shortest-first
priority
aging
resource-aware admission
adaptive batch sizing
```

Each policy introduces trade-offs.

## Aging

An aging policy increases a request's effective priority as it waits.

Conceptually:

\[
priority_i
=
base_i+\alpha\cdot waiting\_time_i
\]

This prevents a request from waiting forever.

## Why ForgeServe Starts Simple

The educational scheduler should first make the basic state transitions correct:

```text
WAITING
   ↓
RUNNING
   ↓
FINISHED
```

and preserve the memory invariant:

\[
\boxed{
B_{used}+B_{free}=B_{total}
}
\]

Only after correctness is stable should more complex scheduling policies be added.
