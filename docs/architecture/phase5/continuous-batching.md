# Continuous Batching

## Overview

Continuous batching maintains a dynamic set of running requests instead of completing requests one at a time.

```text
[A, B, C, D]
[A, B, C, D]
[A, B, D]
[A, B, D, E]
...
```

Requests can finish independently and waiting requests can enter without stopping the other active requests.

## Request Lifecycle

```text
WAITING
   ↓
RUNNING
   ↓
FINISHED
```

A cancelled or failed request can also leave the running set.

KV blocks remain owned while the request is running and are released when the request finishes.

## Scheduler Responsibilities

The first ForgeServe scheduler is intentionally simple. It handles:

1. request submission
2. FIFO waiting-queue management
3. admission into the running set
4. decode-step selection
5. completion detection
6. KV-block release

The responsibilities are separated as:

```text
Scheduler    → who runs / when
Runtime      → how the model executes
BlockManager → where KV memory lives
```

## Batch Size

Let `B_max` be the maximum number of requests in one forward batch:

\[
|\mathcal{B}_t|\le B_{max}
\]

Batch size is a scheduling constraint. It is not necessarily equal to the number of resident requests.

For example:

```text
12 resident requests
max forward batch = 4
```

means at most four participate in one model forward at a time.

## Dynamic Membership

The batch may change every decode step:

```text
Step 1 → [A, B, C, D]
Step 2 → [A, B, C, D]
C finishes
Step 3 → [A, B, D, E]
A finishes
Step 4 → [B, D, E, F]
```

## Memory Admission

If the pool has `N` blocks and active requests own `B_used` blocks:

\[
B_{free}=N-B_{used}
\]

A waiting request may be admitted only when the required initial allocation is available.

## Completion

When a request finishes:

```text
RUNNING
   ↓
FINISHED
   ↓
release owned blocks
   ↓
free capacity becomes available
```

That capacity can be used to admit waiting requests.

## Trade-off

Continuous batching primarily improves system-level utilization and throughput. It does not guarantee lower per-request latency. Queue waiting and shared execution can increase individual latency, especially with heterogeneous request lengths or a restrictive batch cap.

Important metrics include:

- total wall time
- token throughput
- request throughput
- mean/P50/P95/P99 latency
- queue wait
- average batch size

## Initial ForgeServe Policy

The first policy is:

```text
FIFO admission
+
dynamic running set
+
one decode step per active request
+
immediate cleanup on completion
```

Future work can explore adaptive batch sizing, token/compute budgets, aging, and resource-aware admission.
