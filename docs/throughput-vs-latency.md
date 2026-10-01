# Throughput vs Latency

## Two Different Questions

Inference performance has at least two important dimensions.

### Throughput

How much work does the system complete per unit time?

For token throughput:

\[
\boxed{
throughput
=
\frac{\text{generated tokens}}
{\text{total time}}
}
\]

For request throughput:

\[
\boxed{
request\ throughput
=
\frac{\text{completed requests}}
{\text{total time}}
}
\]

### Latency

How long does one request take?

This can be measured using:

```text
mean
P50
P95
P99
```

Queue waiting time is also important for a serving system.

## Why Batching Can Improve Throughput

Suppose four requests each need 80 tokens.

Sequential execution:

```text
forward(A)
forward(B)
forward(C)
forward(D)
```

Batched execution:

```text
forward([A,B,C,D])
```

If the GPU processes the combined work more efficiently, more tokens are completed per unit time.

## Why Individual Latency Can Increase

Now suppose a system has:

```text
12 requests
max batch size = 4
```

Eight requests may initially wait.

A request could therefore spend substantial time in:

```text
WAITING
```

before it becomes RUNNING.

Even though the entire workload finishes faster, some individual requests can experience higher latency.

## ForgeServe Example

In the measured 12-request queue-pressure workload:

```text
Sequential throughput = 22.8 tok/s
Batched throughput    = 60.2 tok/s
```

Token-throughput gain:

```text
2.64×
```

Wall-time speedup:

```text
2.75×
```

But mean latency changed from:

```text
1410.8 ms
```

to:

```text
3290.2 ms
```

and mean queue wait was:

```text
1663.3 ms
```

So the system became substantially more throughput-efficient while individual requests experienced more waiting.

## Why This Matters

A scheduler should not optimize only:

\[
\max(\text{throughput})
\]

A useful serving objective is closer to:

\[
\boxed{
\text{high throughput}
+
\text{controlled latency}
}
\]

The correct balance depends on the workload.

## Uniform vs Mixed Workloads

Uniform workloads often make batching look very strong.

If every request requires the same number of decode steps:

```text
[A B C D]
[A B C D]
[A B C D]
...
```

the batch stays full.

In a mixed workload:

```text
[A B C D]
[A B D]
[B D]
[B D E]
```

the average batch size falls as short requests finish.

This can reduce the throughput gain while introducing queueing effects.

## Metrics to Record

A continuous-batching benchmark should measure:

```text
wall time
token throughput
request throughput
mean latency
P50 latency
P95 latency
P99 latency
queue wait
average batch size
total generated tokens
```

These metrics describe both system efficiency and user-visible performance.
