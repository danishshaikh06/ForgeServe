# Batch Size vs Resident Requests

These two concepts are easy to confuse.

They are not the same.

## Batch Size

Batch size means:

> How many requests participate in one model forward pass?

If:

\[
B_{max}=4
\]

then:

\[
|\mathcal{B}_t|\leq4
\]

for every decode step.

Example:

```text
forward([A, B, C, D])
```

has batch size 4.

## Resident Requests

Resident requests are requests whose state is currently kept alive by the server.

For example:

```text
12 requests resident
```

does not necessarily mean:

```text
batch size = 12
```

You might instead have:

```text
12 resident requests
max forward batch = 4
```

The scheduler can choose four requests for the next forward.

## Why the Distinction Matters

Suppose:

```text
12 requests
max batch size = 4
```

Conceptually:

```text
Active requests:
A B C D E F G H I J K L
```

A decode step might use:

```text
[A B C D]
```

while other requests remain tracked by the scheduler.

The important point is that:

\[
\boxed{
\text{resident requests} \neq \text{forward batch size}
}
\]

## Batch Size Is a Compute Constraint

Batch size primarily controls how much work goes into one forward operation.

Increasing it can improve GPU utilization, but the forward operation also becomes more expensive.

Therefore:

\[
\text{batch size} \uparrow
\]

does not imply:

\[
\text{latency per token} \text{ stays constant}
\]

## Resident Requests Are a Memory Constraint

Each active request has KV state.

If request \(i\) owns \(B_i\) blocks:

\[
B_{used}
=
\sum_i B_i
\]

and:

\[
B_{free}
=
B_{total}-B_{used}
\]

A new request can only be admitted when enough memory capacity exists.

## Example

Suppose:

```text
512 total blocks
```

and current resident requests consume:

```text
A → 20
B → 30
C → 15
D → 25
```

Then:

\[
B_{used}=90
\]

and:

\[
B_{free}=512-90=422
\]

The system could keep additional requests resident even if the model forward batch is much smaller.

## The Better Mental Model

Think in two dimensions:

```text
            COMPUTE
              ↓
       batch size per forward
              ↑

MEMORY ← resident request set → MEMORY
              ↓
          KV blocks
```

A future resource-aware scheduler can consider both constraints at the same time.
