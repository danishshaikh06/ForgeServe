# Block Allocation Before Batched Forward

## The Rule

Before a decode forward pass:

\[
\boxed{
\text{check}
\rightarrow
\text{allocate}
\rightarrow
\text{forward}
\rightarrow
\text{write KV}
}
\]

Why?

Because the model will produce new K/V values that need somewhere to live.

## Example

Suppose:

```text
block size = 16
```

Request A currently contains:

```text
32 tokens
```

Therefore:

\[
\left\lceil\frac{32}{16}\right\rceil=2
\]

blocks are needed.

Both blocks are full.

Now we are about to decode token 33.

The required blocks become:

\[
\left\lceil\frac{33}{16}\right\rceil=3
\]

So A needs one new block.

## Correct Order

Before the forward pass:

```text
A
 ↓
needs third block
 ↓
allocate block
 ↓
A now owns 3 blocks
```

Then:

```text
forward(A)
```

produces:

```text
K_new
V_new
```

and they can be written to the newly available storage.

## What Goes Wrong If We Allocate After?

Imagine:

```text
forward(A)
    ↓
K_new / V_new produced
    ↓
try to write
    ↓
no free block
```

Now the GPU has already done the work.

The new cache state cannot be stored safely.

The model output and runtime cache state are now inconsistent.

## Batched Version

Suppose:

```text
A → needs new block
B → does not
C → needs new block
D → does not
```

Before the single batched forward:

```text
A → allocate
B → nothing
C → allocate
D → nothing
```

Now every request has enough storage for the new KV.

Then:

```text
forward([A, B, C, D])
```

can execute safely.

## Mathematical Condition

For request \(i\), if the current sequence contains \(T_i\) tokens and one new token will be added:

\[
B_i^{required}
=
\left\lceil
\frac{T_i+1}{b}
\right\rceil
\]

where \(b\) is block size.

If:

\[
B_i^{required}>B_i^{current}
\]

then additional blocks must be allocated before the forward pass.

## Why This Is a Scheduling Concern

The scheduler cannot only think about:

```text
who should run?
```

It must also think about:

```text
does every selected request have enough KV storage?
```

So scheduling and memory management are connected.

## Mental Model

Think of the block as a reserved parking space.

The model is about to arrive with a new car.

You reserve the space first.

Then the car arrives.

Never do:

```text
car arrives
   ↓
look for parking afterward
```

The cache has the same requirement.
