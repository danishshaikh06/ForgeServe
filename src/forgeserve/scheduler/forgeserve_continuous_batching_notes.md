# ForgeServe — Continuous Batching and Batched Inference Study Notes

> A reusable reference for understanding continuous batching, scheduling, KV-cache memory, paged KV storage, and batched decode in ForgeServe.
>
> The goal of these notes is not to memorize an implementation. The goal is to understand **what the system is doing mathematically, what state must exist in memory, and why each software component exists**.

---

## Table of Contents

1. [Why We Need Continuous Batching](#1-why-we-need-continuous-batching)
2. [The Most Important Mental Model](#2-the-most-important-mental-model)
3. [Sequential Serving vs Continuous Batching](#3-sequential-serving-vs-continuous-batching)
4. [Static Batching vs Continuous Batching](#4-static-batching-vs-continuous-batching)
5. [What the Scheduler Actually Does](#5-what-the-scheduler-actually-does)
6. [Request State](#6-request-state)
7. [Request Lifecycle](#7-request-lifecycle)
8. [Global Scheduler State](#8-global-scheduler-state)
9. [The Running Batch](#9-the-running-batch)
10. [Batch Size](#10-batch-size)
11. [Why Batch Size Is Not Completely Arbitrary](#11-why-batch-size-is-not-completely-arbitrary)
12. [Compute Constraint vs Memory Constraint](#12-compute-constraint-vs-memory-constraint)
13. [KV Blocks and the Global Pool](#13-kv-blocks-and-the-global-pool)
14. [Block Allocation Mathematics](#14-block-allocation-mathematics)
15. [Dynamic KV Growth During Decode](#15-dynamic-kv-growth-during-decode)
16. [Admission Control](#16-admission-control)
17. [What Happens When There Is Not Enough Memory](#17-what-happens-when-there-is-not-enough-memory)
18. [Request Completion and Block Release](#18-request-completion-and-block-release)
19. [Worked Example: 256 Blocks](#19-worked-example-256-blocks)
20. [Scheduler Timeline](#20-scheduler-timeline)
21. [The Scheduler as a State Machine](#21-the-scheduler-as-a-state-machine)
22. [Mathematical Scheduler Loop](#22-mathematical-scheduler-loop)
23. [Scheduler Policy for ForgeServe](#23-scheduler-policy-for-forgeserve)
24. [What Continuous Batching Does Not Mean](#24-what-continuous-batching-does-not-mean)
25. [The Important GPU Mental Model](#25-the-important-gpu-mental-model)
26. [Why Batched Forward Can Be Faster](#26-why-batched-forward-can-be-faster)
27. [Weight Memory vs Repeated GPU Work](#27-weight-memory-vs-repeated-gpu-work)
28. [Question 1: Collecting the Current Tokens](#28-question-1-collecting-the-current-tokens)
29. [Question 2: Attention Masks](#29-question-2-attention-masks)
30. [Question 3: Batched KV Cache](#30-question-3-batched-kv-cache)
31. [Dense Batched KV Cache](#31-dense-batched-kv-cache)
32. [Paged KV Storage vs Dense Batched Cache](#32-paged-kv-storage-vs-dense-batched-cache)
33. [True Paged Attention](#33-true-paged-attention)
34. [Question 4: Distributing Output Back to Requests](#34-question-4-distributing-output-back-to-requests)
35. [Position Information and Cache Position](#35-position-information-and-cache-position)
36. [The Complete Batched Decode Step](#36-the-complete-batched-decode-step)
37. [Current ForgeServe Architecture](#37-current-forgeserve-architecture)
38. [Architecture We Are Building Toward](#38-architecture-we-are-building-toward)
39. [Stage 1: Dense Batched Cache](#39-stage-1-dense-batched-cache)
40. [Stage 2: True Paged Attention](#40-stage-2-true-paged-attention)
41. [Correctness Invariants](#41-correctness-invariants)
42. [Implementation Responsibilities](#42-implementation-responsibilities)
43. [Pseudocode](#43-pseudocode)
44. [Common Confusions](#44-common-confusions)
45. [What Our Benchmarks Should Measure](#45-what-our-benchmarks-should-measure)
46. [How to Interpret the Results](#46-how-to-interpret-the-results)
47. [Final Mental Model](#47-final-mental-model)
48. [Quick Revision Sheet](#48-quick-revision-sheet)

---

# 1. Why We Need Continuous Batching

An LLM inference server may receive many requests at overlapping times.

Suppose four users send requests:

```text
A → Explain KV cache
B → Explain attention
C → Explain batching
D → Explain paged memory
```

A sequential engine handles them like this:

```text
A A A A A A A A ... finish
B B B B B B B B ... finish
C C C C C C C C ... finish
D D D D D D D D ... finish
```

Only one request is actively being decoded at a time.

The GPU therefore receives many small, separate pieces of work.

Continuous batching changes the scheduling unit. Instead of thinking:

```text
run one request until it finishes
```

we think:

```text
at every decode step, choose the requests that should run now
```

The active set can change over time.

---

# 2. The Most Important Mental Model

The central idea is:

> **Continuous batching is a dynamically changing set of request states that share decode steps.**

At time `t`:

\[
\mathcal{R}_{running}(t)
\]

is the set of requests currently being decoded.

The decode batch is:

\[
\mathcal{B}_t = \mathcal{R}_{running}(t)
\]

for our simple scheduler.

At the next step:

\[
\mathcal{B}_{t+1}
\]

may be different.

Requests can enter and leave the batch without forcing all other requests to stop.

That is the defining idea behind continuous batching.

---

# 3. Sequential Serving vs Continuous Batching

## Sequential serving

```text
A → decode until finished
      ↓
   free A
      ↓
B → decode until finished
      ↓
   free B
      ↓
C → ...
```

At any point, approximately one request is resident.

## Continuous batching

```text
A ─┐
B ─┤
C ─┼→ decode together
D ─┘

A ─┐
B ─┤
C ─┼→ decode together
D ─┘

C finishes

A ─┐
B ─┤
D ─┼→ decode together
E ─┘
```

The batch membership changes while the system continues to produce tokens.

---

# 4. Static Batching vs Continuous Batching

## Static batching

A batch is created and remains fixed for some execution window.

For example:

```text
batch = [A, B, C, D]
```

If C finishes early, a static batch may have to tolerate the mismatch or padding until the batch boundary.

Conceptually:

\[
\mathcal{B}_t \approx \mathcal{B}_{t+1}
\]

for the lifetime of that static batch.

## Continuous batching

The batch is continuously rebuilt:

```text
step 1 → [A, B, C, D]
step 2 → [A, B, C, D]
step 3 → [A, B, D]
step 4 → [A, B, D, E]
```

Therefore:

\[
\boxed{\mathcal{B}_t \neq \mathcal{B}_{t+1}\ \text{in general}}
\]

---

# 5. What the Scheduler Actually Does

The scheduler does **not** generate language itself.

The model generates tokens.

The scheduler decides:

1. Which requests are waiting.
2. Which requests are currently running.
3. Which waiting requests can be admitted.
4. Which active requests belong to the next decode batch.
5. Which requests have finished.
6. When finished requests should release their KV blocks.

A useful summary is:

\[
\boxed{\text{Scheduler} = \text{Admission} + \text{Selection} + \text{Completion}}
\]

---

# 6. Request State

For each request \(i\), define:

\[
R_i = (P_i, G_i, T_i, B_i, S_i)
\]

where:

- \(P_i\) = prompt token count
- \(G_i\) = generated token count so far
- \(T_i = P_i + G_i\) = total sequence tokens represented by the request
- \(B_i\) = currently allocated KV blocks
- \(S_i\) = request status

Possible statuses:

```text
WAITING
RUNNING
FINISHED
CANCELLED
```

For example:

```text
Request A

P_A = 40
G_A = 17

T_A = 40 + 17 = 57
```

If the block size is 16 tokens:

\[
B_A = \left\lceil \frac{57}{16} \right\rceil = 4
\]

---

# 7. Request Lifecycle

A useful state machine is:

```text
                  submit
                    │
                    ▼
               ┌─────────┐
               │ WAITING │
               └────┬────┘
                    │
              enough blocks?
                yes │
                    ▼
               ┌─────────┐
               │ RUNNING │
               └────┬────┘
                    │
               decode token
                    │
              ┌─────┴─────┐
              │           │
          finished?     continue
              │           │
             yes          │
              ▼           │
        ┌──────────┐       │
        │ FINISHED │◄──────┘
        └────┬─────┘
             │
        release blocks
```

The important property is:

> **Blocks belong to the request while the request is running, and are released when the request finishes or is cancelled.**

---

# 8. Global Scheduler State

The complete scheduler can be thought of as:

\[
\mathcal{S}(t) = (Q_{waiting}, Q_{running}, B_{free}, B_{used})
\]

where:

- \(Q_{waiting}\) = waiting requests
- \(Q_{running}\) = active requests
- \(B_{free}\) = available blocks
- \(B_{used}\) = allocated blocks

For a fixed pool of \(N\) blocks:

\[
\boxed{B_{free} + B_{used} = N}
\]

This is one of the most important system invariants.

---

# 9. The Running Batch

At a decode step, each running request normally has exactly one next token to process.

For our simple scheduler:

\[
\mathcal{B}_t = Q_{running}
\]

If the running requests are:

```text
A
B
C
D
```

then:

\[
\mathcal{B}_t = \{A,B,C,D\}
\]

The batch size is:

\[
|\mathcal{B}_t| = 4
\]

If C finishes, the next step may be:

\[
\mathcal{B}_{t+1} = \{A,B,D\}
\]

---

# 10. Batch Size

Batch size means:

\[
\boxed{\text{number of requests processed in one decode step}}
\]

A configurable maximum can be written as:

\[
B_{max}
\]

For example:

```python
max_batch_size = 4
```

means:

\[
|\mathcal{B}_t| \leq 4
\]

It does **not** mean:

\[
|\mathcal{B}_t| = 4
\]

because continuous batching changes the active request set.

Example:

```text
step 1 → [A, B, C, D]  batch=4
step 2 → [A, B, C, D]  batch=4
step 3 → [A, B, D]     batch=3
step 4 → [A, B, D, E]  batch=4
```

---

# 11. Why Batch Size Is Not Completely Arbitrary

You can choose a maximum batch size as an engineering policy, but the actual batch is constrained by hardware and memory.

The scheduler is limited by at least two resources:

### Compute constraint

\[
|\mathcal{B}_t| \leq B_{max}
\]

### Memory constraint

The active requests need KV memory:

\[
\sum_{i \in \mathcal{R}_{active}} B_i \leq N
\]

So conceptually:

\[
\boxed{
\text{actual batch} \leq
\min(\text{configured batch limit},\text{memory capacity},\text{other GPU limits})
}
\]

---

# 12. Compute Constraint vs Memory Constraint

These are different problems.

## Compute side

The scheduler decides how many requests participate in a model forward:

```text
[A, B, C, D]
```

## Memory side

The block manager decides whether all their KV state can remain resident:

```text
A → 12 blocks
B → 18 blocks
C →  7 blocks
D → 15 blocks
```

Then:

\[
B_{used} = 12 + 18 + 7 + 15 = 52
\]

With 256 total blocks:

\[
B_{free}=256-52=204
\]

The scheduler therefore has to coordinate **compute scheduling** and **KV memory availability**.

---

# 13. KV Blocks and the Global Pool

ForgeServe uses a fixed pool of KV blocks.

In our Phase 4 setup:

- Total blocks = **256**
- Block size = **16 tokens**
- Total reserved KV memory = **48 MB**

The benchmark source explicitly describes the fixed pool, its 256 blocks, 16-token block size, and approximately 48 MB reserved KV memory. [See Phase 4 benchmark details.]

A request owns some subset of the pool.

For example:

```text
A → [B7, B3, B1]
B → [B5, B2]
```

The block IDs do not need to be contiguous in physical memory.

The request's logical KV sequence is represented by an ordered list of blocks.

---

# 14. Block Allocation Mathematics

Let the block size be:

\[
b = 16
\]

If a request currently has \(T_i\) total sequence tokens:

\[
\boxed{
B_i = \left\lceil \frac{T_i}{b} \right\rceil
}
\]

Example:

\[
T_i=40+128=168
\]

Then:

\[
B_i = \left\lceil \frac{168}{16} \right\rceil
= \lceil10.5\rceil
=11
\]

So the request needs 11 blocks.

The Phase 4 benchmark confirms the distinction between **blocks per request** and **total generated tokens across the benchmark**. [See the Phase 4 block-allocation explanation.]

---

# 15. Dynamic KV Growth During Decode

A request does not need a new block on every token.

Suppose:

\[
b=16
\]

and a request currently contains 31 tokens:

\[
B=\left\lceil\frac{31}{16}\right\rceil=2
\]

After one token:

\[
T=32
\Rightarrow B=2
\]

After one more token:

\[
T=33
\Rightarrow B=3
\]

So the block count changes only when a sequence crosses a block boundary.

Conceptually:

```text
31 tokens → 2 blocks
32 tokens → 2 blocks
33 tokens → 3 blocks
```

The runtime must therefore detect when:

\[
\left\lceil\frac{T_i}{b}\right\rceil
>
B_i^{current}
\]

and allocate another block.

---

# 16. Admission Control

Suppose a new request needs 12 blocks initially.

If:

\[
B_{free}=129
\]

then:

\[
129\geq12
\]

so the request can be admitted.

After admission:

\[
B_{free}=129-12=117
\]

The transition is:

```text
WAITING → RUNNING
```

The basic admission rule is:

\[
\boxed{
B_{free} \geq B_{required}
}
\]

---

# 17. What Happens When There Is Not Enough Memory

Suppose:

\[
B_{free}=7
\]

and a request needs 12 blocks.

Then:

\[
7 < 12
\]

The scheduler should **not** admit the request.

The request stays:

```text
WAITING
```

until capacity becomes available.

If another request finishes and releases 10 blocks:

\[
B_{free}=7+10=17
\]

Then the waiting request can be admitted:

\[
17\geq12
\]

This is memory-based admission control.

---

# 18. Request Completion and Block Release

Suppose request C owns four blocks:

\[
B_C=4
\]

and generates its final token.

Then:

\[
C: RUNNING \rightarrow FINISHED
\]

Its blocks are released:

\[
B_{free}\leftarrow B_{free}+4
\]

and:

\[
B_C\leftarrow0
\]

The released blocks can immediately be reused by a new request.

This is one of the key mechanisms that allows continuous batching to keep the GPU busy while request lifetimes differ.

---

# 19. Worked Example: 256 Blocks

Use:

\[
N=256
\]

blocks and:

\[
b=16
\]

tokens/block.

Requests:

| Request | Prompt Tokens | Max New Tokens |
|---|---:|---:|
| A | 31 | 6 |
| B | 30 | 10 |
| C | 15 | 4 |
| D | 47 | 8 |
| E | 20 | 6 |

At admission, only prompt tokens are resident.

### A

\[
\lceil31/16\rceil=2
\]

### B

\[
\lceil30/16\rceil=2
\]

### C

\[
\lceil15/16\rceil=1
\]

### D

\[
\lceil47/16\rceil=3
\]

Total:

\[
B_{used}=2+2+1+3=8
\]

Free:

\[
B_{free}=256-8=248
\]

---

# 20. Scheduler Timeline

A compact example:

| Time | Running Batch | Used Blocks | Free Blocks | Event |
|---|---|---:|---:|---|
| t=0 | A B C D | 8 | 248 | Initial prompt admission |
| t=1 | A B C D | 8 | 248 | Decode |
| t=2 | A B C D | 9 | 247 | A grows and needs a block |
| t=3 | A B C D E | 11 | 245 | E admitted |
| t=4 | A B C D E | 14 | 242 | B, C, D grow |
| t=5 | A B D E | 12 | 244 | C finishes and releases 2 blocks |
| t=6 | A B D E F | 15 | 241 | F admitted |

The important observation is that the batch and block usage change independently but are coordinated:

```text
Batch changes because requests enter/leave RUNNING.

Block usage changes because sequences grow/shrink their ownership.
```

---

# 21. The Scheduler as a State Machine

The scheduler can be summarized as:

```text
WAITING
   │
   │ enough capacity
   ▼
RUNNING
   │
   ├── decode one token
   │
   ├── maybe allocate another block
   │
   └── maybe finish
   │
   ▼
FINISHED
   │
   └── release all blocks
```

The same lifecycle may be extended later with:

```text
CANCELLED
ERROR
```

The key invariant is that every terminal path must release the request's KV blocks.

---

# 22. Mathematical Scheduler Loop

At time \(t\):

### 1. Admission

Choose waiting requests \(W_t\) such that their required capacity fits:

\[
\sum_{i\in W_t}B_i^{initial}\leq B_{free}
\]

Move them from WAITING to RUNNING.

### 2. Build the decode batch

\[
\mathcal{B}_t = Q_{running}
\]

for our simple policy.

### 3. Decode

For each request \(i\in\mathcal{B}_t\):

\[
x_{i,t+1}=DecodeStep(R_i)
\]

### 4. Update sequence length

\[
G_i\leftarrow G_i+1
\]

\[
T_i\leftarrow P_i+G_i
\]

### 5. Grow KV allocation if needed

\[
B_i^{required}
=
\left\lceil\frac{T_i}{b}\right\rceil
\]

If:

\[
B_i^{required}>B_i^{current}
\]

allocate another block.

### 6. Complete finished requests

For finished request \(i\):

\[
B_{free}\leftarrow B_{free}+B_i
\]

and remove the request from RUNNING.

### 7. Repeat

This can be written conceptually as:

\[
\boxed{
State_{t+1}
=
Complete(Grow(Decode(Admit(State_t))))
}
\]

---

# 23. Scheduler Policy for ForgeServe

Our first educational policy is deliberately simple:

- FIFO admission
- active requests participate in each decode step
- optional `max_batch_size`
- dynamic block allocation
- immediate block release on completion
- no preemption
- no priority scheduling

This is enough to understand continuous batching before adding more sophisticated policy.

---

# 24. What Continuous Batching Does Not Mean

Continuous batching does **not** mean:

- all requests have the same sequence length
- batch size stays constant
- requests must finish together
- one request owns the entire batch's KV cache
- the scheduler itself performs model inference
- the system automatically has a fused CUDA kernel

The key property is simply:

> **The active request set can change between decode steps.**

---

# 25. The Important GPU Mental Model

A very important correction:

> The model weights are not normally copied from CPU to VRAM on every forward pass.

Once the model is loaded onto the GPU, its weights remain resident in GPU memory.

The expensive repeated work is that every forward pass repeatedly **reads/uses those weights and launches the associated GPU computation**.

Separate requests therefore cause separate model executions:

```text
forward(A)
forward(B)
forward(C)
forward(D)
```

Batching changes this to:

```text
forward([A, B, C, D])
```

The same model weights participate in one larger set of matrix operations.

---

# 26. Why Batched Forward Can Be Faster

Consider a linear layer:

\[
Y=XW^T
\]

Sequential requests:

\[
Y_1=X_1W^T
\]

\[
Y_2=X_2W^T
\]

\[
\cdots
\]

\[
Y_N=X_NW^T
\]

Batched execution:

\[
X=
\begin{bmatrix}
X_1\\
X_2\\
\vdots\\
X_N
\end{bmatrix}
\]

and:

\[
\boxed{Y=XW^T}
\]

The GPU can execute a larger matrix operation instead of many separate small operations.

The benefit can come from better GPU utilization and reduced repeated execution overhead.

The exact speedup is workload-dependent.

---

# 27. Weight Memory vs Repeated GPU Work

It is useful to separate two concepts:

### Weights

The model weights occupy GPU memory and generally stay resident.

### Compute

Every forward pass still uses those weights.

So moving from:

```text
N separate forwards
```

to:

```text
1 batched forward
```

means we are not necessarily moving the weights N times between CPU and GPU. Instead, we are changing **how the GPU performs the repeated model computation**.

This distinction is important when reasoning about latency.

---

# 28. Question 1: Collecting the Current Tokens

At a decode step, each active request normally provides one current token.

For request \(i\):

\[
x_i \in \mathbb{N}
\]

For \(N\) requests:

\[
X_{tokens}
\in
\mathbb{N}^{N\times1}
\]

So:

\[
\boxed{batch\_tokens.shape=(N,1)}
\]

Example with four requests:

```text
A → 1532
B → 291
C → 8912
D → 42
```

becomes:

```python
batch_tokens = tensor([
    [1532],
    [291],
    [8912],
    [42],
])
```

Shape:

```text
(4, 1)
```

A conceptual construction is:

```python
batch_tokens = torch.cat(
    [request.last_token for request in active_requests],
    dim=0,
)
```

or, when the per-request token tensors have shape `(1,)`:

```python
batch_tokens = torch.stack(
    [request.last_token for request in active_requests],
    dim=0,
)
```

The critical requirement is consistent ordering:

\[
\boxed{batch[i]\leftrightarrow request_i}
\]

---

# 29. Question 2: Attention Masks

Requests can have different sequence lengths.

Example:

```text
A → 47 cached tokens
B → 23 cached tokens
C → 35 cached tokens
```

A normal dense tensor needs rectangular dimensions.

Define:

\[
L_{max}=\max(47,23,35)=47
\]

During a one-token decode step, the model conceptually has access to the past plus the new token.

A common dense representation therefore needs room for a shared maximum sequence dimension, with padding/masking for shorter sequences.

Conceptually:

```text
A: [real real real ... real real]
B: [real real real ... pad  real]
C: [real real real ... pad  real]
```

The attention mask tells the model which positions are meaningful.

A simplified rectangular representation is:

\[
M\in\{0,1\}^{N\times(L_{max}+1)}
\]

where:

\[
M_{i,j}=1
\]

means a real token position, and:

\[
M_{i,j}=0
\]

means padding/invalid position.

### Important caution

An attention mask by itself does **not** solve the full variable-length KV-cache problem.

The KV cache representation must also be compatible with a batched attention implementation.

---

# 30. Question 3: Batched KV Cache

This is the hardest part of turning the scheduler into true batched inference.

Suppose:

```text
Request A:
blocks = [B7, B3, B1]
seq_len = 47

Request B:
blocks = [B5, B2]
seq_len = 23
```

These are independent logical caches.

A typical dense batched cache is conceptually shaped like:

\[
K_l,V_l
\in
\mathbb{R}^{N\times H_{kv}\times L\times D}
\]

for each transformer layer \(l\).

where:

- \(N\) = batch size
- \(H_{kv}\) = number of KV heads
- \(L\) = shared dense sequence dimension
- \(D\) = head dimension

The model therefore expects **one batched cache representation**, not an arbitrary Python list of unrelated cache objects.

---

# 31. Dense Batched KV Cache

The easiest first implementation strategy is:

```text
Paged block pool
      ↓
Gather each request's logical KV sequence
      ↓
Build dense batched K/V tensors
      ↓
Single model.forward()
```

For example:

```text
A logical KV → K_A, V_A
B logical KV → K_B, V_B
C logical KV → K_C, V_C
```

then build:

\[
K_{batch}
\in
\mathbb{R}^{N\times H_{kv}\times L_{max}\times D}
\]

and:

\[
V_{batch}
\in
\mathbb{R}^{N\times H_{kv}\times L_{max}\times D}
\]

Shorter sequences use the chosen padding/masking representation.

### Why this is useful

It lets us obtain the important architectural improvement:

```text
many request-specific model forwards
        ↓
one batched model forward
```

without requiring us to write a custom attention kernel immediately.

### What this does not solve

Gathering can itself become a performance cost.

If we eventually do:

```text
A → gather
B → gather
C → gather
D → gather
        ↓
one forward
```

then the model forward is batched, but the cache preparation still has per-request overhead.

---

# 32. Paged KV Storage vs Dense Batched Cache

These are different concepts.

## Paged KV storage

Our current block manager stores KV data in fixed blocks:

```text
A → [B7, B3, B1]
B → [B5, B2]
```

The blocks may be physically non-contiguous.

This is a **memory-management abstraction**.

## Dense batched cache

The model-facing representation may instead be:

```text
K_batch: [batch, heads, max_seq_len, head_dim]
V_batch: [batch, heads, max_seq_len, head_dim]
```

This is a **model execution representation**.

Therefore:

> Paged storage does not automatically mean the model is executing paged attention.

---

# 33. True Paged Attention

A true paged-attention path would allow attention itself to read from the paged block pool directly.

Conceptually:

```text
Paged KV blocks
      +
block tables
      +
sequence lengths
      ↓
Paged-attention kernel
      ↓
attention output
```

Instead of:

```text
Paged KV blocks
      ↓
Gather into dense cache
      ↓
DynamicCache / dense attention
      ↓
attention output
```

For ForgeServe, the distinction is:

### Current educational implementation

\[
\boxed{
Paged\ KV\ storage
\rightarrow
Gather
\rightarrow
Dense\ cache
\rightarrow
Attention
}
\]

### Future kernel-level implementation

\[
\boxed{
Paged\ KV\ storage
\rightarrow
Paged\ attention\ kernel
}
\]

The second path is more advanced and is not something we should claim unless the kernel-level implementation actually exists.

---

# 34. Question 4: Distributing Output Back to Requests

Suppose the active batch contains:

```text
[A, B, C, D]
```

After one forward pass:

```text
logits.shape = (4, 1, vocab_size)
```

Take the final decode position:

```python
next_logits = outputs.logits[:, -1, :]
```

Now:

\[
next\_logits
\in
\mathbb{R}^{4\times V}
\]

where \(V\) is vocabulary size.

For greedy decoding:

\[
x_{i,next}=\arg\max_v next\_logits[i,v]
\]

This produces one token per request.

Example:

```text
row 0 → A next token
row 1 → B next token
row 2 → C next token
row 3 → D next token
```

The mapping is simple as long as the batch ordering is stable:

```python
for i, request in enumerate(active_requests):
    request.next_token = next_tokens[i]
```

The most important rule is:

\[
\boxed{batch[i] \leftrightarrow request_i}
\]

for the entire decode operation.

---

# 35. Position Information and Cache Position

Different requests are at different sequence positions.

Example:

```text
A → 47 tokens already present
B → 23 tokens already present
C → 35 tokens already present
```

The next token position for each request is therefore different.

Conceptually:

\[
P=
\begin{bmatrix}
47\\
23\\
35
\end{bmatrix}
\]

This matters for positional mechanisms such as RoPE.

A batched decode implementation therefore has to preserve per-request position information rather than pretending every request is at the same position.

Think of the batched input as:

\[
\boxed{(tokens, mask, KV, position)}
\]

not just:

```text
tokens
```

---

# 36. The Complete Batched Decode Step

Suppose the running requests are:

```text
A, B, C
```

with current sequence lengths:

```text
A → 47
B → 23
C → 35
```

The complete conceptual process is:

## Step 1 — Collect current tokens

\[
X_t=
\begin{bmatrix}
x_A\\
x_B\\
x_C
\end{bmatrix}
\]

Shape:

\[
(3,1)
\]

## Step 2 — Prepare attention representation

Create a dense model-facing representation with the appropriate sequence dimension and mask.

## Step 3 — Prepare batched KV

Gather the logical KV for each request into a shared model-facing representation.

## Step 4 — Prepare positions

\[
P_t=
\begin{bmatrix}
47\\
23\\
35
\end{bmatrix}
\]

## Step 5 — Single model forward

Conceptually:

```python
outputs = model(
    input_ids=batch_tokens,
    attention_mask=batch_attention_mask,
    past_key_values=batch_cache,
    position_ids=position_ids,
    ...,
)
```

## Step 6 — Extract next-token logits

```python
next_logits = outputs.logits[:, -1, :]
```

Shape:

```text
(batch_size, vocab_size)
```

## Step 7 — Sample one token per request

\[
x_{i,t+1}=Sampler(next\_logits_i)
\]

## Step 8 — Update each request

For each request:

- append the new token logically
- update generated token count
- write new K/V to its own cache
- allocate a new block if a boundary is crossed
- mark finished if EOS/max tokens is reached

## Step 9 — Rebuild the next batch

Finished requests leave.

Waiting requests may enter.

The next batch may therefore have a different membership.

---

# 37. Current ForgeServe Architecture

Before batch-aware forward execution, the logical flow was:

```text
Scheduler
   │
   ├── Request A → forward(A)
   ├── Request B → forward(B)
   ├── Request C → forward(C)
   └── Request D → forward(D)
```

This means that although the scheduler knows multiple requests are active, the model execution is still request-by-request.

The major improvement we are working toward is:

```text
Scheduler
     │
     ▼
Active requests [A, B, C, D]
     │
     ▼
Build one batched input
     │
     ▼
ONE model.forward()
     │
     ▼
[A logits, B logits, C logits, D logits]
```

This is the difference between:

> **continuous request scheduling**

and:

> **actual batched model execution**.

---

# 38. Architecture We Are Building Toward

The desired architecture is:

```text
                    Waiting Queue
                         │
                         ▼
                  ┌─────────────┐
                  │  Scheduler  │
                  │             │
                  │ admission   │
                  │ selection   │
                  │ completion  │
                  └──────┬──────┘
                         │
                         ▼
                Active Request Set
                  [A, B, C, D, E]
                         │
                         ▼
                Batch Preparation
              ┌──────────┼───────────┐
              │          │           │
           tokens       masks       KV state
              │          │           │
              └──────────┼───────────┘
                         ▼
                  ONE MODEL FORWARD
                         │
                         ▼
                  Batch of logits
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
              A          B           C ...
              │          │           │
              ▼          ▼           ▼
           update KV  update KV   update KV
                         │
                         ▼
                    next scheduler step
```

---

# 39. Stage 1: Dense Batched Cache

This is the practical first implementation for ForgeServe.

### Goal

Get all active requests through **one model forward pass**.

### Strategy

```text
per-request paged KV
        ↓
collect/gather
        ↓
dense batched KV
        ↓
model.forward()
```

### Advantages

- Easier to integrate with the existing model API.
- Lets us validate batched execution first.
- Keeps scheduler and block manager concepts understandable.
- Provides a clean baseline for measuring the benefit of batched forward execution.

### Disadvantage

KV gathering can become a new bottleneck.

This gives us a clean next experiment:

> Does the one-forward-pass improvement outweigh the cost of preparing the batched KV state?

---

# 40. Stage 2: True Paged Attention

Once Stage 1 works, a more advanced direction is:

```text
block pool
   +
block tables
   +
sequence lengths
   ↓
paged attention kernel
```

This avoids converting the paged cache into a dense representation before attention.

This is a significantly deeper GPU/kernel project.

It may involve:

- specialized attention kernels
- block tables
- variable-length sequence handling
- custom CUDA/Triton work
- careful memory access patterns
- kernel benchmarking

This is **future work**, not something required to understand continuous batching.

---

# 41. Correctness Invariants

These are the properties the implementation should continuously preserve.

## Invariant 1 — Pool conservation

\[
\boxed{B_{used}+B_{free}=N}
\]

For our current pool:

\[
B_{used}+B_{free}=256
\]

## Invariant 2 — Request ownership

For every active request:

\[
\boxed{B_i=|owned\_blocks_i|}
\]

## Invariant 3 — No double ownership

A single block must not belong to two independent requests.

For \(i\neq j\):

\[
owned(R_i)\cap owned(R_j)=\emptyset
\]

## Invariant 4 — Finished request owns no blocks

\[
R_i=FINISHED
\Rightarrow
B_i=0
\]

## Invariant 5 — Block count matches sequence length

\[
\boxed{
B_i=
\left\lceil\frac{T_i}{b}\right\rceil
}
\]

for exact dynamic allocation.

## Invariant 6 — Capacity is never exceeded

\[
\boxed{B_{used}\leq N}
\]

## Invariant 7 — Batch/request ordering

If request `A` occupies batch row 0, its logits must be interpreted as row 0.

\[
\boxed{batch[i]\leftrightarrow request_i}
\]

This must remain true from token collection all the way through output distribution.

---

# 42. Implementation Responsibilities

The architecture becomes much easier when responsibilities are separated.

## RequestState

Knows:

- request ID
- prompt
- token counts
- status
- logits
- paged KV state

## Scheduler

Knows:

- waiting queue
- running set
- admission
- request completion
- batch membership

## BlockManager

Knows:

- total blocks
- free blocks
- request ownership
- allocation
- release
- block reuse

## PagedRuntime

Knows:

- tokenization
- prefill
- decode mechanics
- KV extraction/update
- interaction with the model

## Batch Runtime / Batch Preparation

Knows:

- collecting tokens
- building the batch input
- building/collecting KV cache state
- building masks/positions
- one model forward
- separating outputs back to request states

This yields a useful separation:

\[
\boxed{Scheduler = when/who}
\]

\[
\boxed{Runtime = how}
\]

\[
\boxed{BlockManager = where memory lives}
\]

---

# 43. Pseudocode

## Simple continuous batching loop

```text
while server is running:

    admit waiting requests while capacity exists

    active = current running requests

    if active is empty:
        continue / wait for request

    build one decode batch

    for each active request:
        collect its current token

    prepare batched attention/cache representation

    run ONE model forward

    obtain one logits row per request

    sample one next token per request

    for every request:
        update generated token count
        update KV state
        allocate block if needed

        if EOS or max tokens:
            release blocks
            mark finished

    rebuild active set
```

## Batch tensor relationship

For batch size \(N\):

```text
batch_tokens        → (N, 1)
logits               → (N, 1, vocab_size)
next_logits          → (N, vocab_size)
next_tokens          → (N, 1)
```

The exact attention-mask/cache shapes depend on the chosen model/cache representation.

---

# 44. Common Confusions

## Confusion 1: "Continuous batching means all requests run truly simultaneously."

Not necessarily.

At the software level, the scheduler decides that multiple requests belong to the same decode batch.

The GPU executes the batched tensor operation as one model invocation.

The exact hardware execution is still controlled by CUDA and the kernels underneath.

---

## Confusion 2: "Batch size means total requests received by the server."

No.

Batch size is the number of requests processed in one decode batch.

A server may have:

```text
100 waiting requests
```

while:

```text
max_batch_size = 8
```

Only a subset may participate in the current batch.

---

## Confusion 3: "8 requests means 8 caches are always resident."

Not necessarily.

The Phase 4 benchmark had eight requests but processed them sequentially.

Therefore one request's blocks were freed before the next request began.

The benchmark generated:

\[
8\times128=1024
\]

total tokens, but those tokens were not simultaneously resident.

The Phase 4 benchmark explicitly distinguishes **total generated tokens** from **simultaneously resident tokens** and notes that the tested eight-request scenario was sequential rather than concurrent. [See Phase 4 benchmark explanation.]

---

## Confusion 4: "If the model is batched, KV caches become one logical request."

No.

The requests remain independent.

They are merely represented together in one model input/cache representation.

Each request still has its own:

- token count
- position
- KV state
- block ownership
- completion condition

---

## Confusion 5: "Attention mask solves variable-length KV caches."

Not by itself.

The model also needs a compatible cache representation and position information.

The hard part of batching is not only creating:

```text
attention_mask
```

but making:

```text
request-specific KV state
```

look like one valid batched cache representation.

---

## Confusion 6: "Paged KV storage is the same as PagedAttention."

No.

Paged KV storage is a memory-management technique.

True PagedAttention means the attention computation itself can directly consume paged KV blocks.

Our first batch-aware implementation is much easier if we gather paged KV into a dense batched representation first.

---

## Confusion 7: "Weights are transferred from CPU to GPU on every forward pass."

Normally no.

Once the model is loaded onto the GPU, the weights stay there.

Separate forwards repeatedly **use/read the same resident weights** and launch separate GPU computations.

Batching reduces the amount of separate model execution overhead and allows larger matrix operations.

---

# 45. What Our Benchmarks Should Measure

For continuous batching, useful measurements include:

## Throughput

\[
\text{tokens/sec}
=
\frac{\text{generated tokens}}{\text{total time}}
\]

## TTFT

Time to first token.

## TPOT

Time per output token during decode.

## Mean request latency

Average time from request start until completion.

## Batch size

Actual number of requests in each decode batch.

## Running requests

How many requests are resident at each step.

## Waiting requests

How many requests are queued but not yet admitted.

## Used blocks

\[
B_{used}
\]

## Free blocks

\[
B_{free}
\]

## Peak GPU memory

Measured process peak, not necessarily raw KV-cache size.

## Total generated tokens

Useful for throughput calculations, but must not be confused with simultaneously resident tokens.

---

# 46. How to Interpret the Results

The most important comparison is:

```text
Sequential per-request forward
vs
Batched one-forward-pass decode
```

For example, if four requests each need 80 decode steps:

### Sequential execution

Approximately:

\[
4\times80=320
\]

separate decode forward calls.

### Batched execution

Approximately:

\[
80
\]

batched forward calls, each processing four requests.

This is the conceptual source of the large reduction in model invocation count.

The exact speedup is hardware- and workload-dependent.

---

# 47. Final Mental Model

The whole system can be remembered as five layers.

## Layer 1 — Requests

```text
A, B, C, D, E
```

Each has its own state.

## Layer 2 — Scheduler

Decides:

```text
who waits?
who runs?
who finishes?
```

## Layer 3 — KV Memory

Block manager answers:

```text
where is each request's KV state?
how many blocks are free?
can another request be admitted?
```

## Layer 4 — Batch Construction

Transforms independent request states into:

```text
batch_tokens
attention representation
batched KV representation
positions
```

## Layer 5 — Model Forward

One batched model invocation returns one output row per request.

The whole loop is therefore:

\[
\boxed{
Requests
\rightarrow
Scheduler
\rightarrow
KV/Batch Preparation
\rightarrow
One Forward Pass
\rightarrow
Per-request Outputs
\rightarrow
Updated Requests
}
\]

---

# 48. Quick Revision Sheet

## Continuous batching

> A dynamically changing set of active requests sharing decode steps.

## Batch size

\[
|\mathcal{B}_t|
\]

Number of requests processed in one decode step.

## Maximum batch size

\[
|\mathcal{B}_t|\leq B_{max}
\]

A policy/configuration limit, not necessarily the actual batch size.

## Request total tokens

\[
T_i=P_i+G_i
\]

## Required blocks

\[
B_i=
\left\lceil\frac{T_i}{b}\right\rceil
\]

## Pool conservation

\[
B_{used}+B_{free}=N
\]

## Admission

\[
B_{free}\geq B_{required}
\]

## Running batch

For our simple scheduler:

\[
\mathcal{B}_t=Q_{running}
\]

## Batched token tensor

\[
\boxed{batch\_tokens.shape=(N,1)}
\]

## Batched logits

\[
logits.shape=(N,1,V)
\]

then:

\[
next\_logits.shape=(N,V)
\]

## Output mapping

\[
\boxed{batch[i]\leftrightarrow request_i}
\]

## Dense batched KV

Conceptually:

\[
K,V
\in
\mathbb{R}^{N\times H_{kv}\times L_{max}\times D}
\]

## Current Stage 1 architecture

\[
Paged\ KV
\rightarrow
Gather
\rightarrow
Dense\ batch\ cache
\rightarrow
One\ forward
\]

## Future Stage 2 architecture

\[
Paged\ KV
\rightarrow
Paged\ attention\ kernel
\]

## Three core responsibilities

\[
\boxed{Scheduler=when/who}
\]

\[
\boxed{Runtime=how}
\]

\[
\boxed{BlockManager=where}
\]

---

# Closing Perspective

The most useful way to think about continuous batching is not:

> "Put several prompts into a tensor."

It is:

> "Maintain many independent request states, keep their KV memory resident, dynamically decide which requests should receive the next decode step, transform those states into one valid batched model input, execute one forward pass, and then distribute the resulting token decisions back to the individual requests."

That mental model connects all of the concepts we have studied:

```text
Request lifecycle
      ↓
Scheduler
      ↓
Batch membership
      ↓
KV block ownership
      ↓
Variable sequence lengths
      ↓
Attention masks
      ↓
Batched KV representation
      ↓
Position information
      ↓
ONE model forward
      ↓
Per-request logits
      ↓
Sampling
      ↓
KV growth / completion
      ↓
Scheduler again
```

Once this loop is understood, the code becomes an implementation of the model rather than a collection of unrelated tensor operations.

---------------------------------------------------------------------------------------------------------------------------------------------
Continuous Batching — Batched Decode Questions

Setup: Two Requests Running Simultaneously

Request A: prompt was 4 tokens, has generated 2 tokens so far
           total sequence: [tok0, tok1, tok2, tok3, tok4, tok5]
           seq_len = 6

Request B: prompt was 8 tokens, has generated 5 tokens so far
           total sequence: [tok0, tok1, tok2, tok3, tok4, tok5, tok6, tok7, tok8, tok9, tok10, tok11, tok12]
           seq_len = 13

We want to decode the next token for BOTH in one forward pass.

Question 1 — What position is the new token?

For Request A, the new token is position 6 (after 6 existing tokens).

For Request B, the new token is position 13 (after 13 existing tokens).

When we pad the attention mask to L_max=13:

Request A mask (padded): [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
                          ← 7 zeros →  ← 6 ones for existing → + 1 for new
Request B mask (padded): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                          ← 13 ones for existing → + 1 for new

HuggingFace computes position_ids by counting non-zero entries in the mask up to each position. So:

Request A: model sees 7 zeros then 7 ones
           new token position_id = number of ones - 1 = 6  ✅ correct
           
Request B: model sees 13 ones
           new token position_id = 13  ✅ correct

The padding zeros tell the model "these positions do not exist." The model automatically figures out where the new token sits. You do not need to pass position_ids manually — the mask handles it.

Question 2 — Which position in the output KV contains the new token?

After the forward pass, HuggingFace returns updated KV of shape:

(N, num_heads, L_max + 1, head_dim)

The + 1 is the new token we just processed.

For Request A (seq_len=6, padded to 13, new token at position 6):

Output KV positions:
[pad, pad, pad, pad, pad, pad, pad, tok0, tok1, tok2, tok3, tok4, tok5, NEW]
  0    1    2    3    4    5    6    7     8     9     10    11    12    13

The new token is always at position -1 (the last position).

This is true for ALL requests regardless of their sequence length. The new token is always appended at the end. So we always extract [:, :, -1, :] — the last position in the sequence dimension.

This is exactly what _extract_last_token_kv already does in your PagedRuntime. The batched version does the same thing but for each request index i separately.

Question 3 — Why allocate blocks BEFORE the forward pass?

Think about what happens if you try to allocate AFTER:

Step 1: Run batched forward pass
        → success, logits computed

Step 2: Try to write new KV tokens to blocks
        Request A current block: 15/16 slots filled
        We write token → block is now FULL

Step 3: Try to write next token for Request A
        → KVCacheOutOfMemoryError: no free blocks!
        
But we already ran the forward pass.
The logits are computed.
We cannot "un-run" the GPU operation.
The token was sampled.
Now we cannot store its KV anywhere.
The state is corrupted.

If you allocate BEFORE:

Step 1: Check all requests
        Request A current block: 15/16 slots → will be full after this step
        Allocate new block for Request A now
        If allocation fails → skip this decode step, wait for blocks
        No GPU work wasted.

Step 2: Run batched forward pass
        All blocks guaranteed to have space.
        Write KV → success.

The rule is: never run GPU work you cannot store the result of.

Now Let Me Answer All Three Together Simply

Q1: The attention mask handles position ids automatically.
    Left-pad shorter sequences with zeros.
    Model counts the ones to find each token's position.
    You do not need to do anything extra.

Q2: The new token is always at position -1 in the output KV.
    For batch index i, extract: kv[i, :, -1, :]
    Write that into request i's current block.

Q3: Allocate blocks before the forward pass.
    Reason: if you cannot store the result, do not compute it.
    Check all requests → allocate where needed → then forward.
