# Phase 4 Concept: Continuous Batching & Iteration-Level Scheduling

## 1. Why We Need Continuous Batching

An LLM inference server receives requests at overlapping times.

Suppose four users send requests simultaneously:
```text
A → Explain KV cache
B → Explain attention
C → Explain batching
D → Explain paged memory
```

A **sequential engine** processes them one by one:
```text
A A A A A A A A ... finish
B B B B B B B B ... finish
C C C C C C C C ... finish
D D D D D D D D ... finish
```
Only one request is actively being decoded at a time, delivering small, inefficient pieces of work to the GPU.

**Continuous batching** changes the scheduling unit: instead of running one request until it finishes, at **every decode step** the engine chooses the dynamic set of active requests that should run now.

---

## 2. The Core Mental Model

> **Continuous batching is a dynamically changing set of request states that share decode steps.**

At iteration step $t$:
$$\mathcal{R}_{running}(t)$$
is the set of requests currently being decoded. The decode batch is $\mathcal{B}_t = \mathcal{R}_{running}(t)$.

At the next step $t+1$:
$$\mathcal{B}_{t+1}$$
may be completely different. Requests enter and leave the batch without forcing all other requests to pause or stop.

$$\boxed{\mathcal{B}_t \neq \mathcal{B}_{t+1}\ \text{in general}}$$

---

## 3. Serving Paradigm Comparisons

### Sequential Serving
```text
A → decode until finished -> free A
B → decode until finished -> free B
C → ...
```
At any point, only one request is resident on the device.

### Static Batching
A fixed batch is formed and executed as a rigid window:
```text
batch = [A, B, C, D]
```
If Request C finishes early, static batching wastes GPU cycles or pads execution until the entire batch boundary finishes.

### Continuous Batching
The batch is continuously rebuilt step by step:
```text
step 1 → [A, B, C, D]
step 2 → [A, B, C, D]
step 3 → [A, B, D]     (C finished and leaves instantly!)
step 4 → [A, B, D, E]  (E admitted instantly!)
```

---

## 4. What the Scheduler Actually Does

The scheduler does **not** generate language tokens itself (the PyTorch model does). The scheduler decides:
1. Which requests are waiting.
2. Which requests are currently running.
3. Which waiting requests can be admitted.
4. Which active requests belong to the next decode batch.
5. Which requests have finished.
6. When finished requests should release their physical KV blocks.

$$\boxed{\text{Scheduler} = \text{Admission} + \text{Selection} + \text{Completion}}$$

---

## 5. Request State & Lifecycle

For each request $i$:
$$R_i = (P_i, G_i, T_i, B_i, S_i)$$

Where:
* $P_i$ = prompt token count
* $G_i$ = generated token count so far
* $T_i = P_i + G_i$ = total sequence tokens
* $B_i = \lceil T_i / b \rceil$ = allocated physical KV blocks ($b = \text{block\_size}$)
* $S_i$ = request status (`WAITING`, `RUNNING`, `FINISHED`, `CANCELLED`)

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

---

## 6. Global Scheduler State & Invariants

The global scheduler state is represented as:
$$\mathcal{S}(t) = (Q_{waiting}, Q_{running}, B_{free}, B_{used})$$

For a fixed pool of $N$ blocks:
$$\boxed{B_{free} + B_{used} = N}$$

### Compute vs. Memory Constraints

The scheduler capacity is governed by two boundaries:
1. **Compute Constraint**: $|\mathcal{B}_t| \le B_{max}$
2. **Memory Constraint**: $\sum_{i \in \mathcal{R}_{active}} B_i \le N$

$$\boxed{\text{actual batch} \le \min(\text{configured batch limit}, \text{memory capacity}, \text{GPU limits})}$$

---

## 7. Worked Example: 256 Physical Blocks

Setup: $N = 256$ blocks, $b = 16$ tokens/block (48 MB total KV memory pool).

| Request | Prompt Tokens ($P_i$) | Initial Blocks Needed ($\lceil P_i / 16 \rceil$) |
| :--- | :--- | :--- |
| **A** | 31 | 2 |
| **B** | 30 | 2 |
| **C** | 15 | 1 |
| **D** | 47 | 3 |
| **Total** | | **8 Blocks Used** ($B_{free} = 248$) |

### Timeline Progression:
| Time Step | Running Batch | Used Blocks ($B_{used}$) | Free Blocks ($B_{free}$) | Event |
| :--- | :--- | :--- | :--- | :--- |
| **t=0** | A, B, C, D | 8 | 248 | Initial prompt prefill & admission |
| **t=1** | A, B, C, D | 8 | 248 | Decode step 1 |
| **t=2** | A, B, C, D | 9 | 247 | Request A crosses token 32 boundary $\rightarrow$ allocates Block 3 |
| **t=3** | A, B, C, D, E | 11 | 245 | Request E admitted |
| **t=5** | A, B, D, E | 12 | 244 | Request C finishes $\rightarrow$ releases 2 physical blocks |
| **t=6** | A, B, D, E, F | 15 | 241 | Request F admitted |

---

## 8. Hardware & GPU Execution Mechanics

### Weight Memory vs Repeated GPU Work
Model weights occupy VRAM and remain resident in GPU memory once loaded.

Moving from sequential execution to continuous batching changes:
$$\text{N separate forward calls } forward(A), forward(B) \longrightarrow \text{1 batched call } forward([A, B, C, D])$$

For a linear layer $Y = X W^T$, batching stacks input vectors:
$$X = \begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_N \end{bmatrix} \implies \boxed{Y = X W^T}$$

This increases matrix multiplication dimensions (GEMM utilization), reducing launch overhead and saturating CUDA cores efficiently.

---

## 9. Benchmark Interpretation Metrics

To evaluate continuous batching accurately:
* **Throughput (tokens/sec)**: $\frac{\text{Total Generated Tokens}}{\text{Total Time}}$
* **TTFT (Time To First Token)**: Latency from submission to prefill completion.
* **TPOT (Time Per Output Token)**: Average decode iteration step latency.
* **Block Conservation**: Confirming $B_{used} + B_{free} = N$ dynamically.
