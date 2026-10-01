# Phase 4 Concept: Continuous Batching & Iteration-Level Scheduling

## Overview

Traditional batching in deep learning (Static Batching) processes requests in **fixed batches**. In LLM serving, static batching leads to huge inefficiencies because sequences have **variable output lengths**.

In Static Batching:
* Short requests finish early and sit idle while long requests complete.
* New incoming requests must wait until the **entire batch** finishes before execution starts.

**Continuous Batching** (also known as Iteration-level Scheduling or In-flight Batching) operates at the level of individual engine iterations (tokens) rather than entire sequence generations.

---

## Static Batching vs. Continuous Batching

### Static Batching (Inefficient)

```text
Time --->
Req 1: [ Prefill ] [ Decode ] [ Decode ] [ Finished ] ------------- (GPU Idle) ----|
Req 2: [ Prefill ] [ Decode ] [ Decode ] [ Decode ] [ Decode ] [ Finished ] ---------| Batch Ends
Req 3: [ Prefill ] [ Decode ] [ Finished ] ------------------------- (GPU Idle) ----|

Incoming Req 4 has to WAIT until ALL 3 finish!
```

### Continuous Batching (High Throughput)

```text
Time --->
Step 1: Req 1 (Decode) | Req 2 (Decode) | Req 3 (Decode)
Step 2: Req 1 (Decode) | Req 2 (Decode) | Req 3 (Finished!) -> INSERT Req 4 (Prefill)!
Step 3: Req 1 (Decode) | Req 2 (Decode) | Req 4 (Decode)
```

At **every single iteration step**, finished sequences leave the batch immediately, and waiting incoming sequences are inserted into the running batch seamlessly!

---

## Prefill vs. Decode Phase Multiplexing

Inference consists of two distinct compute profiles:

1. **Prefill Phase (Compute-Bound)**:
   * Processes the initial input prompt tokens in parallel.
   * High matrix multiplication utilization (GEMM).
2. **Decode Phase (Memory-Bound)**:
   * Generates tokens one by one autoregressively.
   * Bound by memory bandwidth (GEMV / KV Cache transfers).

Continuous Batching allows the engine to schedule a mix of **Prefill steps** and **Decode steps** in the same iteration to maximize both compute utilization and memory bandwidth throughput.

---

## Request Preemption & Swapping

When physical GPU memory blocks are exhausted:
* The scheduler can **preempt** running low-priority sequences.
* Preempted sequence KV blocks can be **swapped to CPU RAM** or **recomputed** later.
* Once GPU memory frees up, the preempted sequence is resumed.

---

## Key Benefits

* **Minimal Latency**: New requests start processing immediately without waiting for existing long generations to finish.
* **Maximized GPU Utilization**: No wasted GPU cycles waiting for idle sequences in a static batch.
* **High Requests Per Second (RPS)**: Up to 5x-10x throughput improvement over static batching.
