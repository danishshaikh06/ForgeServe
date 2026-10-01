# Phase 5 Concept: Async Serving Engine & Streaming Architecture

## Overview

Phase 5 brings all component modules (Model Execution, PagedAttention Memory Management, Continuous Scheduler, and Sampler) into a non-blocking, multi-threaded **Async Engine**.

This architecture powers high-concurrency production deployments by providing:
* Async/Await request submission.
* Real-time stream generation (Server-Sent Events / WebSockets).
* Graceful client cancellation & timeout handling.
* OpenAI API compatibility layer (`/v1/completions`, `/v1/chat/completions`).

---

## Architectural Workflow

```text
Client Request
      |
      v
FastAPI / Async HTTP Server
      |
      v
AsyncLLMEngine.add_request()  ---> Pushes Request to Input Queue
      |
+-------------------------------------------------------------+
|                 Background Engine Event Loop                |
|                                                             |
|  1. Step Scheduler (Continuous Batching)                    |
|  2. Execute Model Iteration (PagedAttention Forward Pass)   |
|  3. Sample Tokens                                           |
|  4. Stream New Tokens to Request Output Streams             |
+-------------------------------------------------------------+
      |
      v
Async Streams (AsyncGenerator yields tokens back to client)
```

---

## Key Async Server Features

### 1. Asynchronous Queue & Background Loop
The engine runs a background event loop task (`_step_loop`). The HTTP server threads hand off incoming requests via asynchronous queues, preventing web threads from blocking on model forward passes.

### 2. Stream Generation
Token responses are streamed back to the client token-by-token using Python `async generator` patterns, enabling low **Time-To-First-Token (TTFT)**.

### 3. Early Request Abort
If a client disconnects mid-generation, the server immediately notifies the engine to abort the request sequence ID. The `Scheduler` and `BlockAllocator` instantly free the physical GPU memory blocks occupied by the aborted sequence.
