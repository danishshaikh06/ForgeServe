# KV Cache

## Introduction

KV Cache is an optimization used during autoregressive text generation.

It prevents the model from repeatedly computing the same attention information for tokens that have already been processed.

In Phase 1, ForgeServe processed the complete sequence during every generation step.

In Phase 2, ForgeServe stores previously computed Key and Value tensors and reuses them during decoding.

This significantly reduces unnecessary computation.

---

# Why Do We Need KV Cache?

Remember how autoregressive generation works.

Suppose the prompt contains three tokens:

```text
Token 1 → Token 2 → Token 3
```

To generate the next token, the model processes these tokens.

After generating Token 4, the sequence becomes:

```text
Token 1 → Token 2 → Token 3 → Token 4
```

In the Phase 1 implementation, the model processes the entire sequence again.

```text
Step 1

Forward(Token 1, Token 2, Token 3)

↓

Token 4


Step 2

Forward(Token 1, Token 2, Token 3, Token 4)

↓

Token 5


Step 3

Forward(Token 1, Token 2, Token 3, Token 4, Token 5)

↓

Token 6
```

The model repeatedly processes tokens it has already seen.

This creates unnecessary computation.

---

# What Does KV Mean?

KV stands for:

* **K — Key**
* **V — Value**

These are two of the three components used by the attention mechanism:

```text
Query
Key
Value
```

Attention uses these tensors to determine which previous tokens are important when processing the current token.

During autoregressive generation, the Keys and Values for previous tokens do not need to be recomputed.

They can be stored and reused.

This stored information is called the **KV Cache**.

---

# KV Cache Without Going Too Deep

Consider a sequence:

```text
Token 1
Token 2
Token 3
```

During the first forward pass, the model calculates the Keys and Values for these tokens.

Instead of throwing them away, we store them:

```text
KV Cache

Token 1 → K, V
Token 2 → K, V
Token 3 → K, V
```

When Token 4 needs to be generated, the model only needs to calculate the new Key and Value for Token 4.

The previous values are reused.

```text
Existing Cache

Token 1 → K, V
Token 2 → K, V
Token 3 → K, V

        +

New Token

Token 4 → New K, V
```

The cache is then updated.

```text
KV Cache

Token 1 → K, V
Token 2 → K, V
Token 3 → K, V
Token 4 → K, V
```

The process continues for every generated token.

---

# Prefill and Decode

KV Cache introduces an important distinction between two stages of inference.

## Prefill

The first stage processes the complete input prompt.

For example:

```text
"What is machine learning?"
```

The entire prompt is processed and the model creates the initial KV Cache.

```text
Prompt
  │
  ▼
Transformer
  │
  ▼
Initial KV Cache
```

This stage is called **prefill**.

---

## Decode

After prefill, the model generates one token at a time.

For each new token:

```text
New Token
   │
   ▼
Transformer
   │
   +
Existing KV Cache
   │
   ▼
Updated KV Cache
   │
   ▼
Next Token
```

Only the new token needs to be processed while previously calculated Keys and Values are reused.

---

# Phase 1 vs Phase 2

## Phase 1

The sequence grows after every generated token.

```text
Forward(100 tokens)

Forward(101 tokens)

Forward(102 tokens)

Forward(103 tokens)
```

The amount of input processed repeatedly increases.

---

## Phase 2

The initial prompt is processed once.

```text
Prefill(100 tokens)

        ↓

KV Cache

        ↓

Decode(1 token)

        ↓

Update Cache

        ↓

Decode(1 token)

        ↓

Update Cache
```

This avoids repeatedly processing the entire sequence.

---

# KV Cache in ForgeServe

ForgeServe introduces a dedicated KV Cache abstraction.

The main components are:

```text
Generation Engine
       │
       ▼
Runtime
       │
       ▼
KV Cache
       │
       ▼
Transformer Model
```

The Generation Engine controls the generation process.

The Runtime communicates with the transformer.

The KV Cache stores the previously computed Key and Value tensors.

---

# KV Cache Lifecycle

The cache follows a simple lifecycle.

```text
Create
  │
  ▼
Prefill
  │
  ▼
Store K/V
  │
  ▼
Decode
  │
  ▼
Update K/V
  │
  ▼
Decode
  │
  ▼
...
  │
  ▼
Generation Complete
  │
  ▼
Clear / Release
```

A cache belongs to a generation request.

Once generation finishes, the cache should no longer be used for that request.

---

# Why Cache Ownership Matters

KV Cache is not global state.

Different generation requests have different sequences.

For example:

```text
Request A

"Hello"

↓

Cache A
```

and:

```text
Request B

"What is AI?"

↓

Cache B
```

These caches must remain separate.

Mixing the caches would cause the model to use information from one request while generating another request.

This is why cache ownership becomes an important systems problem as ForgeServe evolves.

---

# Current Implementation

ForgeServe Phase 2 uses the model's existing support for KV caching.

The underlying transformer calculates and returns the Key and Value tensors.

ForgeServe is responsible for managing the cache during generation and passing the cached values back into subsequent decode steps.

This distinction is important.

ForgeServe is an **inference engine**, not a reimplementation of the transformer architecture.

---

# Why Didn't We Implement the Attention Kernel?

The goal of ForgeServe is to understand and build an inference system around existing transformer models.

Reimplementing the entire transformer would turn the project into a different problem.

Instead, ForgeServe focuses on the systems layer:

```text
Model
  │
  ▼
Runtime
  │
  ▼
Cache Management
  │
  ▼
Generation
  │
  ▼
Scheduling
  │
  ▼
Serving
```

This allows us to study how modern inference systems optimize and serve existing models.

---

# Benefits of KV Cache

KV Cache provides several important benefits:

* Avoids redundant computation.
* Makes autoregressive decoding more efficient.
* Reduces the amount of computation required for each new token.
* Provides the foundation for more advanced inference optimizations.

However, KV Cache also introduces a new problem.

The cache consumes GPU memory.

As the sequence length and number of concurrent requests increase, KV Cache can become one of the largest consumers of GPU memory.

This memory-management problem becomes extremely important in later phases.

---

# The Next Problem

KV Cache solves one problem:

> Avoid recomputing Keys and Values.

But it introduces another:

> How do we efficiently manage a large amount of KV Cache memory?

Suppose many users are generating text simultaneously.

```text
Request A → KV Cache
Request B → KV Cache
Request C → KV Cache
Request D → KV Cache
...
```

Each request requires memory.

Managing this memory efficiently becomes increasingly difficult.

This leads to more advanced techniques such as **PagedAttention** and better memory allocation strategies.

---

# Relationship With FlashAttention

KV Cache and FlashAttention solve different problems.

### KV Cache

Focuses on avoiding redundant computation across autoregressive decoding steps.

```text
Previous tokens
      ↓
Reuse K/V
```

### FlashAttention

Focuses on making the attention operation itself more memory-efficient by reducing unnecessary memory movement and intermediate memory usage.

```text
Attention computation
      ↓
Efficient memory access
```

They can therefore be used together.

ForgeServe will explore FlashAttention in a future phase.

---

# Summary

KV Cache stores the Key and Value tensors produced for previously processed tokens.

During autoregressive generation, these cached values can be reused instead of being recomputed.

Phase 2 of ForgeServe introduces KV Cache management into the inference pipeline.

The generation process now consists of two main stages:

```text
Prefill
   ↓
KV Cache
   ↓
Decode
   ↓
Update Cache
   ↓
Decode
   ↓
...
```

KV Cache is one of the fundamental optimizations used by modern LLM inference systems.

It improves decoding efficiency, but it also introduces a new systems challenge: efficiently managing GPU memory.

That challenge will become increasingly important as ForgeServe moves toward PagedAttention and concurrent request serving.
