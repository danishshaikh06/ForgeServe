# KV Cache

## What Is KV Cache?

KV Cache stands for **Key-Value Cache**.

It is an optimization used during autoregressive text generation.

To understand KV Cache, we first need to understand what happens inside the Transformer's attention mechanism.

Attention uses three things:

```text
Query (Q)
Key   (K)
Value (V)
```

During generation, the model repeatedly processes tokens.

The Key and Value tensors belonging to tokens that have already been processed do not need to be calculated again.

Instead, they can be stored and reused.

This stored information is called the **KV Cache**.

---

# Why Is KV Cache Needed?

LLMs generate text one token at a time.

For example:

```text
Prompt
  ↓
"The sky is"
  ↓
"blue"
  ↓
"because"
  ↓
"light"
  ↓
...
```

Every new token depends on the tokens that came before it.

Without KV Cache, the model would repeatedly process the previous tokens.

For example:

```text
Step 1:

"The sky is"
       ↓
   Transformer
       ↓
    "blue"


Step 2:

"The sky is blue"
       ↓
   Transformer
       ↓
   "because"


Step 3:

"The sky is blue because"
       ↓
   Transformer
       ↓
    "light"
```

Notice that the model repeatedly processes:

```text
"The sky is"
```

and the tokens generated before the current token.

This creates redundant computation.

KV Cache reduces this redundancy.

---

# What Are Keys and Values?

In the attention mechanism, every token is transformed into:

```text
Query
Key
Value
```

A simplified attention operation is:

```text
Attention(Q, K, V)
```

The Query determines what the current token is looking for.

The Keys represent information that can be matched against the Query.

The Values contain the information that is actually retrieved.

A useful simplified way to think about them is:

```text
Query → "What information do I need?"

Key → "What information does this token represent?"

Value → "Here is the information associated with this token."
```

The exact mathematical behavior is more detailed, but this mental model is useful when learning inference systems.

---

# What Gets Cached?

The KV Cache stores:

```text
Keys
Values
```

It does **not** store Queries in the same way.

For every Transformer layer, the model maintains Key and Value tensors for the tokens that have already been processed.

Conceptually:

```text
Layer 1
  ├── Key Cache
  └── Value Cache

Layer 2
  ├── Key Cache
  └── Value Cache

Layer 3
  ├── Key Cache
  └── Value Cache

...

Layer N
  ├── Key Cache
  └── Value Cache
```

This means that a complete KV Cache can become quite large for long sequences.

---

# Prefill

The first stage of generation is called **prefill**.

Suppose the user sends:

```text
"Explain machine learning"
```

The model processes the complete prompt.

During this process, the model generates the initial Key and Value tensors.

These are stored in the KV Cache.

```text
Prompt
   │
   ▼
Transformer
   │
   ├──────────► K
   │
   └──────────► V
                 │
                 ▼
             KV Cache
```

After prefill, the model is ready to generate new tokens.

---

# Decode

After prefill, generation enters the **decode** stage.

The model generates one token at a time.

Suppose the model generates:

```text
"Machine"
```

The new token produces a new Key and Value.

These are added to the existing cache.

```text
Existing KV Cache
       +
New K/V
       ↓
Updated KV Cache
```

The process repeats:

```text
Prefill
   ↓
KV Cache
   ↓
Decode Token 1
   ↓
Update Cache
   ↓
Decode Token 2
   ↓
Update Cache
   ↓
Decode Token 3
   ↓
...
```

---

# Without KV Cache

Imagine that the prompt contains 100 tokens.

The model needs to generate 10 additional tokens.

Without KV Cache, the model repeatedly processes the growing sequence:

```text
Step 1 → 100 tokens
Step 2 → 101 tokens
Step 3 → 102 tokens
Step 4 → 103 tokens
...
Step 10 → 109 tokens
```

The previous tokens are repeatedly involved in the computation.

---

# With KV Cache

With KV Cache, the prompt is processed during prefill.

```text
Prefill

100 tokens
    ↓
KV Cache
```

Then decoding can reuse the cached Keys and Values:

```text
Decode 1 → New token + existing KV Cache
Decode 2 → New token + existing KV Cache
Decode 3 → New token + existing KV Cache
...
```

The model does not need to recompute the previous Keys and Values for every generation step.

---

# A Simple Analogy

Imagine you are reading a book and answering questions about it.

Without a cache, every time someone asks a question you reread the entire book.

```text
Question 1
   ↓
Read entire book

Question 2
   ↓
Read entire book

Question 3
   ↓
Read entire book
```

With a cache, you remember the information you already processed.

```text
Read book once
     ↓
Store useful information
     ↓
Question 1 → Use stored information
Question 2 → Use stored information
Question 3 → Use stored information
```

KV Cache works with a similar idea during autoregressive generation.

---

# KV Cache and Memory

KV Cache improves computation efficiency, but it requires GPU memory.

The longer the sequence becomes, the more Key and Value tensors need to be stored.

For a single request:

```text
Short sequence
     ↓
Small KV Cache

Long sequence
     ↓
Large KV Cache
```

Now imagine many users generating text simultaneously:

```text
Request A → KV Cache
Request B → KV Cache
Request C → KV Cache
Request D → KV Cache
...
```

The memory requirement can become very large.

This creates an important inference-system problem:

> How can we manage KV Cache memory efficiently?

This question leads to techniques such as **PagedAttention**.

---

# KV Cache Is Per Request

KV Cache contains information about a specific sequence.

For example:

```text
Request A
"Tell me about space"
       ↓
    Cache A
```

and:

```text
Request B
"Explain neural networks"
       ↓
    Cache B
```

These caches cannot be mixed.

Each request needs its own cache state.

This becomes especially important when ForgeServe starts supporting multiple concurrent requests.

---

# KV Cache and Batch Generation

Suppose we have three requests:

```text
Request A → 20 tokens
Request B → 50 tokens
Request C → 10 tokens
```

Each request has its own sequence and therefore its own KV state.

A production inference engine must keep track of:

```text
Request
   │
   └── KV Cache
```

As the number of requests increases, managing these caches becomes a systems problem involving:

* GPU memory
* Allocation
* Deallocation
* Fragmentation
* Scheduling
* Batching

These concerns will become important in later ForgeServe phases.

---

# KV Cache and FlashAttention

KV Cache and FlashAttention are related to inference performance, but they solve different problems.

### KV Cache

Reduces redundant computation between generation steps.

```text
Previous K/V
    ↓
Reuse
```

### FlashAttention

Improves the efficiency of the attention computation by reducing unnecessary memory movement and intermediate memory usage.

```text
Attention
    ↓
Efficient memory access
```

Therefore, they can be used together.

A simplified inference stack can look like:

```text
Generation
    │
    ▼
KV Cache
    │
    ▼
Attention
    │
    ▼
FlashAttention
    │
    ▼
GPU
```

---

# KV Cache in ForgeServe

In ForgeServe, KV Cache was introduced in Phase 2.

The generation process now follows:

```text
User Prompt
     │
     ▼
   Prefill
     │
     ▼
Create KV Cache
     │
     ▼
    Decode
     │
     ▼
Update KV Cache
     │
     ▼
    Decode
     │
     ▼
    ...
```

The underlying Transformer model is responsible for producing the Key and Value tensors.

ForgeServe manages the generation process and the cache state.

---

# Important Terms

### Prefill

Processing the initial prompt and creating the initial KV Cache.

### Decode

Generating new tokens using the existing KV Cache.

### Key

One of the tensors used by the attention mechanism to determine how tokens relate to the current Query.

### Value

The information retrieved by the attention mechanism after matching Queries with Keys.

### KV Cache

Stored Key and Value tensors from previously processed tokens.

### Context Length

The maximum number of tokens that can be processed as part of the model's context.

### Cache Growth

The KV Cache grows as new tokens are generated.

---

# The Main Idea

The most important idea to remember is:

> **KV Cache stores previously computed Keys and Values so they can be reused during autoregressive decoding.**

Without KV Cache:

```text
Recompute previous K/V
        ↓
Generate token
        ↓
Recompute previous K/V
        ↓
Generate token
```

With KV Cache:

```text
Compute K/V once
       ↓
Store
       ↓
Reuse
       ↓
Generate token
       ↓
Add new K/V
       ↓
Reuse
       ↓
Generate next token
```

This is one of the fundamental optimizations used by modern LLM inference systems.

However, the cache consumes GPU memory.

Therefore, once we move from a single request to many concurrent requests, **efficient KV Cache memory management becomes one of the most important problems in LLM serving**.
