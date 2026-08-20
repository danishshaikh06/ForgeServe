**One physical `KVBlock` contains:**

```text
Every layer
   ↓
Each layer's K cache
   ↓
Each layer's V cache
   ↓
For a fixed number of tokens = block_size
```

For example, if:

```text
num_layers = 4
num_heads  = 8
block_size = 16
head_dim   = 64
```

then **one block** contains:

```text
Block 7
│
├── Layer 0
│    ├── K → 16 tokens
│    └── V → 16 tokens
│
├── Layer 1
│    ├── K → 16 tokens
│    └── V → 16 tokens
│
├── Layer 2
│    ├── K → 16 tokens
│    └── V → 16 tokens
│
└── Layer 3
     ├── K → 16 tokens
     └── V → 16 tokens
```

The tensor shape is:

```text
K = (4, 8, 16, 64)
V = (4, 8, 16, 64)
      │  │  │  │
      │  │  │  └── head dimension
      │  │  └───── tokens in this block
      │  └──────── KV heads
      └─────────── layers
```

So the key sentence to remember is:

> **A block represents a chunk of tokens, and for that chunk it stores K/V data for every transformer layer.**

For example, if `block_size = 16`:

```text
Block 0 → tokens 0–15, for ALL layers
Block 1 → tokens 16–31, for ALL layers
Block 2 → tokens 32–47, for ALL layers
```

# Paged KV Cache — Complete Notes

## 1. Big picture

The main problem we're solving is:

> **How do we store and reuse the KV cache efficiently during LLM inference?**

There are three important pieces:

```text
KVBlock
   ↓
actual KV-cache storage

BlockManager
   ↓
manages/allocates the KV blocks

PagedKVCache
   ↓
manages the blocks for one request and
writes/gathers KV data
```

A simple way to remember:

```text
KVBlock      → "What is the storage?"
BlockManager → "Who gets the storage?"
PagedKVCache → "How do I use the storage for this request?"
```

---

# 2. What is the KV cache?

During transformer attention, every layer produces:

```text
Q = Query
K = Key
V = Value
```

For autoregressive generation, we don't want to recompute the K and V for all previous tokens every time we generate a new token.

So we **cache K and V** from previous tokens.

For example:

```text
Input:

T0 T1 T2 T3
```

After processing them, we have:

```text
Layer 0 → K/V for T0 T1 T2 T3
Layer 1 → K/V for T0 T1 T2 T3
Layer 2 → K/V for T0 T1 T2 T3
...
```

These K/V values are stored in the **KV cache**.

When a new token arrives:

```text
T4
```

the model can reuse:

```text
K/V for T0 T1 T2 T3
```

instead of recomputing them.

---

# 3. Why do we store KV for every layer?

This is very important.

You might think:

> "The last layer has the most information, so why not just store the last layer's KV?"

Because **every attention layer needs its own historical K/V**.

Suppose we have 3 layers:

```text
Layer 0 → K0,V0
Layer 1 → K1,V1
Layer 2 → K2,V2
```

When a new token comes:

```text
T4
 ↓
Layer 0
```

Layer 0 needs its previous:

```text
K0,V0
```

Then:

```text
Layer 1
```

needs:

```text
K1,V1
```

And:

```text
Layer 2
```

needs:

```text
K2,V2
```

So:

```text
Layer 0 → needs Layer 0's KV history
Layer 1 → needs Layer 1's KV history
Layer 2 → needs Layer 2's KV history
```

The last layer's K/V cannot replace the earlier layers' K/V.

### Remember:

> **Each layer has its own attention operation and therefore needs its own K/V cache.**

---

# 4. What is a KVBlock?

A `KVBlock` represents **one physical piece of KV-cache memory**.

For example:

```python
block_id: int
block_size: int
num_layers: int
num_heads: int
head_dim: int
device: str
num_filled: int = 0
```

It also contains:

```python
k_cache
v_cache
```

These are the actual GPU tensors.

---

# 5. Shape of one KVBlock

The code has:

```python
shape = (
    self.num_layers,
    self.num_heads,
    self.block_size,
    self.head_dim
)
```

So:

```text
K shape =
(num_layers, num_heads, block_size, head_dim)

V shape =
(num_layers, num_heads, block_size, head_dim)
```

For example:

```text
num_layers = 4
num_heads = 8
block_size = 16
head_dim = 64
```

One block contains:

```text
K = (4, 8, 16, 64)
V = (4, 8, 16, 64)
```

---

# 6. Does every layer have a separate block?

**No.**

This was an important point to clarify.

A single physical block contains KV data for **every layer**.

Think:

```text
Block 5
│
├── Layer 0
│    ├── K
│    └── V
│
├── Layer 1
│    ├── K
│    └── V
│
├── Layer 2
│    ├── K
│    └── V
│
└── Layer 3
     ├── K
     └── V
```

So:

> **One block represents a chunk of tokens, and for that chunk it stores K/V for every layer.**

For `block_size = 4`:

```text
Block 0 → T0 T1 T2 T3 for ALL layers
Block 1 → T4 T5 T6 T7 for ALL layers
Block 2 → T8 T9 ... for ALL layers
```

---

# 7. What does `block_size` mean?

`block_size` means:

> **How many token positions can one physical block hold.**

For example:

```text
block_size = 4
```

means:

```text
Block 0
┌────┬────┬────┬────┐
│ T0 │ T1 │ T2 │ T3 │
└────┴────┴────┴────┘
```

The same block also contains Layer 0, Layer 1, etc. for those four token positions.

---

# 8. What is `num_filled`?

Each block has:

```python
num_filled
```

It tells us:

> **How many token positions in this particular block are currently being used.**

Example:

```text
block_size = 4
```

Initially:

```text
Block 3:

[   ][   ][   ][   ]

num_filled = 0
```

After T4:

```text
[T4][   ][   ][   ]

num_filled = 1
```

After T5:

```text
[T4][T5][   ][   ]

num_filled = 2
```

After four tokens:

```text
[T4][T5][T6][T7]

num_filled = 4
```

The block is now full.

---

# 9. Why do we pre-create blocks?

This is an important question.

The blocks are created **before a request's prefill**.

When `BlockManager` is initialized, it can pre-allocate all the KV memory.

For example:

```text
num_blocks = 5
```

The manager creates:

```text
Block 0
Block 1
Block 2
Block 3
Block 4
```

Initially:

```text
Block 0 → FREE
Block 1 → FREE
Block 2 → FREE
Block 3 → FREE
Block 4 → FREE
```

The actual K/V tensors have already been allocated on the GPU.

So:

> **Creating a block = allocating its KV memory.**

---

# 10. What is `_pool`?

The `_pool` keeps references to **all the physical KVBlock objects**.

You can think of:

```python
self._pool
```

as:

```text
_pool
│
├── Block 0
├── Block 1
├── Block 2
├── Block 3
└── Block 4
```

> **`_pool` keeps track of all the KV blocks that have been created.**

---

# 11. What is `_free_stack`?

The manager also needs to know:

> Which blocks are currently available?

That's what:

```python
self._free_stack
```

is used for.

Initially:

```text
_free_stack = [0, 1, 2, 3, 4]
```

This means all five blocks are available.

When Block 4 is allocated:

```text
_free_stack = [0, 1, 2, 3]
```

When another block is allocated:

```text
_free_stack = [0, 1, 2]
```

The `pop()` operation is O(1), so getting a free block is fast.

---

# 12. What is `_owned`?

`_owned` tracks which blocks belong to which request.

Example:

```python
_owned = {
    "request_A": [4, 3],
    "request_B": [1],
}
```

Meaning:

```text
Request A → Block 4, Block 3
Request B → Block 1
```

So:

```text
_pool
→ What blocks exist?

_free_stack
→ Which blocks are free?

_owned
→ Which request owns which blocks?
```

---

# 13. BlockManager's job

The `BlockManager` is responsible for:

* creating/pre-allocating blocks
* tracking all blocks
* tracking free blocks
* allocating blocks to requests
* tracking ownership
* freeing blocks when requests finish

It is essentially the **memory manager for KV blocks**.

---

# 14. `allocate()`

Method:

```python
def allocate(
    self,
    request_id: str,
    num_blocks: int = 1
)
```

means:

> "Give this request `num_blocks` physical KV blocks."

For example:

```python
allocate("request_A", 2)
```

Suppose:

```text
_free_stack = [0, 1, 2, 3, 4]
```

The method does:

```python
block_id = self._free_stack.pop()
```

Maybe it gets:

```text
Block 4
```

Then:

```text
Block 3
```

So:

```text
Request A → Block 4, Block 3
```

---

# 15. Why does `allocate()` call `block.reset()`?

When a block is reused, it may contain old data from another request.

So:

```python
block.reset()
```

resets its bookkeeping/state before giving it to the new request.

Conceptually:

```text
Old Request B
    ↓
Block 4

Request B finishes
    ↓
Block 4 becomes free
    ↓
Request A gets Block 4
    ↓
reset()
    ↓
Block 4 ready for Request A
```

---

# 16. Understanding `_owned[request_id].extend(...)`

You had:

```python
self._owned[request_id].extend(
    b.block_id for b in allocated
)
```

Suppose:

```python
allocated = [Block 4, Block 3]
```

Then:

```python
b.block_id for b in allocated
```

produces:

```text
4
3
```

So:

```python
_owned["request_A"].extend([4, 3])
```

If initially:

```python
_owned["request_A"] = []
```

afterwards:

```python
_owned["request_A"] = [4, 3]
```

Meaning:

> Request A owns blocks 4 and 3.

---

# 17. Now the PagedKVCache

The `PagedKVCache` is the **per-request layer**.

Its description was:

> "Per-request block table and KV gather logic."

It bridges:

```text
Non-contiguous physical blocks
             ↓
       PagedKVCache
             ↓
Contiguous/logical representation
             ↓
HuggingFace model
```

Its two important operations are:

```text
write_token()
gather()
```

---

# 19. What is `block_table`?

For a particular request, the `PagedKVCache` maintains something like:

```text
block_table = [Block 4, Block 3, Block 2]
```

This means:

```text
Logical block 0 → Physical Block 4
Logical block 1 → Physical Block 3
Logical block 2 → Physical Block 2
```

The blocks don't need to be physically adjacent.

That's the **paging** idea.

---

# 20. Why is this useful?

Suppose physical memory looks like:

```text
Block 0 → Request B
Block 1 → Request C
Block 2 → Request A
Block 3 → Request B
Block 4 → Request A
```

Request A can still have:

```text
block_table = [4, 2]
```

Its logical sequence is continuous even though the physical blocks are scattered.

---

# 21. `write_token()`

The purpose of your function is:

> **Take one token's K/V values produced by the model and store them into the current physical KV block.**

The model gives:

```text
past_key_values
```

which contains:

```text
Layer 0 → K,V
Layer 1 → K,V
Layer 2 → K,V
...
```

For each layer, the K and V tensors contain values for the sequence.

---

# 22. What does `past_key_values` look like?

Conceptually:

```text
past_key_values
│
├── Layer 0 → (K0, V0)
├── Layer 1 → (K1, V1)
├── Layer 2 → (K2, V2)
└── ...
```

For one layer:

```text
K shape:

(batch, num_heads, seq_len, head_dim)

V shape:

(batch, num_heads, seq_len, head_dim)
```

Example:

```text
batch = 1
num_heads = 2
seq_len = 5
head_dim = 4
```

Then:

```text
K = (1, 2, 5, 4)
V = (1, 2, 5, 4)
```

---

# 23. Selecting one token

code:

```python
k_token = k[0, :, token_position, :]
v_token = v[0, :, token_position, :]
```

Suppose:

```python
token_position = 4
```

Then:

```text
k[0, :, 4, :]
```

means:

```text
batch 0
all heads
token position 4
all head dimensions
```

So:

```text
(1, num_heads, seq_len, head_dim)
```

becomes:

```text
(num_heads, head_dim)
```

for that one token.

---

# 24. Why loop over layers?

code:

```python
for layer_idx, (k, v) in enumerate(past_key_values):
```

goes through:

```text
Layer 0
Layer 1
Layer 2
...
```

For each layer:

```text
extract token's K/V
        ↓
write into same physical block
        ↓
at that layer's slice
```

So one token's storage looks conceptually like:

```text
Block 3
│
├── Layer 0 → K/V for T4
├── Layer 1 → K/V for T4
├── Layer 2 → K/V for T4
└── Layer 3 → K/V for T4
```

---

# 25. `current_block`

code:

```python
block_index = token_position // self.block_size
current_block = self.block_table[block_index]
```

means:

> Take the last block assigned to this request as the current block.

Example:

```text
block_table = [Block 4, Block 3]
```

Then:

```python
self.block_table[block_index]
```

gives:

```text
Block 3
```

So new token KV gets written into Block 3.

Your surrounding logic needs to make sure a new block is allocated when the current one becomes full.

---

# 26. `increment_filled()`

After writing one token:

```python
current_block.increment_filled()
```

updates:

```text
num_filled
```

For example:

```text
Before:
num_filled = 1

Write T5

After:
num_filled = 2
```

---

# 27. `seq_len`

Then:

```python
self.seq_len += 1
```

tracks the total sequence length for the request.

Important distinction:

```text
num_filled
→ tokens currently stored in THIS block

seq_len
→ total tokens in THIS request
```

Example:

```text
Block 4 → 4 tokens
Block 3 → 2 tokens

seq_len = 6
```

---

# 28. Prefill stage

Now let's connect everything.

Suppose the user sends:

```text
"Hello my name is John"
```

The tokenizer runs first.

For example:

```text
"Hello my name is John"
            ↓
[T0 T1 T2 T3 T4]
```

Now we know:

```text
seq_len = 5
```

**before the prefill forward pass.**

This is the question:

> How do we know Request A needs 5 tokens?

Because the **input is tokenized before/during prefill setup**.

---

# 29. Determine how many blocks are needed

Suppose:

```text
block_size = 4
```

Input length:

```text
5 tokens
```

Therefore:

```text
5 / 4 → needs 2 blocks
```

Conceptually:

```text
ceil(5 / 4) = 2
```

So:

```python
block_manager.allocate(
    request_id="request_A",
    num_blocks=2
)
```

---

# 30. The blocks already existed

This is very important.

Before Request A arrives:

```text
Block 0 → FREE
Block 1 → FREE
Block 2 → FREE
Block 3 → FREE
Block 4 → FREE
```

These were created when `BlockManager` was initialized.

Then Request A arrives.

The manager assigns:

```text
Block 4
Block 3
```

Now:

```text
Block 4 → allocated to A but EMPTY
Block 3 → allocated to A but EMPTY
```

**Allocation does not mean the blocks already contain A's KV.**

It only means:

> "These physical memory regions are reserved for Request A."

---

# 31. Then prefill happens

Now the model processes:

```text
T0 T1 T2 T3 T4
```

Each layer processes the sequence.

For example:

```text
Layer 0 → T0 T1 T2 T3 T4
Layer 1 → T0 T1 T2 T3 T4
Layer 2 → T0 T1 T2 T3 T4
...
```

Each layer produces K/V for those token positions.

So `past_key_values` contains:

```text
Layer 0 → K/V for T0-T4
Layer 1 → K/V for T0-T4
Layer 2 → K/V for T0-T4
...
```

---

# 32. Store the KV into blocks

Now `write_token()` takes the model-produced KV and stores it.

Conceptually:

```text
T0 → Block 4 position 0
T1 → Block 4 position 1
T2 → Block 4 position 2
T3 → Block 4 position 3

T4 → Block 3 position 0
```

Final layout:

```text
Block 4
┌────┬────┬────┬────┐
│ T0 │ T1 │ T2 │ T3 │
└────┴────┴────┴────┘

Block 3
┌────┬────┬────┬────┐
│ T4 │    │    │    │
└────┴────┴────┴────┘
```

But remember:

**Each `T` represents K/V data for every layer inside that block.**

---

# 33. Prefill is now complete

At the end of prefill:

```text
Request A
│
└── PagedKVCache
      │
      ├── Block 4
      │    └── T0 T1 T2 T3
      │
      └── Block 3
           └── T4
```

The KV cache remains in GPU memory.

We don't throw it away.

---

# 34. Decode stage

Now the model generates a new token.

Suppose:

```text
T5 = "How"
```

The previous KV is already cached:

```text
T0 T1 T2 T3 T4
```

We don't need to recompute their K/V.

The model processes the new token:

```text
T5
 ↓
model
 ↓
new K/V for T5
```

Then we store T5:

```text
Block 3
┌────┬────┬────┬────┐
│ T4 │ T5 │    │    │
└────┴────┴────┴────┘
```

---

# 35. What happens when a block becomes full?

Suppose:

```text
Block 3
┌────┬────┬────┬────┐
│ T4 │ T5 │ T6 │ T7 │
└────┴────┴────┴────┘
```

Now it is full.

Suppose another token T8 arrives.

`PagedKVCache` needs another physical block.

So it asks:

```text
BlockManager
    ↓
allocate another block
```

Suppose it gets Block 2.

Now:

```text
block_table = [4, 3, 2]
```

and:

```text
Block 2 → T8
```

So blocks can grow dynamically during decode.

---

# 36. Why is it called "paged"?

Because the logical sequence doesn't have to occupy contiguous physical memory.

Logical view:

```text
T0 T1 T2 T3 | T4 T5 T6 T7 | T8
```

Physical view:

```text
Block 4     | Block 3     | Block 2
```

Maybe physical memory actually looks like:

```text
Block 0 → Request B
Block 1 → FREE
Block 2 → Request A
Block 3 → Request A
Block 4 → Request A
```

The request still sees a continuous logical sequence.

This is similar to pages in virtual memory.

---

# 37. `gather()`

`PagedKVCache` also has:

```text
gather()
```

Its purpose is to take the request's physical blocks:

```text
[Block 4, Block 3, Block 2]
```

and assemble the KV data in logical token order:

```text
T0 T1 T2 T3 T4 T5 T6 T7 T8
```

This is useful when the HuggingFace interface expects a contiguous `past_key_values` representation.

Conceptually:

```text
Physical blocks
      ↓
Block 4 + Block 3 + Block 2
      ↓
    gather()
      ↓
Logical contiguous KV
      ↓
HuggingFace forward
```

Note: A true paged-attention implementation may avoid physically gathering everything by letting the attention kernel read the blocks directly, but **This class's stated design uses `gather()` as the bridge to the representation HuggingFace expects.**

---

# 39. Complete architecture

Put everything together:

```text
                         USER TEXT
                            │
                            ↓
                        Tokenizer
                            │
                            ↓
                   [T0 T1 T2 T3 T4]
                            │
                            ↓
                  Know sequence length
                            │
                            ↓
                 Calculate blocks needed
                            │
                            ↓
                    BlockManager
                            │
                     allocate blocks
                            │
             ┌──────────────┴──────────────┐
             ↓                             ↓
         KVBlock                         KVBlock
         Block 4                         Block 3
         EMPTY                           EMPTY
             │                             │
             └──────────────┬──────────────┘
                            ↓
                         PREFILL
                            │
                            ↓
                     Transformer
                            │
          ┌─────────────────┼────────────────┐
          ↓                 ↓                ↓
       Layer 0           Layer 1          Layer 2
          │                 │                │
        K/V               K/V              K/V
          │                 │                │
          └─────────────────┼────────────────┘
                            ↓
                    past_key_values
                            │
                            ↓
                       write_token()
                            │
                            ↓
                     Physical blocks
                            │
             ┌──────────────┴──────────────┐
             ↓                             ↓
          Block 4                        Block 3
       T0 T1 T2 T3                         T4
                            │
                            ↓
                         DECODE
                            │
                       New token T5
                            │
                            ↓
                  Reuse previous KV
                            +
                  compute new K/V
                            │
                            ↓
                       write_token()
                            │
                            ↓
                       Block 3
                    T4 T5 T6 ...
```

---

# 40. The three classes — final summary

## `KVBlock`

**Job: actual storage.**

```text
KVBlock
│
├── block_id
├── block_size
├── num_layers
├── num_heads
├── head_dim
├── num_filled
├── k_cache
└── v_cache
```

One block stores:

```text
K/V for block_size tokens
across ALL transformer layers
```

---

## `BlockManager`

**Job: memory management.**

It:

```text
creates blocks
      ↓
keeps all blocks in _pool
      ↓
tracks free blocks with _free_stack
      ↓
allocates blocks to requests
      ↓
tracks ownership with _owned
      ↓
frees/reuses blocks
```

Remember:

```text
_pool
→ all blocks

_free_stack
→ available blocks

_owned
→ which request owns which blocks
```

---

## `PagedKVCache`

**Job: per-request KV organization.**

It:

```text
maintains block_table
        ↓
knows which physical blocks belong to request
        ↓
write_token()
        ↓
stores model-produced K/V into blocks
        ↓
gather()
        ↓
reconstructs logical KV representation
```

---

# 41. The most important mental model

If you remember only this, you're good:

```text
                 KV CACHE SYSTEM

KVBlock
   │
   │  "I am the actual memory."
   ↓
┌───────────────────────────────┐
│ Layer 0 → K/V                 │
│ Layer 1 → K/V                 │
│ Layer 2 → K/V                 │
│ ...                           │
│ Layer N → K/V                 │
│                               │
│ for block_size tokens         │
└───────────────────────────────┘


BlockManager
   │
   │ "I manage all these blocks."
   ↓
[Block 0][Block 1][Block 2][Block 3]...


PagedKVCache
   │
   │ "I manage the blocks for Request A."
   ↓
Request A → [Block 4, Block 3, Block 2]
```

And the lifecycle is:

```text
USER INPUT
    ↓
TOKENIZE
    ↓
Know number of input tokens
    ↓
Calculate number of blocks needed
    ↓
BlockManager allocates already-created blocks
    ↓
PREFILL
    ↓
Every layer produces K/V
    ↓
write_token() stores K/V in physical blocks
    ↓
KV CACHE PERSISTS
    ↓
DECODE
    ↓
New token
    ↓
Reuse old KV + compute new K/V
    ↓
write new K/V into blocks
    ↓
If block is full → BlockManager gives another block
    ↓
Continue generating
```

### One sentence to memorize

> **`KVBlock` provides the physical KV storage, `BlockManager` manages and allocates those physical blocks, and `PagedKVCache` maps a request's logical token sequence onto those blocks and stores the K/V produced by each transformer layer.**

