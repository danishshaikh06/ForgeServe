# Runtime Architecture

## What is the Runtime?

The Runtime is the component that talks directly to the language model.

Its job is to provide a very small and clean interface to the rest of ForgeServe.

Think of it as a translator between our engine and Hugging Face.

---

# Why do we need a Runtime?

Without a Runtime, every part of the code would need to know how Hugging Face works.

For example:

```python
tokenizer(text, return_tensors="pt")
model(input_ids=input_ids, attention_mask=attention_mask)
tokenizer.decode(token_ids)
```

This would spread Hugging Face code across the entire project.

That makes the code harder to understand, harder to test, and harder to replace in the future.

Instead, ForgeServe centralizes all model interactions inside the Runtime.

---

# Runtime Responsibilities

The Runtime has only three responsibilities.

## 1. Tokenize text

```text
Text
  ↓
Token IDs
```

Method: `tokenize()`

---

## 2. Run a forward pass

```text
Token IDs
  ↓
Model
  ↓
Logits
```

Method: `forward()`

---

## 3. Decode token IDs back to text

```text
Token IDs
  ↓
Text
```

Method: `decode()`

---

# What the Runtime Does NOT Do

The Runtime does **not**:

* choose the next token
* run the generation loop
* append tokens
* stop on EOS
* implement sampling strategies
* manage requests

These responsibilities belong to other components.

Keeping responsibilities separate makes the architecture easier to maintain.

---

# Data Flow

The Runtime sits between the Generation Engine and the model.

```text
Generation Engine
       │
       ▼
    Runtime
       │
       ▼
 Hugging Face Model
```

The engine never talks directly to the model.

---

# Public API

The Runtime exposes three methods.

## tokenize()

```python
encoded = runtime.tokenize("Hello")
```

Returns a `BatchEncoding` containing `input_ids` and `attention_mask`.

---

## forward()

```python
outputs = runtime.forward(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
)
```

Returns the model output object containing logits.

---

## decode()

```python
text = runtime.decode(token_ids)
```

Converts token IDs back into readable text.

---

# Example

```python
runtime = Runtime("Qwen/Qwen2.5-0.5B-Instruct")

encoded = runtime.tokenize("Hello")

outputs = runtime.forward(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
)

text = runtime.decode(encoded["input_ids"][0])
```

This demonstrates the complete Runtime workflow.

---

# Why This Design Is Important

Imagine that one day we replace Hugging Face with another backend such as TensorRT-LLM.

Only the Runtime would need to change.

The Generation Engine and Sampler would continue to work exactly the same way.

This is an example of **encapsulation** and **dependency isolation**.

---

# Future Extensions

In later phases, the Runtime will gain additional methods:

* `prefill()`
* `decode_step()`
* KV cache support
* FlashAttention support
* Quantized execution support

The public API will remain stable while the internal implementation becomes more optimized.

---

# Key Takeaway

The Runtime is the single gateway between ForgeServe and the language model.

Its purpose is to hide model-specific details and provide a simple interface for tokenization, inference, and decoding.

If you remember one sentence from this document, remember this:

> **Runtime = Tokenize + Forward + Decode. Nothing more.**
