# Phase 1 — Building a Reference Inference Engine

## Introduction

In this phase, we built the first working version of ForgeServe.

The goal was **not** to make inference fast.

The goal was to understand how Large Language Models (LLMs) generate text by implementing the generation process ourselves.

A very important rule during this phase was:

> **Do not use `model.generate()`.**

Instead, we implemented every step manually.

By the end of this phase, ForgeServe can generate text using its own generation loop.

---

# Why are we doing this?

Libraries like Hugging Face provide a simple API:

```python
model.generate(...)
```

Although this is convenient, it hides many important details.

Inside `generate()` a lot of things happen:

* The prompt is tokenized.
* The model performs a forward pass.
* The next token is selected.
* The selected token is appended to the input.
* The process repeats until the model decides to stop.

Since all of this is hidden, it becomes difficult to understand how inference actually works.

ForgeServe removes this abstraction and implements every step ourselves.

---

# How does text generation work?

Suppose the user enters:

```
Hello! How are you?
```

The complete generation pipeline looks like this:

```
Prompt
   │
   ▼
Tokenize
   │
   ▼
Forward Pass
   │
   ▼
Logits
   │
   ▼
Greedy Sampling
   │
   ▼
Next Token
   │
   ▼
Append Token
   │
   ▼
Repeat
   │
   ▼
Decode
   │
   ▼
Generated Text
```

Every box in this pipeline is implemented inside ForgeServe.

---

# Components Built

## ModelLoader

The ModelLoader is responsible for loading the model and tokenizer.

Responsibilities:

* Load the model
* Load the tokenizer
* Move the model to the selected device
* Set the model to evaluation mode

The ModelLoader does nothing else.

---

## Runtime

The Runtime provides a simple interface for interacting with the model.

Responsibilities:

* Tokenize text
* Run the forward pass
* Decode generated tokens

Instead of letting the rest of the application communicate directly with Hugging Face, every interaction happens through the Runtime.

This keeps the project modular and easier to maintain.

---

## Sampler

After every forward pass, the model returns logits.

Logits are simply scores for every token in the vocabulary.

The sampler decides which token should be generated next.

During Phase 1 we implemented Greedy Sampling.

Greedy Sampling always selects the token with the highest score.

```
Logits

↓

Argmax

↓

Next Token
```

Later phases will introduce more advanced sampling methods such as Top-k, Top-p, and Temperature Sampling.

---

## Generation Engine

The Generation Engine is the heart of ForgeServe.

It connects all the other components together.

Its responsibilities are:

* Tokenize the prompt
* Run the forward pass
* Ask the sampler for the next token
* Append the token
* Stop when the EOS token is generated
* Decode the final output

The Generation Engine does not know how tokenization works.

It does not know how the model works internally.

It simply coordinates the generation process.

---

## Generation Response

Instead of returning only the generated text, ForgeServe returns a GenerationResponse object.

This makes it easy to extend the API in future phases.

Current fields:

* Generated text
* Number of generated tokens
* Finish reason

Future versions will include generation time, throughput, token ids, and other useful information.

---

# The Generation Loop

The core algorithm implemented in Phase 1 looks like this:

```
Tokenize Prompt

Repeat

    Forward Pass

    Get Last Logits

    Sample Next Token

    Append Token

Until

    EOS Token

or

    Maximum Number of Tokens

Decode Output
```

This is the same idea used by modern inference engines.

The difference is that production systems apply many optimizations to make this loop faster.

---

# Testing

Before moving to optimization, we added tests.

Three types of tests were written.

## Unit Tests

Unit tests verify that individual components work correctly.

Examples:

* Runtime
* Greedy Sampler
* Generation Engine

---

## Integration Tests

Integration tests verify that multiple components work together correctly.

---

## End-to-End Tests

These tests generate text using the real model and verify that ForgeServe successfully completes the entire inference pipeline.

---

# Current Limitation

Although our implementation is correct, it is not efficient.

Imagine a prompt containing 100 tokens.

If we generate one more token, the model processes all 100 tokens again.

When another token is generated, it processes 101 tokens.

Then 102.

Then 103.

This repeated computation makes decoding slower than necessary.

Modern inference engines solve this problem using a technique called KV Cache.

That will be the focus of the next phase.

---

# What We Learned

By completing Phase 1, we learned:

* How text is converted into tokens.
* How a transformer predicts the next token.
* What logits are.
* How greedy decoding works.
* How autoregressive text generation works.
* How different software components work together to build an inference engine.
* How to design clean and modular software using software engineering principles.

---

# What's Next?

Phase 2 introduces KV Cache.

Instead of recomputing attention for the entire sequence every time a new token is generated, the model will reuse previously computed information.

The generated text will remain exactly the same.

Only the execution becomes significantly faster.

Phase 1 gave us a correct implementation.

Phase 2 will teach us how modern inference engines achieve high performance.
