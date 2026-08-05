# Sampler

## Introduction

The Sampler is responsible for selecting the next token during text generation.

After every forward pass, the language model produces a set of scores called **logits**.

The Sampler looks at these logits and decides which token should be generated next.

The model predicts **what is possible**.

The Sampler decides **what to choose**.

---

# Why do we need a Sampler?

A language model does not directly generate words.

Instead, it produces a score for every token in its vocabulary.

For example:

```text
Token          Score
-------------------------
Hello          8.5
Hi             7.9
Goodbye        1.2
Apple          0.4
```

The model does not decide which token to output.

It only provides these scores.

The Sampler is responsible for converting these scores into the next token.

---

# Responsibilities

The Sampler has one responsibility:

* Receive the logits from the model.
* Select the next token.
* Return the selected token ID.

It does not know anything about:

* Tokenization
* The transformer model
* KV Cache
* Attention
* Text generation loops

Its only job is selecting the next token.

---

# Current Implementation

In Phase 1, ForgeServe implements **Greedy Sampling**.

Greedy Sampling simply selects the token with the highest score.

The process is:

```text
Logits
   │
   ▼
Argmax
   │
   ▼
Next Token ID
```

This method is simple, fast, and deterministic.

Given the same logits, it will always produce the same output.

---

# Why is the Sampler an Abstraction?

ForgeServe defines a base `Sampler` interface instead of directly implementing Greedy Sampling inside the Generation Engine.

This design makes the project easier to extend.

Today:

```text
Sampler
   │
   ▼
GreedySampler
```

In the future:

```text
Sampler
├── GreedySampler
├── TopKSampler
├── TopPSampler
├── TemperatureSampler
└── BeamSearchSampler
```

The Generation Engine does not need to know which sampling strategy is being used.

It simply asks the Sampler to select the next token.

This follows the **Open/Closed Principle** from software engineering:

* Open for extension.
* Closed for modification.

New sampling strategies can be added without changing the Generation Engine.

---

# Benefits of This Design

Separating sampling into its own component provides several advantages.

* Each class has a single responsibility.
* New sampling algorithms can be added easily.
* The Generation Engine remains simple.
* Individual samplers can be tested independently.
* The codebase becomes easier to maintain as the project grows.

---

# Current Limitations

Greedy Sampling always selects the token with the highest score.

While this makes generation deterministic, it can also make responses repetitive and less creative.

Many real-world applications use other sampling methods to produce more diverse outputs.

These include:

* Temperature Sampling
* Top-k Sampling
* Top-p (Nucleus) Sampling
* Beam Search

These strategies will be introduced in future phases of ForgeServe.

---

# Summary

The Sampler is a small but important component in ForgeServe.

It separates the decision of **which token to generate** from the rest of the inference pipeline.

This keeps the architecture modular, easy to understand, and ready for future extensions without requiring changes to the Generation Engine.
