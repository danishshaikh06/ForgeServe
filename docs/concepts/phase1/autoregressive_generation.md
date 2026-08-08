# Autoregressive Generation

## Introduction

Large Language Models (LLMs) do not generate an entire sentence at once.

Instead, they generate **one token at a time**.

Each newly generated token is added to the existing input, and the updated sequence is used to predict the next token.

This process is called **autoregressive generation**.

The word *autoregressive* simply means:

> **The model uses everything it has generated so far to predict what comes next.**

---

# Understanding the Idea

Imagine the user enters the following prompt:

```text
The capital of France is
```

The model does not immediately produce the complete answer.

Instead, it works step by step.

### Step 1

Input:

```text
The capital of France is
```

Prediction:

```text
Paris
```

The sequence now becomes:

```text
The capital of France is Paris
```

---

### Step 2

The updated sequence is sent back to the model.

Input:

```text
The capital of France is Paris
```

Prediction:

```text
.
```

The sequence now becomes:

```text
The capital of France is Paris.
```

---

### Step 3

The updated sequence is sent back again.

The model may now predict an End-of-Sequence (EOS) token, which tells the generation process to stop.

---

# The Generation Process

Every generated token follows the same cycle.

```text
Prompt
   │
   ▼
Forward Pass
   │
   ▼
Predict Next Token
   │
   ▼
Append Token
   │
   ▼
Repeat
```

This loop continues until:

* An End-of-Sequence (EOS) token is generated, or
* The maximum number of new tokens is reached.

---

# Why One Token at a Time?

A transformer predicts the probability of the **next token only**.

It does not predict the next sentence or the next paragraph in a single forward pass.

For example:

Input:

```text
I love machine
```

The model predicts the next token.

Possible predictions:

```text
learning
```

```text
translation
```

```text
vision
```

Once one token is selected, it becomes part of the input, and the model predicts the next token again.

This process repeats until the response is complete.

---

# Why Is This Important?

Autoregressive generation is the foundation of modern language model inference.

Every inference engine follows this process.

Whether you use:

* Hugging Face Transformers
* vLLM
* TensorRT-LLM
* SGLang
* llama.cpp
* ForgeServe

the model still generates one token at a time.

The difference between these systems is **how efficiently** they perform this loop.

---

# The Challenge

Suppose the prompt contains 100 tokens.

To generate one additional token, the model processes all 100 tokens.

To generate another token, it processes 101 tokens.

Then 102.

Then 103.

```text
Forward(100)

↓

Forward(101)

↓

Forward(102)

↓

Forward(103)

↓

...
```

Much of this computation is repeated even though most of the input has not changed.

This repeated work makes text generation slower than necessary.

---

# How Modern Inference Engines Solve This

Modern inference engines avoid recomputing everything for every new token.

Instead, they store information from previous forward passes and reuse it during generation.

This technique is called **KV Cache**.

By caching previously computed attention information, the model only performs the new computation required for the latest token.

The generated text stays exactly the same, but decoding becomes much faster.

KV Cache will be the focus of the next phase of ForgeServe.

---

# Summary

Autoregressive generation is the process of generating text one token at a time.

Each new token becomes part of the input for the next prediction.

Although this approach is simple, it introduces repeated computation during decoding.

Understanding this process is essential because every optimization in modern LLM inference is designed to make this generation loop faster without changing the generated output.
