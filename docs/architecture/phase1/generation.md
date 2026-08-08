# Generation Engine

## Introduction

The Generation Engine is the heart of ForgeServe.

Its job is to coordinate the complete text generation process.

It does **not** know how to tokenize text.

It does **not** know how the transformer model works internally.

It does **not** decide which token to generate.

Instead, it brings all the components together and controls the flow of inference.

---

# Why do we need a Generation Engine?

Generating text is not a single function call.

The model predicts only **one token at a time**.

To generate an entire sentence, we need to repeatedly:

1. Run the model.
2. Select the next token.
3. Add that token to the input.
4. Repeat until generation stops.

The Generation Engine manages this entire process.

---

# Generation Pipeline

The complete pipeline looks like this:

```text
Prompt
   │
   ▼
Runtime.tokenize()
   │
   ▼
Forward Pass
   │
   ▼
Logits
   │
   ▼
Sampler
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
GenerationResponse
```

Every generated token goes through this pipeline.

---

# Responsibilities

The Generation Engine has five main responsibilities.

## 1. Tokenize the Prompt

The input text is converted into token IDs using the Runtime.

Example:

```text
Hello!

↓

[9707, 0]
```

These token IDs become the initial input to the model.

---

## 2. Run the Forward Pass

The Generation Engine asks the Runtime to perform a forward pass.

The model returns logits for every token in the vocabulary.

---

## 3. Select the Next Token

The logits are passed to a Sampler.

In Phase 1 we use Greedy Sampling.

The sampler selects the token with the highest score.

```text
Logits

↓

Argmax

↓

Next Token ID
```

---

## 4. Append the Token

The selected token is added to the existing sequence.

Example:

Before:

```text
[1, 25, 89]
```

Predicted token:

```text
41
```

After appending:

```text
[1, 25, 89, 41]
```

The updated sequence becomes the input for the next iteration.

---

## 5. Stop Generation

The Generation Engine repeats the process until one of the following happens:

* The model generates an End-of-Sequence (EOS) token.
* The maximum number of new tokens is reached.

At that point, the generated token IDs are converted back into text.

---

# Why is the Generation Engine Separate?

The Generation Engine only coordinates the process.

It does not perform tokenization.

It does not perform sampling.

It does not load models.

This separation makes the code easier to understand, test, and extend.

For example, if we replace Greedy Sampling with Top-k Sampling, the Generation Engine does not need to change.

Similarly, if we optimize the Runtime using KV Cache, the Generation Engine can continue using the same interface.

---

# Current Limitation

The current implementation recomputes the entire sequence during every generation step.

Suppose the prompt contains 100 tokens.

The model performs:

```text
Forward(100)

Forward(101)

Forward(102)

Forward(103)

...
```

Most of this computation is repeated.

This is inefficient.

Modern inference engines avoid this by storing previously computed attention information in a KV Cache.

---

# What's Next?

In the next phase, we will keep the Generation Engine's public interface exactly the same.

Instead of changing how users interact with ForgeServe, we will optimize what happens internally.

The first optimization will be **KV Cache**, which allows the model to reuse previous computations and generate tokens much faster.
