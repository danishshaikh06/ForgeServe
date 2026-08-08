# Inference Pipeline

## Introduction

When you ask a Large Language Model (LLM) a question, the model does not immediately return an answer.

Instead, it follows a sequence of steps known as the **inference pipeline**.

Understanding this pipeline is important because every optimization in modern inference engines improves one or more of these steps.

In this document, we will walk through the complete journey from a user prompt to the final generated response.

---

# The Complete Pipeline

The complete inference pipeline in ForgeServe is shown below.

```text id="rwj6yy"
User Prompt
      │
      ▼
Tokenization
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
Decode Tokens
      │
      ▼
Generated Text
```

Each stage has a specific responsibility.

---

# Step 1 – User Prompt

Everything starts with the user's input.

Example:

```text id="shtjlwm"
What is Artificial Intelligence?
```

At this point, the model cannot understand plain text.

The text must first be converted into tokens.

---

# Step 2 – Tokenization

The tokenizer converts text into token IDs.

Example:

```text id="czt9k0"
"What is AI?"

↓

[392, 374, 15592, 30]
```

These token IDs become the input to the transformer model.

---

# Step 3 – Forward Pass

The token IDs are passed through the language model.

The transformer processes the input and predicts the next possible token.

The output of this step is a tensor called **logits**.

---

# Step 4 – Logits

Logits are scores for every token in the model's vocabulary.

For example:

```text id="y7wqsi"
Token        Score
----------------------
is            2.3
AI            8.7
the           1.5
computer      6.2
```

A higher score means the model considers that token more likely.

The model does not choose a token by itself.

It only produces these scores.

---

# Step 5 – Sampling

The logits are sent to a sampler.

In Phase 1, ForgeServe uses **Greedy Sampling**.

The sampler simply selects the token with the highest score.

```text id="r6fkl7"
Logits

↓

Argmax

↓

Next Token
```

The selected token becomes part of the generated response.

---

# Step 6 – Append the Token

The newly generated token is added to the existing sequence.

For example:

Before:

```text id="31kdb8"
What is AI
```

Generated token:

```text id="dgy0dz"
?
```

After appending:

```text id="uzxgtf"
What is AI?
```

The updated sequence is now used for the next prediction.

---

# Step 7 – Repeat

The model repeats the same process.

```text id="wlkzyh"
Forward Pass

↓

Logits

↓

Sampler

↓

Append Token
```

Each iteration generates one new token.

This continues until:

* An End-of-Sequence (EOS) token is generated.
* The maximum number of new tokens is reached.

This process is called **autoregressive generation**.

---

# Step 8 – Decode Tokens

After generation finishes, the sequence of token IDs is converted back into human-readable text.

Example:

```text id="vjlwmz"
[392, 374, 15592, 30]

↓

"What is AI?"
```

The decoded text is returned to the user.

---

# The Complete Flow Inside ForgeServe

The following diagram shows how the main components work together.

```text id="sd4gcv"
User Prompt
      │
      ▼
Generation Engine
      │
      ▼
Runtime
      │
      ▼
Tokenizer
      │
      ▼
Transformer Model
      │
      ▼
Logits
      │
      ▼
Greedy Sampler
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

Each component has a single responsibility, making the system modular and easy to extend.

---

# Current Limitation

The Phase 1 implementation is correct, but it is not optimized.

During every generation step, the model processes the entire sequence again, even though only one new token has been added.

For example:

```text id="a2qjlwm"
Forward(100)

↓

Forward(101)

↓

Forward(102)

↓

Forward(103)
```

Most of the computation is repeated.

Modern inference engines avoid this using a **KV Cache**, which stores previously computed information and reuses it during generation.

This optimization will be introduced in the next phase of ForgeServe.

---

# Summary

The inference pipeline is the complete process of generating text using a language model.

The pipeline consists of:

1. Tokenizing the input.
2. Running a forward pass through the model.
3. Producing logits.
4. Selecting the next token using a sampling strategy.
5. Appending the generated token.
6. Repeating the process until generation is complete.
7. Decoding the final token IDs back into text.

Every optimization that follows in ForgeServe builds on this same pipeline. The overall process remains the same—the goal is simply to make each step faster and more efficient without changing the generated output.
