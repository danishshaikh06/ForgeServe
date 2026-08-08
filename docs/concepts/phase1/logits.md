# Logits

## Introduction

When a language model performs a forward pass, it does **not** directly generate the next word.

Instead, it produces a set of numbers called **logits**.

A logit is simply a **score** that represents how likely the model thinks each token is to be the next token.

These scores are passed to the Sampler, which decides which token should be generated.

---

# Understanding Logits

Imagine a vocabulary containing only four tokens.

```text id="x1b6z9"
Token ID    Token
----------------------
0           cat
1           dog
2           bird
3           fish
```

After a forward pass, the model might produce:

```text id="hk5vdl"
Token      Logit
----------------------
cat         1.2
dog         5.8
bird        0.7
fish        3.1
```

These numbers are the logits.

They represent the model's confidence for each possible next token.

A larger logit means the model considers that token more likely.

---

# Logits Are Not Probabilities

One common misunderstanding is that logits are probabilities.

They are **not**.

Logits can be:

* Positive
* Negative
* Larger than 1
* Smaller than 0

For example:

```text id="uqbjv7"
Token      Logit
----------------------
cat        -2.4
dog         8.1
bird        -0.5
fish         3.7
```

These values do not add up to 1, so they cannot be interpreted as probabilities.

They are simply scores.

---

# Converting Logits to Probabilities

If probabilities are needed, the logits are passed through the **Softmax** function.

For example:

Logits:

```text id="yjlwm7"
[1.0, 2.0, 5.0]
```

After applying Softmax:

```text id="x14uhd"
[0.017, 0.047, 0.936]
```

Now the values:

* Are between 0 and 1.
* Add up to 1.
* Can be interpreted as probabilities.

Many sampling strategies, such as Temperature Sampling and Top-p Sampling, work with these probabilities.

---

# Why Doesn't Greedy Sampling Use Softmax?

In Phase 1, ForgeServe uses **Greedy Sampling**.

Greedy Sampling simply selects the token with the highest score.

For example:

```text id="7x2te4"
Logits

[1.0, 2.0, 5.0]
```

The largest value is:

```text id="49txd3"
5.0
```

So the selected token is the one at index:

```text id="ixvxyo"
2
```

Applying Softmax would produce:

```text id="qwp8lo"
[0.017, 0.047, 0.936]
```

The largest value is still at index **2**.

Since Softmax does not change which value is the largest, Greedy Sampling skips this extra computation and directly applies `argmax` to the logits.

---

# Logits in ForgeServe

Inside ForgeServe, the generation process looks like this:

```text id="0g44gf"
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
```

The Runtime performs the forward pass and returns the logits.

The Generation Engine extracts the logits for the last generated position.

The Sampler then selects the next token based on those logits.

---

# Why Are Logits Important?

Logits are the bridge between the language model and the sampling strategy.

The model predicts **how likely** every token is.

The Sampler decides **which token** to generate.

Separating these responsibilities keeps the inference pipeline modular and allows different sampling strategies to be implemented without changing the model.

---

# Summary

Logits are the raw scores produced by a language model for every token in its vocabulary.

They are not probabilities, but they indicate the model's confidence for each possible next token.

In ForgeServe, logits are passed to the Sampler, which uses them to choose the next token.

Understanding logits is essential because every decoding strategy starts from these model outputs, even though each strategy uses them in a different way.
