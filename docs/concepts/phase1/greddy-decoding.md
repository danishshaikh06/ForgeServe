# Greedy Decoding

## Introduction

After a language model performs a forward pass, it produces **logits** for every token in its vocabulary.

The next step is deciding which token should be generated.

This decision is made by a **decoding strategy**.

In Phase 1 of ForgeServe, we use the simplest decoding strategy called **Greedy Decoding**.

Greedy Decoding always selects the token with the highest score.

---

# How Greedy Decoding Works

Suppose the model produces the following logits:

```text id="g1bq8l"
Token        Logit
-----------------------
Hello         2.1
Hi            1.4
Hey           0.9
Greetings     0.2
```

The highest logit belongs to:

```text id="qk0c8z"
Hello
```

Greedy Decoding selects that token.

No randomness is involved.

---

# The Algorithm

The algorithm is very simple.

```text id="4c5j8m"
Forward Pass
      │
      ▼
Get Logits
      │
      ▼
Find Maximum Value
      │
      ▼
Return Corresponding Token
```

In mathematical terms:

```text id="tkn2dh"
Next Token = argmax(logits)
```

This means:

> Select the token whose logit has the largest value.

---

# Example

Imagine the model predicts:

```text id="3w87p7"
Token ID      Logit
------------------------
0              1.0
1              3.4
2              8.7
3              2.1
```

The highest score is:

```text id="qmkcqn"
8.7
```

which belongs to:

```text id="y8h3v5"
Token ID 2
```

Therefore, Greedy Decoding selects:

```text id="jlwmpr"
Token ID 2
```

---

# Why Is It Called "Greedy"?

The algorithm always chooses the best option **at the current step**.

It never considers future possibilities.

For example:

```text id="yukg5f"
Current Scores

A : 8.5
B : 7.8
C : 4.2
```

Greedy Decoding immediately chooses:

```text id="hqyhmk"
A
```

It does not ask:

* Could choosing B lead to a better sentence later?
* Could choosing C produce a more creative response?

It simply selects the highest-scoring token.

This "best choice right now" behavior is why it is called *greedy*.

---

# Advantages

Greedy Decoding has several benefits.

* Very simple to understand.
* Fast to execute.
* Deterministic.
* Produces the same output for the same input.

Because it always chooses the highest-scoring token, there is no randomness.

---

# Limitations

Although Greedy Decoding is simple, it is not always the best choice.

It may produce:

* Repetitive text.
* Less creative responses.
* Locally optimal choices that are not globally optimal.

For tasks such as story generation or creative writing, other decoding strategies often produce better results.

---

# Other Decoding Strategies

Modern language models support many decoding methods.

Some common strategies include:

* Temperature Sampling
* Top-k Sampling
* Top-p (Nucleus) Sampling
* Beam Search

Unlike Greedy Decoding, these methods introduce controlled randomness or explore multiple candidate sequences to generate more diverse text.

ForgeServe will implement several of these strategies in future phases.

---

# Greedy Decoding in ForgeServe

In ForgeServe, the generation process follows this sequence:

```text id="x8z4ua"
Forward Pass
      │
      ▼
Logits
      │
      ▼
Greedy Decoder
      │
      ▼
Next Token
      │
      ▼
Append Token
      │
      ▼
Repeat
```

The Generation Engine asks the Sampler to select the next token.

The Greedy Sampler applies the `argmax` operation to the logits and returns the selected token ID.

---

# Summary

Greedy Decoding is the simplest decoding strategy used in language model inference.

It always selects the token with the highest logit score.

Because it is deterministic, fast, and easy to understand, it is an excellent starting point for learning how text generation works.

More advanced decoding strategies build upon the same idea but use different methods to choose the next token based on the model's logits.
