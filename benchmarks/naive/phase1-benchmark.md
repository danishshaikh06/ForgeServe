# Phase 1 Benchmark

## Introduction

This document records the performance of the Phase 1 implementation of ForgeServe.

The purpose of this benchmark is **not** to achieve the fastest possible inference.

Instead, it establishes a **baseline** that future optimizations can be compared against.

In Phase 1, ForgeServe performs autoregressive text generation without using optimizations such as KV Cache or FlashAttention.

Every new token requires a complete forward pass over the entire sequence.

---

# Benchmark Configuration

| Setting            | Value                      |
| ------------------ | -------------------------- |
| Model              | Qwen/Qwen2.5-0.5B-Instruct |
| Prompt             | "Hello! how are you?"      |
| Prompt Tokens      | 6                          |
| Maximum New Tokens | 100                        |
| Generation Method  | Greedy Decoding            |
| KV Cache           | Disabled                   |
| FlashAttention     | Disabled                   |
| PagedAttention     | Disabled                   |

---

# Hardware

| Component        | Value                   |
| ---------------- | ----------------------- |
| GPU              | NVIDIA GeForce RTX 4070 |
| Framework        | PyTorch                 |
| Inference Engine | ForgeServe Phase 1      |

---

# Benchmark Result

| Metric           |               Value |
| ---------------- | ------------------: |
| Prompt Tokens    |                   6 |
| Generated Tokens |                  27 |
| Total Latency    |       1.432 seconds |
| Throughput       | 18.86 tokens/second |
| Finish Reason    |                 EOS |

---

# Generated Response

```text
Hello! I'm just a machine learning model, so I don't have feelings or emotions. How can I assist you today?
```

---

# Understanding the Results

### Prompt Tokens

The tokenizer converted the input prompt into **6 tokens** before passing it to the model.

---

### Generated Tokens

The model generated **27 new tokens** before reaching the End-of-Sequence (EOS) token.

---

### Latency

The complete generation process took approximately **1.432 seconds**.

This includes:

* Tokenization
* Multiple forward passes
* Greedy decoding
* Token appending
* Final decoding

---

### Throughput

The model generated approximately **18.86 tokens every second**.

This value is commonly used to measure inference performance.

A higher value indicates faster text generation.

---

# Why Is Phase 1 Relatively Slow?

The Phase 1 implementation recomputes attention for the entire sequence after generating every new token.

For example:

```text
Forward(6)

↓

Forward(7)

↓

Forward(8)

↓

Forward(9)

↓

...
```

Although only one new token is generated each iteration, the model processes the entire sequence again.

This repeated computation increases latency and reduces throughput.

---

# Future Comparison

This benchmark serves as the reference point for all future optimizations.

As new features are implemented, this table will be expanded.

|   Phase | Optimization        | Tokens/sec | Improvement |
| ------: | ------------------- | ---------: | ----------: |
| Phase 1 | Reference Engine    |      18.86 |    Baseline |
| Phase 2 | KV Cache            |        TBD |         TBD |
| Phase 3 | FlashAttention      |        TBD |         TBD |
| Phase 4 | PagedAttention      |        TBD |         TBD |
| Phase 5 | Continuous Batching |        TBD |         TBD |

---

# Conclusion

Phase 1 successfully establishes a correct and modular reference implementation of the ForgeServe inference engine.

Although performance is not yet optimized, the system provides a reliable baseline for future development.

All upcoming optimization phases will be evaluated against these benchmark results to measure their impact on latency and throughput while ensuring that the generated output remains correct.
