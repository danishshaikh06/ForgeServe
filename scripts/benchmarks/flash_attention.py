"""
Phase 3 benchmark: EAGER vs SDPA (FlashAttention).

Scenarios designed to stress FlashAttention specifically:
    - Long prompts stress prefill attention (seq² matrix)
    - Long generation stresses decode attention
    - Larger model has more heads and layers

FlashAttention benefit scales as:
    O(seq_len² × num_heads × num_layers)

Short sequences → small benefit (what we saw before)
Long sequences  → large benefit (what we measure here)
"""

import gc

import torch
from report import print_comparison_eager_sdpa
from runner import run_kvcache_scenario

from forgeserve.engine.config import GenerationConfig
from forgeserve.model.runtime import Runtime
from forgeserve.model.types import AttentionImplementation
from forgeserve.sampler.greedy import GreedySampler

# ── Long prompt construction ──────────────────────────────────────────────────
# We build prompts that are genuinely long by asking for detailed analysis.
# This is more realistic than padding with repeated text.

LONG_PROMPT_800 = (
    "You are a technical educator. Provide an extremely detailed and comprehensive "
    "explanation of the following topics. For each topic, cover the mathematical "
    "foundations, practical implementation considerations, historical context, and "
    "current state of the art.\n\n"
    "Topic 1: Transformer attention mechanisms. Explain scaled dot-product attention "
    "from first principles, derive the mathematical formulation, explain why we scale "
    "by the square root of head dimension, describe multi-head attention and how "
    "multiple heads allow the model to attend to different representation subspaces, "
    "and explain the computational complexity in terms of sequence length.\n\n"
    "Topic 2: Positional encodings. Cover sinusoidal encodings from the original "
    "attention is all you need paper, explain why position information is necessary "
    "for transformers unlike recurrent networks, describe learned positional embeddings, "
    "and explain rotary position embeddings used in modern models like LLaMA and Qwen "
    "including the mathematical formulation and why they generalize better to longer "
    "sequences than absolute position encodings.\n\n"
    "Topic 3: KV cache and inference optimization. Explain why autoregressive generation "
    "is computationally expensive without caching, describe how key-value caching works "
    "during the prefill and decode phases, analyze the memory requirements of KV cache "
    "as a function of sequence length batch size number of layers and head dimension, "
    "and explain the memory bandwidth versus compute trade-offs during inference.\n\n"
    "Topic 4: FlashAttention. Explain the memory hierarchy of modern GPUs including "
    "HBM SRAM and registers, describe why standard attention is memory bandwidth bound "
    "rather than compute bound, explain the tiling strategy used by FlashAttention to "
    "keep intermediate results in SRAM, describe online softmax and why it enables "
    "block-wise computation, and quantify the memory complexity improvement from O(N^2) "
    "to O(N).\n\n"
    "Begin your comprehensive explanation:"
)

MEDIUM_PROMPT_400 = (
    "Provide a detailed technical explanation covering transformer architecture internals. "
    "Start with the embedding layer and how token IDs are converted to dense vectors. "
    "Then explain layer normalization and why it is applied before attention in modern "
    "pre-norm architectures. Describe the complete attention computation step by step "
    "including the linear projections for queries keys and values, the scaled dot product, "
    "the softmax operation, and the final value aggregation. Explain residual connections "
    "and why they are essential for training deep networks. Cover the feed-forward network "
    "component including the role of the up projection, activation function, and down "
    "projection. Finally explain how the language model head converts hidden states to "
    "vocabulary logits and how temperature and sampling strategies affect generation. "
    "Be thorough and include mathematical notation where appropriate. "
    "Explain each component as if teaching a graduate student seeing transformers for "
    "the first time. Cover edge cases and common implementation pitfalls as well."
)


SCENARIOS = [
    # (prompt, max_new_tokens, label)
    # Short baseline — what we measured before
    (
        "Explain how transformer attention mechanisms work.",
        100,
        "short_prompt_100gen",
    ),
    # Medium prompt, medium generation
    (
        MEDIUM_PROMPT_400,
        200,
        "medium_prompt_400tok_200gen",
    ),
    # Long prompt, short generation — stresses prefill attention
    (
        LONG_PROMPT_800,
        50,
        "long_prompt_800tok_50gen",
    ),
    # Long prompt, long generation — stresses both prefill and decode
    (
        LONG_PROMPT_800,
        300,
        "long_prompt_800tok_300gen",
    ),
]

WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def unload_runtime(runtime: Runtime) -> None:
    """Unload runtime from GPU memory before loading next model."""
    del runtime.model
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run(model_name: str) -> None:
    print("\nForgeServe Phase 3 Benchmark — EAGER vs SDPA")
    print(f"Model: {model_name}")
    print(f"Warmup: {WARMUP_RUNS} | Benchmark: {BENCHMARK_RUNS}")
    print("Scenarios designed to stress FlashAttention\n")

    sampler = GreedySampler()

    for prompt, max_new_tokens, label in SCENARIOS:

        # Approximate token count for display
        approx_tokens = len(prompt.split()) * 1.3
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            system_prompt=None,
        )

        print(f"\n>>> Scenario: {label}")
        print(f"    Prompt words: {len(prompt.split())}  (~{approx_tokens:.0f} tokens)")
        print(f"    Max generate: {max_new_tokens} tokens")

        # ── EAGER ─────────────────────────────────────────────
        print("  Loading EAGER runtime...")
        eager_runtime = Runtime(model_name, AttentionImplementation.EAGER)

        eager_result = run_kvcache_scenario(
            runtime=eager_runtime,
            sampler=sampler,
            prompt=prompt,
            config=config,
            engine_name="eager",
            warmup_runs=WARMUP_RUNS,
            benchmark_runs=BENCHMARK_RUNS,
        )

        print("  Unloading EAGER runtime...")
        unload_runtime(eager_runtime)

        # ── SDPA ──────────────────────────────────────────────
        print("  Loading SDPA runtime...")
        sdpa_runtime = Runtime(model_name, AttentionImplementation.SDPA)

        sdpa_result = run_kvcache_scenario(
            runtime=sdpa_runtime,
            sampler=sampler,
            prompt=prompt,
            config=config,
            engine_name="sdpa",
            warmup_runs=WARMUP_RUNS,
            benchmark_runs=BENCHMARK_RUNS,
        )

        print("  Unloading SDPA runtime...")
        unload_runtime(sdpa_runtime)

        #  Results
        print_comparison_eager_sdpa(eager_result, sdpa_result)

        # Interpretation guide
        speedup = eager_result.mean_tpot_ms / sdpa_result.mean_tpot_ms
        mem_delta = sdpa_result.mean_peak_memory_mb - eager_result.mean_peak_memory_mb

        if speedup > 1.3:
            print(f"  ✅ Strong FlashAttention benefit: {speedup:.2f}x TPOT speedup")
        elif speedup > 1.1:
            print(f"  ✅ Moderate FlashAttention benefit: {speedup:.2f}x TPOT speedup")
        else:
            print(f"  ℹ  Small benefit ({speedup:.2f}x) — sequence may still be short")

        if mem_delta < -50:
            print(f"  ✅ Memory saving: {abs(mem_delta):.0f} MB — FlashAttention avoiding HBM writes")
        elif mem_delta < -10:
            print(f"  ✅ Small memory saving: {abs(mem_delta):.0f} MB")
        else:
            print("  ℹ  Memory delta negligible — sequence too short for visible saving")


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    run(model)
