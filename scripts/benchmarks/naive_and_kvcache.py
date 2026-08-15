"""
Phase 2 benchmark scenario.

Compares naive generation vs KV cache generation
across increasing sequence lengths.

Expected pattern:
    TPOT speedup should GROW as max_new_tokens increases.
    If it doesn't, something is wrong with the implementation.
"""

from report import print_comparison, print_scenario_header
from runner import run_scenario

from forgeserve.engine.config import GenerationConfig
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.greedy import GreedySampler

# Scenarios: (prompt_length_words, max_new_tokens)
# We use word counts and let the tokenizer determine exact token counts
SCENARIOS = [
    (
        "Explain in detail how transformer attention mechanisms work, "
        "covering queries keys and values, the mathematical operations, "
        "and why the mechanism is effective for sequence modeling.",
        100,
    ),
    (
        "Explain in detail how transformer attention mechanisms work, "
        "covering queries keys and values, the mathematical operations, "
        "and why the mechanism is effective for sequence modeling.",
        300,
    ),
    (
        "Write a detailed technical explanation of how operating systems "
        "manage virtual memory, covering page tables, TLB, page faults, "
        "and the relationship between physical and virtual address spaces. "
        "Include examples of how modern systems like Linux handle this.",
        100,
    ),
    (
        "Write a detailed technical explanation of how operating systems "
        "manage virtual memory, covering page tables, TLB, page faults, "
        "and the relationship between physical and virtual address spaces. "
        "Include examples of how modern systems like Linux handle this.",
        300,
    ),
]

WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def run(model_name: str) -> None:
    """
    Entry point for Phase 2 benchmarks.

    Args:
        model_name: HuggingFace model identifier to benchmark.
    """
    print("\nForgeServe Phase 2 Benchmark")
    print(f"Model: {model_name}")
    print(f"Warmup runs: {WARMUP_RUNS} | Benchmark runs: {BENCHMARK_RUNS}")

    runtime = Runtime(model_name=model_name)
    sampler = GreedySampler()

    for prompt, max_new_tokens in SCENARIOS:
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            system_prompt=None,
        )

        print_scenario_header(
            prompt_tokens=len(prompt.split()),  # approximate
            max_new_tokens=max_new_tokens,
        )

        naive_result, kvcache_result = run_scenario(
            runtime=runtime,
            sampler=sampler,
            prompt=prompt,
            config=config,
            warmup_runs=WARMUP_RUNS,
            benchmark_runs=BENCHMARK_RUNS,
        )

        print_comparison(naive_result, kvcache_result)


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    run(model)
