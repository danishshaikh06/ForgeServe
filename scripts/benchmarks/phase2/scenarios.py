"""
Phase 2 benchmark scenario.

Compares naive generation vs KV cache generation
across increasing sequence lengths.

Expected pattern:
    TPOT speedup should GROW as max_new_tokens increases.
    If it doesn't, something is wrong with the implementation.
"""

from forgeserve.model.runtime import Runtime
from forgeserve.sampler.greedy import GreedySampler
from forgeserve.engine.config import GenerationConfig
from forgeserve.model.loader import ModelLoader

from runner import run_scenario
from report import print_comparison, print_scenario_header

# Scenarios: (prompt_length_words, max_new_tokens)
# We use word counts and let the tokenizer determine exact token counts
SCENARIOS = [
    ("short prompt for testing",                              50),
    ("short prompt for testing",                             200),
    ("This is a medium length prompt that contains more context for the model to process during benchmarking",  50),
    ("This is a medium length prompt that contains more context for the model to process during benchmarking", 200),
]

WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def run(model_name: str) -> None:
    """
    Entry point for Phase 2 benchmarks.

    Args:
        model_name: HuggingFace model identifier to benchmark.
    """
    print(f"\nForgeServe Phase 2 Benchmark")
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