from report import print_comparison, print_scenario_header
from runner import run_scenario

from forgeserve.engine.config import GenerationConfig
from forgeserve.logger import get_logger
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.greedy import GreedySampler


logger = get_logger(__name__)


# Controlled memory-scaling workloads
#
# The prompt is intentionally reasonably long and is kept the
# same for every experiment. Only max_new_tokens changes.
PROMPT = (
    "Explain in detail how transformer models work, including "
    "self-attention, queries, keys, values, multi-head attention, "
    "grouped query attention, positional encodings, residual "
    "connections, feed-forward networks, normalization, training, "
    "autoregressive inference, and why transformer architectures "
    "scale effectively for language modeling."
)


SCENARIOS = [
    (PROMPT, 100),
    (PROMPT, 250),
    (PROMPT, 300),
]


WARMUP_RUNS = 1
BENCHMARK_RUNS = 1


def run(model_name: str) -> None:
    """
    Entry point for the Phase 2 KV-cache scaling benchmark.
    """

    print("\nForgeServe Phase 2 — KV Cache Scaling Benchmark")
    print(f"Model: {model_name}")
    print(
        f"Warmup runs: {WARMUP_RUNS} | "
        f"Benchmark runs: {BENCHMARK_RUNS}"
    )

    print("\nControlled generation mode:")
    print("EOS stopping: disabled")
    print("max_new_tokens = actual requested decode steps")

    runtime = Runtime(
        model_name=model_name,
    )

    sampler = GreedySampler()

    for prompt, max_new_tokens in SCENARIOS:

        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            system_prompt=None,
        )

        # Tokenize ONLY for reporting the real token count.
        encoded = runtime.tokenize(
            prompt,
            config.system_prompt,
        )

        actual_prompt_tokens = encoded["input_ids"].shape[1]

        print(
            "\n>>> Scenario: "
            f"prompt={actual_prompt_tokens} tokens, "
            f"generate={max_new_tokens} tokens"
        )

        naive_result, kvcache_result = run_scenario(
            runtime=runtime,
            sampler=sampler,
            prompt=prompt,
            config=config,
            warmup_runs=WARMUP_RUNS,
            benchmark_runs=BENCHMARK_RUNS,
            stop_on_eos=False,
        )

        if naive_result is None:
            print("\nNaive: OOM / no successful runs")

        if kvcache_result is None:
            print("\nKV Cache: OOM / no successful runs")

        if naive_result is not None and kvcache_result is not None:
            print_comparison(
                naive_result,
                kvcache_result,
            )


if __name__ == "__main__":
    import sys

    model = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Qwen/Qwen2.5-0.5B-Instruct"
    )

    run(model)