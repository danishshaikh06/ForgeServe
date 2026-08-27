from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow imports from the repository root when running this file directly.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from forgeserve.engine.config import GenerationConfig
from forgeserve.engine.paged_generation import PagedGenerationEngine
from forgeserve.model.paged_runtime import PagedRuntime
from forgeserve.model.types import AttentionImplementation
from forgeserve.page_attention.block_manager import BlockManager
from forgeserve.sampler.greedy import GreedySampler

from timerrr import (
    get_peak_memory_mb,
    reset_peak_memory,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BLOCK_SIZE = 16
NUM_BLOCKS = 256

WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def create_engine() -> tuple[
    PagedGenerationEngine,
    PagedRuntime,
    BlockManager,
]:
    """
    Create the ForgeServe PagedAttention generation stack.

    Returns:
        engine:
            Paged generation engine.

        runtime:
            Model runtime.

        block_manager:
            KV-cache block manager.
    """

    runtime = PagedRuntime(
        model_name=MODEL_NAME,
        attention=AttentionImplementation.SDPA,
    )

    block_manager = BlockManager.from_model_config(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        model=runtime.model,
        device="cuda",
    )

    runtime.attach_block_manager(block_manager)

    engine = PagedGenerationEngine(
        runtime=runtime,
        sampler=GreedySampler(),
    )

    return engine, runtime, block_manager


def run_generation(
    engine: PagedGenerationEngine,
    prompt: str,
    max_new_tokens: int,
):
    """
    Run one generation request.
    """

    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
    )

    return engine.generate(
        prompt=prompt,
        config=config,
    )


def warmup(
    engine: PagedGenerationEngine,
    prompt: str,
    max_new_tokens: int,
) -> None:
    """
    Execute warmup generations before collecting measurements.

    Warmup is important because the first GPU execution may include
    initialization, memory allocation, and kernel setup overhead.
    """

    print(f"Warming up with {WARMUP_RUNS} runs...")

    for _ in range(WARMUP_RUNS):
        run_generation(
            engine=engine,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("Warmup complete.")


def benchmark_scenario(
    engine: PagedGenerationEngine,
    block_manager: BlockManager,
    prompts: list[str],
    max_new_tokens: int,
) -> None:
    """
    Benchmark one concurrency scenario.

    Each prompt is executed sequentially for now.

    This first benchmark measures:
        - total generation time
        - peak GPU memory
        - generated tokens
        - block recovery

    Later we will add true concurrent scheduling once the
    PagedGenerationEngine supports it.
    """

    print("\n" + "=" * 70)
    print(f"Requests       : {len(prompts)}")
    print(f"Max new tokens : {max_new_tokens}")
    print(f"Block size     : {BLOCK_SIZE}")
    print(f"Total blocks   : {NUM_BLOCKS}")
    print("=" * 70)

    for run in range(1, BENCHMARK_RUNS + 1):

        reset_peak_memory()

        initial_free_blocks = block_manager.num_free_blocks

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_event = None
        end_event = None

        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

        total_generated_tokens = 0

        for prompt in prompts:

            response = run_generation(
                engine=engine,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )

            total_generated_tokens += response.generated_tokens

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()

            total_time_ms = start_event.elapsed_time(end_event)

        else:
            total_time_ms = 0.0

        peak_memory_mb = get_peak_memory_mb()

        final_free_blocks = block_manager.num_free_blocks

        block_leak = (
            initial_free_blocks != final_free_blocks
        )

        throughput = 0.0

        if total_time_ms > 0:
            throughput = (
                total_generated_tokens
                / total_time_ms
            ) * 1000.0

        print(f"\nRun {run}/{BENCHMARK_RUNS}")
        print(f"Generated tokens : {total_generated_tokens}")
        print(f"Total time       : {total_time_ms:.2f} ms")
        print(f"Throughput       : {throughput:.2f} tok/s")
        print(f"Peak memory      : {peak_memory_mb:.2f} MB")
        print(f"Free blocks      : {final_free_blocks}")
        print(f"Block leak       : {block_leak}")

        if block_leak:
            raise RuntimeError(
                "PagedAttention block leak detected: "
                f"expected {initial_free_blocks} free blocks, "
                f"got {final_free_blocks}"
            )


def main() -> None:
    """
    Entry point for the PagedAttention benchmark.
    """

    print("=" * 70)
    print("ForgeServe PagedAttention Benchmark")
    print("=" * 70)

    print(f"Model       : {MODEL_NAME}")
    print(f"Block size  : {BLOCK_SIZE}")
    print(f"Num blocks  : {NUM_BLOCKS}")
    print(f"Warmup runs : {WARMUP_RUNS}")
    print(f"Bench runs  : {BENCHMARK_RUNS}")

    engine, runtime, block_manager = create_engine()

    scenarios = [
        (
            "single_request",
            [
                "Explain how transformers work in detail."
            ],
        ),
        (
            "two_requests",
            [
                "Explain how transformers work in detail.",
                "Explain how KV cache works in LLM inference.",
            ],
        ),
        (
            "four_requests",
            [
                "Explain attention mechanisms.",
                "Explain KV cache.",
                "Explain FlashAttention.",
                "Explain PagedAttention.",
            ],
        ),
        (
            "eight_requests",
            [
                "Explain transformers.",
                "Explain attention.",
                "Explain KV cache.",
                "Explain FlashAttention.",
                "Explain PagedAttention.",
                "Explain GQA.",
                "Explain autoregressive generation.",
                "Explain LLM inference.",
            ],
        ),
    ]

    for scenario_name, prompts in scenarios:

        print("\n")
        print(f">>> Scenario: {scenario_name}")

        warmup(
            engine=engine,
            prompt=prompts[0],
            max_new_tokens=128,
        )

        benchmark_scenario(
            engine=engine,
            block_manager=block_manager,
            prompts=prompts,
            max_new_tokens=128,
        )

    print("\n" + "=" * 70)
    print("PagedAttention benchmark completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()