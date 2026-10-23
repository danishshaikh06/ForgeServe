
import torch

from forgeserve.model.paged_runtime import PagedRuntime
from forgeserve.model.types import AttentionImplementation

from forgeserve.page_attention.block_manager import BlockManager

from forgeserve.sampler.greedy import GreedySampler
from forgeserve.executor.executor import Executor

from forgeserve.scheduler.continuous_batching import (
    ContinuousBatchScheduler,
)

from forgeserve.engine.continuous_batching import (
    ContinuousBatchingEngine,
)


def main() -> None:

    # 1. Initialize model runtime
    runtime = PagedRuntime(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        attention=AttentionImplementation.SDPA,
    )

    # 2. Initialize KV block pool
    block_manager = BlockManager.from_model_config(
        num_blocks=128,
        block_size=16,
        model=runtime.model,
        device=runtime.loader.device,
    )

    # 3. Attach block manager to runtime
    runtime.attach_block_manager(block_manager)

    # 4. Initialize sampler
    sampler = GreedySampler()

    # 5. Initialize executor
    executor = Executor(
        runtime=runtime,
        sampler=sampler,
    )

    # 6. Initialize scheduler
    scheduler = ContinuousBatchScheduler(
        runtime=runtime,
        executor=executor,
        max_batch_size=3,
    )

    # 7. Initialize engine
    engine = ContinuousBatchingEngine(
        scheduler=scheduler,
    )

    # 8. Submit multiple requests
    requests = [
        engine.submit(
            prompt="Explain grouped query attention.",
            max_new_tokens=40,
        ),
        engine.submit(
            prompt="What is a KV cache in transformers?",
            max_new_tokens=40,
        ),
        engine.submit(
            prompt="Explain why FlashAttention is memory efficient.",
            max_new_tokens=40,
        ),
    ]

    # 9. Run continuous batching
    results = engine.run_until_idle()

    print("Engine class:", type(engine))
    print("Engine module:", type(engine).__module__)
    print("run_until_idle method:", engine.run_until_idle)
    print("Results:", repr(results))
    print("Results type:", type(results))

    for request in requests:

        response = results[request.request_id]

        print("\n" + "=" * 60)
        print(f"Request ID: {request.request_id}")
        print(f"Prompt: {request.prompt}")
        print(f"Response: {response.text}")
        print(f"Generated Tokens: {response.generated_tokens}")
        print(f"Finish Reason: {response.finish_reason}")

        print(f"TTFT: {request.time_to_first_token_ms}")
        print(f"Total Latency: {request.total_latency_ms}")

    print("\nKV Cache:")
    print(
        f"Free blocks: "
        f"{block_manager.num_free_blocks}/"
        f"{block_manager.num_blocks}"
    )


if __name__ == "__main__":
    main()
