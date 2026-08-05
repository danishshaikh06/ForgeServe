import json
import time
from pathlib import Path

from forgeserve.engine.config import GenerationConfig
from forgeserve.engine.generation import GenerationEngine
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.greedy import GreedySampler


def main() -> None:

    runtime = Runtime(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    sampler = GreedySampler()

    engine = GenerationEngine(
        runtime,
        sampler,
    )

    prompt = "Hello! how are you?"

    config = GenerationConfig(
        max_new_tokens=100,
    )

    start = time.perf_counter()

    response = engine.generate(
        prompt,
        config,
    )

    end = time.perf_counter()

    latency = end - start

    prompt_tokens = len(
        runtime.tokenizer.encode(prompt)
    )

    generated_tokens = response.generated_tokens

    tokens_per_second = (
        generated_tokens / latency
        if latency > 0
        else 0
    )

    benchmark_result = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_new_tokens": config.max_new_tokens,
        "latency_seconds": round(latency, 3),
        "tokens_per_second": round(tokens_per_second, 2),
        "finish_reason": response.finish_reason,
        "generated_text": response.text,
    }

    artifact_path = Path("artifacts/phase1")

    artifact_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = artifact_path / "benchmark.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            benchmark_result,
            file,
            indent=4,
        )

    print(f"Benchmark saved to {output_file}")

    print("\nBenchmark Results")
    print("-" * 40)

    for key, value in benchmark_result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()