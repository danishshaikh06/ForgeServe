import pytest

from forgeserve.engine.generation import GenerationEngine
from forgeserve.engine.response import GenerationResponse
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.greedy import GreedySampler


@pytest.fixture(scope="session")
def engine() -> GenerationEngine:
    runtime = Runtime(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    sampler = GreedySampler()

    return GenerationEngine(
        runtime,
        sampler,
    )


def test_generate_returns_response(
    engine: GenerationEngine,
) -> None:

    response = engine.generate(
        "Hello",
        max_new_token=10,
    )

    assert isinstance(
        response,
        GenerationResponse,
    )


def test_generated_text_is_string(
    engine: GenerationEngine,
) -> None:

    response = engine.generate(
        "Hello",
        max_new_token=10,
    )

    assert isinstance(
        response.text,
        str,
    )

    assert len(response.text) > 0


def test_generated_token_count(
    engine: GenerationEngine,
) -> None:

    max_tokens = 5

    response = engine.generate(
        "Hello",
        max_new_token=max_tokens,
    )

    assert response.generated_tokens <= max_tokens


def test_generation_finish_reason(
    engine: GenerationEngine,
) -> None:

    response = engine.generate(
        "Hello",
        max_new_token=5,
    )

    assert response.finish_reason in {
        "eos",
        "length",
    }