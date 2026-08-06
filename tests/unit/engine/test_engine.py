import pytest

from forgeserve.engine.kvcache import KVCacheGenerationEngine
from forgeserve.engine.response import GenerationResponse
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.greedy import GreedySampler
from forgeserve.engine.config import GenerationConfig


config = GenerationConfig(
    max_new_tokens = 10 
)

@pytest.fixture(scope="session")
def engine() -> KVCacheGenerationEngine:
    runtime = Runtime(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    sampler = GreedySampler()

    return KVCacheGenerationEngine(
        runtime,
        sampler,
    )


def test_generate_returns_response(
    engine: KVCacheGenerationEngine,
) -> None:

    
    response = engine.generate(
        "Hello",
         config,
    )

    assert isinstance(
        response,
        GenerationResponse,
    )


def test_generated_text_is_string(
    engine: KVCacheGenerationEngine,
) -> None:

    response = engine.generate(
        "Hello",
        config
    )

    assert isinstance(
        response.text,
        str,
    )

    assert len(response.text) > 0


def test_generated_token_count(
    engine: KVCacheGenerationEngine,
) -> None:

   

    response = engine.generate(
        "Hello",
        config,
    )

    assert response.generated_tokens <= config.max_new_tokens


def test_generation_finish_reason(
    engine: KVCacheGenerationEngine,
) -> None:

    response = engine.generate(
        "Hello",
        config,
    )

    assert response.finish_reason in {
        "eos",
        "length",
    }