import pytest

from forgeserve.engine.kvcache import GenerationEngine
from forgeserve.engine.response import GenerationResponse
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.greedy import GreedySampler


@pytest.mark.integration
def test_full_generation_pipeline() -> None:
    """
    Tests complete inference pipeline:

    prompt
      |
      Runtime
      |
      GenerationEngine
      |
      Generated response
    """

    runtime = Runtime(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    sampler = GreedySampler()

    engine = GenerationEngine(
        runtime,
        sampler,
    )

    response = engine.generate(
        prompt="Explain what Python is.",
        max_new_token=20,
    )

    assert isinstance(
        response,
        GenerationResponse,
    )

    assert isinstance(
        response.text,
        str,
    )

    assert len(response.text) > 0

    assert response.generated_tokens <= 20

    assert response.finish_reason in {
        "eos",
        "length",
    }
