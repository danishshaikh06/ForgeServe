from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PagedScenario:
    """
    Defines one PagedAttention benchmark workload.

    Attributes:
        name:
            Human-readable scenario name.

        prompts:
            Prompts submitted to the engine.

        max_new_tokens:
            Maximum number of tokens generated per request.
    """

    name: str
    prompts: list[str]
    max_new_tokens: int


def build_scenarios() -> list[PagedScenario]:
    """
    Build the workloads used for the PagedAttention benchmark.

    We intentionally use different request counts so that we can
    observe how the block manager behaves under increasing load.
    """

    base_prompt = (
        "Explain how transformer models work, including attention, "
        "training, inference, and token generation."
    )

    return [
        PagedScenario(
            name="single_request",
            prompts=[base_prompt],
            max_new_tokens=128,
        ),
        PagedScenario(
            name="two_requests",
            prompts=[base_prompt] * 2,
            max_new_tokens=128,
        ),
        PagedScenario(
            name="four_requests",
            prompts=[base_prompt] * 4,
            max_new_tokens=128,
        ),
        PagedScenario(
            name="eight_requests",
            prompts=[base_prompt] * 8,
            max_new_tokens=128,
        ),
    ]