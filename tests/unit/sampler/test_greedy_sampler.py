import pytest
import torch

from forgeserve.sampler.exception import InvalidLogitsShapeError
from forgeserve.sampler.greedy import GreedySampler


@pytest.fixture
def sampler() -> GreedySampler:
    return GreedySampler()


def test_greedy_sampler_normal_logits(
    sampler: GreedySampler,
) -> None:
    logits = torch.tensor([
        [1.0, 2.0, 5.0],
    ])

    token = sampler.sample(logits)

    expected = torch.tensor([2])

    assert torch.equal(token, expected)


def test_greedy_sampler_batch_logits(
    sampler: GreedySampler,
) -> None:
    logits = torch.tensor([
        [1.0, 5.0, 2.0],
        [9.0, 1.0, 0.0],
    ])

    token = sampler.sample(logits)

    expected = torch.tensor([1, 0])

    assert torch.equal(token, expected)


def test_greedy_sampler_invalid_dimension(
    sampler: GreedySampler,
) -> None:
    logits = torch.randn(10)

    with pytest.raises(InvalidLogitsShapeError):
        sampler.sample(logits)


def test_greedy_sampler_three_dimensional_logits(
    sampler: GreedySampler,
) -> None:
    logits = torch.randn(1, 5, 100)

    with pytest.raises(InvalidLogitsShapeError):
        sampler.sample(logits)
