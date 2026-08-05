import pytest
from transformers import BatchEncoding

from forgeserve.model.runtime import Runtime


@pytest.fixture(scope="session")
def runtime() -> Runtime:
    return Runtime(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )


def test_tokenize_returns_batch_encoding(
    runtime: Runtime,
) -> None:

    output = runtime.tokenize(
        "Hello world"
    )

    assert isinstance(output, BatchEncoding)

    assert "input_ids" in output
    assert "attention_mask" in output


def test_tokenize_tensors_on_correct_device(
    runtime: Runtime,
) -> None:

    output = runtime.tokenize(
        "Hello world"
    )

    assert output["input_ids"].device.type == runtime.loader.device
    assert output["attention_mask"].device.type == runtime.loader.device


def test_forward_returns_logits(
    runtime: Runtime,
) -> None:

    tokens = runtime.tokenize(
        "Hello world"
    )

    output = runtime.forward(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
    )

    assert output.logits is not None


def test_forward_correct_shape(
    runtime: Runtime,
) -> None:

    tokens = runtime.tokenize(
        "Hello world"
    )

    output = runtime.forward(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
    )

    assert output.logits is not None

    batch_size = output.logits.shape[0]
    sequence_length = output.logits.shape[1]

    assert batch_size == tokens["input_ids"].shape[0]
    assert sequence_length == tokens["input_ids"].shape[1]


def test_decode_returns_string(
    runtime: Runtime,
) -> None:

    tokens = runtime.tokenize(
        "Hello world"
    )

    token_ids = tokens["input_ids"][0]

    text = runtime.decode(
        token_ids
    )

    assert isinstance(text, str)

    assert len(text) > 0
