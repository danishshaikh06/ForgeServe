import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from forgeserve.model.loader import ModelLoader


@pytest.fixture(scope="session")
def loader() -> ModelLoader:
    return ModelLoader(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )


@pytest.fixture(scope="session")
def loaded_model(loader: ModelLoader) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    return loader.load()


def test_loader_returns_model_and_tokenizer(
    loaded_model: tuple[PreTrainedModel, PreTrainedTokenizerBase],
) -> None:

    model, tokenizer = loaded_model

    assert model is not None
    assert tokenizer is not None


def test_model_is_in_eval_mode(
    loaded_model: tuple[PreTrainedModel, PreTrainedTokenizerBase],
) -> None:

    model, _ = loaded_model

    assert model.training is False


def test_model_on_correct_device(
    loader: ModelLoader,
    loaded_model: tuple[PreTrainedModel, PreTrainedTokenizerBase],
) -> None:

    model, _ = loaded_model

    model_device = next(model.parameters()).device

    assert model_device.type == loader.device


def test_tokenizer_can_encode(
    loaded_model: tuple[PreTrainedModel, PreTrainedTokenizerBase],
) -> None:

    _, tokenizer = loaded_model

    output = tokenizer(
        "Hello world",
        return_tensors="pt",
    )

    assert "input_ids" in output
    assert "attention_mask" in output


def test_model_can_run_forward_pass(
    loaded_model: tuple[PreTrainedModel, PreTrainedTokenizerBase],
) -> None:

    model, tokenizer = loaded_model

    tokens = tokenizer(
        "Hello world",
        return_tensors="pt",
    )

    tokens = {
        key: value.to(model.device)
        for key, value in tokens.items()
    }

    with torch.inference_mode():
        output = model(**tokens)

    assert output.logits is not None
