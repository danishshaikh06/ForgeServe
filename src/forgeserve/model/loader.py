import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class ModelLoader:
    """
    Loads pretrained Hugging Face causal language models.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:

        self.model_name = model_name
        self._device = device
        self.dtype = dtype

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Loads the model and tokenizer.
        Returns:
            model: The loaded Hugging Face model.
            tokenizer: The corresponding tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            self.model_name,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=self.dtype, trust_remote_code=True)
        model.to(self.device)  # type: ignore[arg-type]
        model.eval()  # Set the model to evaluation mode

        return model, tokenizer
