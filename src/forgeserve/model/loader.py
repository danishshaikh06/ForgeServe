import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from forgeserve.model.types import AttentionImplementation
from forgeserve.logger import get_logger
from forgeserve.model.exception import ModelException

logger = get_logger(__name__)


class ModelLoader:
    """
    Loads pretrained Hugging Face causal language models.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        attention: AttentionImplementation = AttentionImplementation.SDPA
    ) -> None:

        self.model_name = model_name
        self._device = device
        self.dtype = dtype
        self.attention = attention

        logger.info(
            "ModelLoader configured: model=%s device=%s dtype=%s attention=%s",
            model_name, device, dtype, attention.value,
        )

    @property
    def device(self) -> str:
        return self._device

    def verify_attention_backend(self, model: PreTrainedModel) -> None:
        """
        Verify which attention backend is active after loading.
        Logs a warning if FlashAttention conditions are not met.
        """
        if self.attention == AttentionImplementation.EAGER:
            logger.info("Attention backend: EAGER (baseline)")
            return

        if not torch.cuda.is_available():
            logger.warning(
                "SDPA requested but no CUDA GPU found. "
                "Running standard attention on CPU."
            )
            return

        # Check dtype
        first_param = next(model.parameters())
        if first_param.dtype not in (torch.float16, torch.bfloat16):
            logger.warning(
                "Model dtype is %s. FlashAttention requires fp16 or bf16. "
                "Will fall back to standard attention.",
                first_param.dtype,
            )
            return

        logger.info(
            "Attention backend: SDPA (FlashAttention eligible). "
            "dtype=%s device=%s",
            first_param.dtype, self._device,
        )

    def load(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Loads the model and tokenizer.
        Returns:
            model: The loaded Hugging Face model.
            tokenizer: The corresponding tokenizer.
        """
        try:
            logger.info("Loading tokenizer for %s", self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                self.model_name,
                trust_remote_code=True,
            )

            logger.info(
                "Loading model with torch_dtype=%s attn_implementation=%s",
                self.dtype, self.attention.value,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                dtype=self.dtype, 
                trust_remote_code=True,
                attn_implementation= self.attention.value # Flash Attention -> automatically uses FlashAttention when conditions are met.
                )
            model.to(self.device)  # type: ignore[arg-type]
            model.eval()  # Set the model to evaluation mode

        except Exception as e:
            logger.exception("Failed to load model %s", self.model_name)
            raise ModelException(f"Failed to load model {self.model_name}: {e}") from e
        
        self.verify_attention_backend(model)

        return model, tokenizer
    
        
