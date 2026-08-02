from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ModelLoader:
    """
    Loads pretrained Hugging Face causal language models.
    """
    def __init__(self,
                 model_name: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 dtype: torch.dtype = torch.bfloat16) -> None:
        
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

    def load(self):
        """
        Loads the model and tokenizer.
        Returns:
            model: The loaded Hugging Face model.
            tokenizer: The corresponding tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            trust_remote_code=True
            )
        model.to(self.device)
        model.eval()  # Set the model to evaluation mode

        return model, tokenizer

    @property
    def device(self):

        return self.model.device