""""
LLM Inference Pipeline (Decoder-Only Transformer)

1. Input text
      │
      ▼
2. Tokenizer
      - Converts text → token IDs (input_ids)
      - Creates attention_mask
      │
      ▼
3. Token Embedding
      - Converts token IDs → dense vectors
      - Shape: (B, S, hidden_dim)
      │
      ▼
4. Positional Information
      - Applies positional encoding / rotary embeddings (RoPE)
      │
      ▼
5. Transformer Layers (Repeated N times)
      a. LayerNorm
      b. Multi-Head Self-Attention
      c. Residual Connection
      d. LayerNorm
      e. Feed Forward Network (MLP)
            - Up Projection
            - Activation (e.g. SwiGLU/GELU)
            - Down Projection
      f. Residual Connection
      │
      ▼
6. Final LayerNorm
      │
      ▼
7. LM Head (Output Projection)
      - Hidden states: (B, S, hidden_dim)
      - Multiplied by LM Head weights:
            (hidden_dim × vocab_size)
      - Produces logits:
            (B, S, vocab_size)
      │
      ▼
8. Select Last Position
      - logits[:, -1, :]
      - Scores for every token in the vocabulary
      │
      ▼
9. Sampling Strategy
      - Greedy (argmax)
      - Temperature
      - Top-k
      - Top-p
      │
      ▼
10. Next Token ID
      │
      ▼
11. Tokenizer Decode
      - Token ID → Text
      │
      ▼
12. Append Token to Input
      - Repeat until EOS or max_new_tokens is reached.
"""

from forgeserve.model.loader import ModelLoader
import torch
from forgeserve.logger import get_logger

logger = get_logger(__name__)

class Runtime:
    """
    Runtime class that initializes the model and tokenizer using ModelLoader.
    """
    def __init__(
            self,
            model_name: str,
            ) -> None:
        self.model_name = model_name
        self.loader = ModelLoader(model_name=self.model_name)
        logger.info(f"Loading model and tokenizer for {self.model_name}...")
        self.model, self.tokenizer = self.loader.load()

    def tokenized(self, text: str) -> dict:
        """
        Tokenizes the input text using the loaded tokenizer.
        Args:
            text (str): The input text to tokenize.
        Returns:
            tokenized_output: The tokenized representation of the input text.
        """
        logger.info(f"Tokenizing input text: {text}")
        tokenized_output = self.tokenizer(text, return_tensors="pt").to(self.loader.device)
        logger.info(f"Tokenized output shape: {tokenized_output['input_ids'].shape}")
        return tokenized_output

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """
        Performs a forward pass through the model.
        Args:
            input_ids: The tokenized input IDs.
            attention_mask: The attention mask for the input.
        Returns:
            output: The model's output.
        """
        with torch.no_grad():
            logger.info("Performing forward pass through the model...")
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logger.info("Forward pass completed.")
        return output # the output is a tuple containing the model's output and other information, depending on the model architecture.

if __name__ == "__main__":
    runtime = Runtime(model_name="Qwen/Qwen2.5-0.5B-Instruct")

    sample_text = "Hello, how are you?"

    tokenized_output = runtime.tokenized(sample_text)
    output = runtime.forward(**tokenized_output)
    print(tokenized_output)
    print(output)
     

    logits = output.logits

    next_token = logits[:, -1, :].argmax(dim=-1)
    print(f"Next token ID: {next_token.item()}")

    token_str = runtime.tokenizer.decode(next_token)
    print(f"Next token string: {token_str}")