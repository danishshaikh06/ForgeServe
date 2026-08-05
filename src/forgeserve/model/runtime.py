""" "
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

from typing import Any, cast

import torch
from transformers import BatchEncoding
from transformers.modeling_outputs import CausalLMOutputWithPast
import time 

from forgeserve.logger import get_logger
from forgeserve.model.exception import ModelException
from forgeserve.model.loader import ModelLoader

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
        try:
            logger.info(f"Initializing Runtime with model: {self.model_name}")
            self.loader = ModelLoader(model_name=self.model_name)

        except Exception as e:
            logger.exception(f"Failed to initialize Runtime for {self.model_name}: {e}")
            raise ModelException(f"Failed to initialize Runtime for {self.model_name}: {e}") from e

        else:
            logger.info(f"ModelLoader initialized successfully for {self.model_name}.")
            self.model, self.tokenizer = self.loader.load()

    def tokenize(self, text: str, system_prompt: str | None = None) -> BatchEncoding:
        """
        Tokenizes the input text using the loaded tokenizer.
        Args:
            text (str): The input text to tokenize.
        Returns:
            tokenized_output: The tokenized representation of the input text.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        logger.debug("Tokenizing input text")
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenized_output = cast(
            BatchEncoding,
            self.tokenizer(
                formatted_text,
                return_tensors="pt",
            ).to(self.loader.device),
        )
        logger.debug("Tokenization completed successfully")

        return tokenized_output

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: Any
    ) -> CausalLMOutputWithPast:
        """
        Performs a forward pass through the model.
        Args:
            input_ids (torch.Tensor): The tokenized input IDs.
            attention_mask (torch.Tensor): The attention mask for the input.
            **kwargs: Keyword arguments for the model's forward method.
        Returns:
            output: The model's output.
        """
        with torch.inference_mode():
            start = time.perf_counter()

            logger.debug("Performing forward pass through the model...")
            output = cast(
                CausalLMOutputWithPast, self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            )
            end = time.perf_counter()

            time_taken_for = end - start

        logger.debug(f"Forward pass completed. Time taken for forward pass{time_taken_for}")
        return output

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decodes the token IDs back to text using the loaded tokenizer.
        Args:
            token_ids (torch.Tensor): The token IDs to decode.
        Returns:
            decoded_text (str): The decoded text.
        """
        start = time.perf_counter()
        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        end = time.perf_counter()

        time_taken_dec = end - start 

        logger.debug(f"Decoded text successfully.Time taken to decode{time_taken_dec}")
        return decoded_text
