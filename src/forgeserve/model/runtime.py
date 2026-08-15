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

import time
from typing import Any, cast

import torch
from transformers import BatchEncoding
from transformers.modeling_outputs import CausalLMOutputWithPast

from forgeserve.kv_cache.cache import KVCache
from forgeserve.logger import get_logger
from forgeserve.model.exception import ModelException
from forgeserve.model.loader import ModelLoader
from forgeserve.model.types import AttentionImplementation

logger = get_logger(__name__)


class Runtime:
    """
    Runtime class that initializes the model and tokenizer using ModelLoader.
    """

    def __init__(
        self,
        model_name: str,
        attention: AttentionImplementation = AttentionImplementation.SDPA,
    ) -> None:

        self.model_name = model_name
        try:
            logger.info(
                "Initializing Runtime: model=%s attention=%s",
                model_name, attention.value,
            )
            self.loader = ModelLoader(model_name=self.model_name, attention=attention)

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

    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Prefill phase: process the full prompt once.

        Passes use_cache=True so the model returns past_key_values.
        Returns the logits for the last token and the populated KV cache.

        Args:
            input_ids: Full prompt token ids. Shape: (batch, prompt_len)
            attention_mask: Attention mask. Shape: (batch, prompt_len)

        Returns:
            logits: Shape (batch, vocab_size) — logits for next token
            cache: KVCache populated with K/V for all prompt tokens
        """
        logger.debug("Prefill: processing %d prompt token", input_ids.shape[1])

        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache = True,
        )

        assert output.logits is not None
        assert output.past_key_values is not None

        logits = output.logits[:, -1, :] # last token logits

        cache = KVCache.from_prefill(
            past_key_values=output.past_key_values,
            prompt_len= input_ids.shape[1]
        )

        logger.debug(
            "Prefill complete. Cache has %d layers, seq_len=%d",
            cache.num_layers,
            cache.seq_len,
        )

        return logits, cache

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

    def decode_step(
        self,
        token_id: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: KVCache
        ) -> tuple[torch.Tensor, KVCache]:
        """
        Decode phase: process one new token using the KV cache.
        Key insight: input_ids contains ONLY the new token (shape: batch, 1).
        The model attends to previous tokens via past_key_values, not input_ids.
        The attention_mask must cover the FULL sequence (prompt + all decoded tokens).
        Args:
            token_id: The last generated token. Shape: (batch, 1)
            attention_mask: Full sequence mask. Shape: (batch, cache.seq_len + 1)
            cache: Current KV cache
        Returns:
            logits: Shape (batch, vocab_size)
            updated_cache: KVCache with new token's K/V appended
        """
        logger.debug(
            "Decode step at position %d",
            cache.seq_len,
        )

        output = self.forward(
            input_ids = token_id, # only one new token
            attention_mask = attention_mask, # full sequence mask
            past_key_values = cache.past_key_values,
            use_cache = True,
        )

        assert output.logits is not None
        assert output.past_key_values is not None

        logits = output.logits[:, -1, :]
        updated_cache = cache.update(output.past_key_values)

        return logits, updated_cache



