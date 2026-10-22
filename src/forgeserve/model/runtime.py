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
    """Loads a causal language model and provides cached inference operations."""

    def __init__(
        self,
        model_name: str,
        attention: AttentionImplementation = AttentionImplementation.SDPA,
    ) -> None:
        self.model_name = model_name

        try:
            logger.info(
                "Initializing Runtime: model=%s attention=%s",
                model_name,
                attention.value,
            )

            self.loader = ModelLoader(
                model_name=self.model_name,
                attention=attention,
            )
            self.model, self.tokenizer = self.loader.load()

            self.model.eval()

        except Exception as exc:
            logger.exception(
                "Failed to initialize Runtime for %s",
                self.model_name,
            )
            raise ModelException(
                f"Failed to initialize Runtime for {self.model_name}: {exc}"
            ) from exc

        logger.info(
            "Runtime initialized successfully for %s",
            self.model_name,
        )

    def tokenize(
        self,
        text: str,
        system_prompt: str | None = None,
    ) -> BatchEncoding:
        """Format and tokenize a chat prompt."""

        messages = []

        if system_prompt:
            messages.append(
                {"role": "system", "content": system_prompt}
            )

        messages.append({"role": "user", "content": text})

        logger.debug("Tokenizing input text")

        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        tokenized_output = self.tokenizer(
            formatted_text,
            return_tensors="pt",
        ).to(self.loader.device)

        logger.debug("Tokenization completed successfully")
        return cast(BatchEncoding, tokenized_output)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Perform a model forward pass without tracking gradients."""

        with torch.inference_mode():
            start = time.perf_counter()

            output = cast(
                CausalLMOutputWithPast,
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                ),
            )

            elapsed = time.perf_counter() - start

        logger.debug(
            "Forward pass completed in %.4f seconds",
            elapsed,
        )

        return output

    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Process the entire prompt once.

        Returns:
            logits: (batch, vocab_size), for the next token.
            cache: KV cache populated with the prompt.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape (batch, sequence_length)"
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask and input_ids must have identical shapes"
            )

        logger.debug(
            "Prefill: processing %d prompt tokens",
            input_ids.shape[1],
        )

        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

        if output.logits is None:
            raise ModelException("Model did not return logits during prefill")

        if output.past_key_values is None:
            raise ModelException(
                "Model did not return past_key_values during prefill"
            )

        logits = output.logits[:, -1, :]

        cache = KVCache.from_prefill(
            past_key_values=output.past_key_values,
            prompt_len=input_ids.shape[1],
        )

        logger.debug(
            "Prefill complete: layers=%d seq_len=%d",
            cache.num_layers,
            cache.seq_len,
        )

        return logits, cache

    def decode(
        self,
        token_ids: torch.Tensor | list[int],
    ) -> str:
        """Convert token IDs into text."""

        start = time.perf_counter()

        decoded_text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

        elapsed = time.perf_counter() - start

        logger.debug(
            "Decoded text in %.4f seconds",
            elapsed,
        )

        return decoded_text

    def decode_step(
        self,
        token_id: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: KVCache,
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Process exactly one new token using the existing KV cache.

        token_id:       (batch, 1)
        attention_mask: (batch, cache.seq_len + 1)
        """

        expected_mask_shape = (
            token_id.shape[0],
            cache.seq_len + 1,
        )

        if token_id.ndim != 2 or token_id.shape[1] != 1:
            raise ValueError(
                "token_id must have shape (batch, 1)"
            )

        if attention_mask.shape != expected_mask_shape:
            raise ValueError(
                f"Expected attention_mask shape {expected_mask_shape}, "
                f"got {tuple(attention_mask.shape)}"
            )

        if token_id.device != attention_mask.device:
            raise ValueError(
                "token_id and attention_mask must be on the same device"
            )

        logger.debug(
            "Decode step at position %d",
            cache.seq_len,
        )

        output = self.forward(
            input_ids=token_id,
            attention_mask=attention_mask,
            past_key_values=cache.past_key_values,
            use_cache=True,
        )

        if output.logits is None:
            raise ModelException(
                "Model did not return logits during decode"
            )

        if output.past_key_values is None:
            raise ModelException(
                "Model did not return past_key_values during decode"
            )

        logits = output.logits[:, -1, :]
        updated_cache = cache.update(output.past_key_values)

        return logits, updated_cache

    @staticmethod
    def log_gpu_memory(stage: str) -> None:
        """Log CUDA allocated, reserved, and peak allocated memory."""

        if not torch.cuda.is_available():
            logger.debug(
                "[%s] CUDA unavailable; skipping GPU memory logging",
                stage,
            )
            return

        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        logger.info(
            "[%s] Allocated: %.2f MiB | Reserved: %.2f MiB | "
            "Peak allocated: %.2f MiB",
            stage,
            allocated,
            reserved,
            peak,
        )
