from enum import StrEnum


class AttentionImplementation(StrEnum):
    """
    Attention backend selection for HuggingFace models.
    EAGER: Standard PyTorch attention. Materializes full score matrix to HBM.
           Used as Phase 1/2 baseline. Always available.

    SDPA:  torch.nn.functional.scaled_dot_product_attention.
           Automatically selects FlashAttention kernel when conditions are met:
           - fp16 or bf16 dtype
           - CUDA GPU (Ampere or newer for full benefit)
           - Contiguous Q/K/V tensors
           - Supported head dimensions (64, 128 most common)
           Falls back to efficient_attention or math if conditions not met.

    FLASH: Explicit flash_attn library. Requires separate installation.
           More aggressive optimization than SDPA but less portable.
           Reserved for future phases.
    """
    EAGER = "eager"
    SDPA  = "sdpa"
    FLASH = "flash_attention_2"
