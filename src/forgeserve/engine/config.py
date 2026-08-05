from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    do_sample: bool = False

