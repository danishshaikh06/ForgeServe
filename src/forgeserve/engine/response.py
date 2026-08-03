from dataclasses import dataclass


@dataclass(slots=True)
class GenerationResponse:
    """
    Response returned by the generation engine.
    """

    text: str
    generated_tokens: int
    finish_reason: str
