class SamplerError(Exception):
    """Base sampler exception."""

class InvalidLogitsShapeError(SamplerError):
    """Raised when logits have an invalid shape."""