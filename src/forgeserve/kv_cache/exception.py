class  KVCacheException(Exception):
    """Base exception for KV Cache errors."""

class KVCacheNotInitializedError(Exception):
    """Raised when decode is attempted before prefill."""

class KVCacheShapeMismatchError(Exception):
    """Raised when cache shape is inconsistent with model config."""
