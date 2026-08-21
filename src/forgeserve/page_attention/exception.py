class KVCacheOutOfMemoryError(Exception):
    """
    Raised when block pool is exhausted.
    Request should be placed in waiting queue, not retried immediately.
    """

class KVCacheBlockNotOwnedError(Exception):
    """Raised when freeing blocks for an unknown request."""
