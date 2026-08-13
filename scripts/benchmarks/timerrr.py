"""
GPU-accurate timing utilities.
Key insight:
    Always use CUDA events for GPU timing.
    Always synchronize before reading results.
    Never mix CPU and GPU timers for the same measurement.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass

import torch

@dataclass
class TimingResult:
    elapsed_ms: float

@contextmanager
def cuda_timer():
    """
    Context manager for accurate GPU timing.
    Usage:
        with cuda_timer() as t:
            run_gpu_operation()
        print(t.elapsed_ms)
    Design:
        We yield a mutable container so the caller can read
        elapsed_ms after the context exits.
    """
    result = TimingResult(elapsed_ms=0.0)

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield result
        end.record()
        torch.cuda.synchronize()
        result.elapsed_ms = start.elapsed_time(end)
    else:
        # CPU fallback for development without GPU
        import time
        t0 = time.perf_counter()
        yield result
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0


def reset_peak_memory() -> None:
    """Reset GPU peak memory tracker before each run."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb() -> float:
    """Read peak GPU memory allocated during this run."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0