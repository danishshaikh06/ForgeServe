"""
Metric collection for ForgeServe benchmarks.

Design decision:
    We use torch.cuda.Event for GPU timing, not time.perf_counter.
    Reason: CUDA is asynchronous. CPU timers measure kernel launch,
    not kernel execution. GPU events record timestamps on the GPU itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch


@dataclass
class BenchmarkResult:
    """
    All metrics collected for a single benchmark run.
    Attributes:
        engine_name:        Which engine was used (naive / kv_cache)
        prompt_tokens:      Number of tokens in the input prompt
        generated_tokens:   Number of tokens actually generated
        ttft_ms:            Time to first token in milliseconds
        total_time_ms:      End to end generation time in milliseconds
        tpot_ms:            Average time per output token in milliseconds
        tokens_per_second:  Decode throughput
        peak_memory_mb:     Peak GPU memory during generation in MB
    """
    engine_name: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    total_time_ms: float
    tpot_ms: float
    tokens_per_second: float
    peak_memory_mb: float

    @classmethod
    def compute(
        cls,
        engine_name: str,
        prompt_tokens: int,
        generated_tokens: int,
        ttft_ms: float,
        total_time_ms: float,
        peak_memory_mb: float,
    ) -> BenchmarkResult:
        """
        Compute derived metrics from raw measurements.

        tpot and throughput are derived — never measure them directly.
        Always compute from total_time and generated_tokens.
        """
        decode_time_ms = total_time_ms - ttft_ms
        decode_tokens = max(generated_tokens - 1, 1)  # exclude first token

        tpot_ms = decode_time_ms / decode_tokens
        tokens_per_second = (generated_tokens / total_time_ms) * 1000.0

        return cls(
            engine_name=engine_name,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_ms=ttft_ms,
            total_time_ms=total_time_ms,
            tpot_ms=tpot_ms,
            tokens_per_second=tokens_per_second,
            peak_memory_mb=peak_memory_mb,
        )


@dataclass
class AggregatedResult:
    """
    Statistics across multiple runs of the same scenario.
    Why aggregate?
        A single run is noisy. GPU scheduling, memory allocation,
        OS interrupts all add variance. We run N times and report
        mean and standard deviation. High std_dev means unstable results.
    """
    engine_name: str
    prompt_tokens: int
    generated_tokens: int
    runs: int

    mean_ttft_ms: float
    std_ttft_ms: float

    mean_tpot_ms: float
    std_tpot_ms: float

    mean_total_ms: float
    std_total_ms: float

    mean_tokens_per_second: float
    mean_peak_memory_mb: float

    @classmethod
    def from_results(cls, results: list[BenchmarkResult]) -> AggregatedResult:
        """Aggregate a list of BenchmarkResult into statistics."""
        import statistics

        assert len(results) > 0, "Cannot aggregate empty results"
        assert len({r.engine_name for r in results}) == 1, "Mixed engines in aggregation"

        def mean(values: list[float]) -> float:
            return sum(values) / len(values)

        def std(values: list[float]) -> float:
            return statistics.stdev(values) if len(values) > 1 else 0.0

        ttfts = [r.ttft_ms for r in results]
        tpots = [r.tpot_ms for r in results]
        totals = [r.total_time_ms for r in results]
        tps = [r.tokens_per_second for r in results]
        mems = [r.peak_memory_mb for r in results]

        return cls(
            engine_name=results[0].engine_name,
            prompt_tokens=results[0].prompt_tokens,
            generated_tokens=results[0].generated_tokens,
            runs=len(results),
            mean_ttft_ms=mean(ttfts),
            std_ttft_ms=std(ttfts),
            mean_tpot_ms=mean(tpots),
            std_tpot_ms=std(tpots),
            mean_total_ms=mean(totals),
            std_total_ms=std(totals),
            mean_tokens_per_second=mean(tps),
            mean_peak_memory_mb=mean(mems),
        )