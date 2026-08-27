"""
Core benchmark runner for ForgeServe.

Design philosophy:
    The runner calls Runtime methods directly instead of engine.generate().
    This gives precise control over timing boundaries and memory measurements.

Why direct Runtime calls?
    Engine.generate() is intentionally a user-facing abstraction.
    A benchmark needs a white-box view so that we can measure:

        - TTFT
        - decode time
        - total latency
        - peak GPU memory

Important benchmark behavior:
    The naive implementation explicitly disables KV cache.

    The KV-cache implementation explicitly uses prefill + decode_step.

    EOS stopping can be enabled or disabled.

    For controlled sequence-length experiments, use:
        stop_on_eos=False

    This ensures max_new_tokens represents the requested number of
    generation steps rather than allowing an early EOS to shorten
    the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from forgeserve.engine.config import GenerationConfig
from forgeserve.logger import get_logger
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.base import Sampler

from metrics import AggregatedResult, BenchmarkResult
from timerrr import (
    cuda_timer,
    get_peak_memory_mb,
    reset_peak_memory,
)

logger = get_logger(__name__)

@dataclass
class BenchmarkRun:
    """
    Result of one benchmark execution.

    Attributes:
        result:
            BenchmarkResult when the run succeeded.

        oom:
            True when the run failed because of CUDA out-of-memory.

        error_message:
            Original CUDA OOM message when available.
    """

    result: BenchmarkResult | None
    oom: bool = False
    error_message: str | None = None

# CUDA cleanup helpers
def _cleanup_cuda() -> None:
    """
    Release unused CUDA allocator memory before starting a new measurement.

    Important:
        torch.cuda.empty_cache() only releases memory that is no longer
        referenced by live tensors. Therefore callers must delete temporary
        tensors before invoking this function.
    """

    if not torch.cuda.is_available():
        return

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

# Naive generation
def run_naive_once(
    runtime: Runtime,
    sampler: Sampler,
    prompt: str,
    config: GenerationConfig,
    stop_on_eos: bool = True,
) -> BenchmarkRun:
    """
    Benchmark one naive generation run.

    Naive generation intentionally disables the model KV cache.

    TTFT:
        Initial forward pass over the complete prompt.

    Total generation:
        Initial prompt forward pass + one complete-sequence forward
        pass for every subsequent generation step.

    Args:
        runtime:
            ForgeServe model runtime.

        sampler:
            Sampling strategy.

        prompt:
            Input prompt.

        config:
            Generation configuration.

        stop_on_eos:
            If True, stop when EOS is generated.
            If False, continue until max_new_tokens is reached.
    """

    encoded = runtime.tokenize(
        prompt,
        config.system_prompt,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    prompt_tokens = input_ids.shape[1]

    # TTFT measurement
    try:

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        reset_peak_memory()

        with cuda_timer() as ttft_timer:

            output = runtime.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            logits = output.logits[:, -1, :]

            first_token = sampler.sample(logits)

        ttft_ms = ttft_timer.elapsed_ms

        # IMPORTANT:
        # The TTFT tensors remain referenced by Python variables.
        # empty_cache() cannot free memory that is still referenced.
        # Release them before the full-generation measurement.
        del output
        del logits
        del first_token

        _cleanup_cuda()
        reset_peak_memory()

        # Full generation measurement

        # Re-tokenize from scratch so the full measurement starts from
        # exactly the same prompt state.
        encoded = runtime.tokenize(
            prompt,
            config.system_prompt,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        generated = 0
        is_eos = False

        with cuda_timer() as full_timer:
            # First forward pass
            output = runtime.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            logits = output.logits[:, -1, :]

            next_token = sampler.sample(logits)

            input_ids, attention_mask = _append_token(
                input_ids,
                attention_mask,
                next_token,
            )

            generated = 1

            if stop_on_eos:
                is_eos = _is_eos(
                    next_token,
                    runtime,
                )

            # Remaining decode steps
            if not is_eos:

                for _ in range(config.max_new_tokens - 1):

                    output = runtime.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )

                    logits = output.logits[:, -1, :]

                    next_token = sampler.sample(logits)

                    input_ids, attention_mask = _append_token(
                        input_ids,
                        attention_mask,
                        next_token,
                    )

                    generated += 1

                    if stop_on_eos and _is_eos(
                        next_token,
                        runtime,
                    ):
                        is_eos = True

                        logger.debug(
                            "Naive generation reached EOS at token %d",
                            generated,
                        )

                        break

        # Release tensors generated by the final forward pass.
        del output
        del logits
        del next_token

        peak_memory_mb = get_peak_memory_mb()

        result = BenchmarkResult.compute(
            engine_name="naive",
            prompt_tokens=prompt_tokens,
            generated_tokens=generated,
            ttft_ms=ttft_ms,
            total_time_ms=full_timer.elapsed_ms,
            peak_memory_mb=peak_memory_mb,
        )

        return BenchmarkRun(result=result)

    except torch.cuda.OutOfMemoryError as exc:

        logger.warning(
            "Naive generation hit CUDA OOM. "
            "prompt_tokens=%d max_new_tokens=%d",
            prompt_tokens,
            config.max_new_tokens,
        )

        _cleanup_cuda()

        return BenchmarkRun(
            result=None,
            oom=True,
            error_message=str(exc),
        )

# KV-cache generation
def run_kvcache_once(
    runtime: Runtime,
    sampler: Sampler,
    prompt: str,
    config: GenerationConfig,
    stop_on_eos: bool = True,
) -> BenchmarkRun:
    """
    Benchmark one KV-cache generation run.

    TTFT:
        Time spent during prefill.

    Decode:
        Each subsequent step passes only the new token together with
        the existing KV cache.

    Args:
        runtime:
            ForgeServe runtime.

        sampler:
            Sampling strategy.

        prompt:
            Input prompt.

        config:
            Generation configuration.

        stop_on_eos:
            If True, stop on EOS.
            If False, continue until max_new_tokens is reached.
    """

    encoded = runtime.tokenize(
        prompt,
        config.system_prompt,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    prompt_tokens = input_ids.shape[1]

    try:
        # TTFT / Prefill measurement
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        reset_peak_memory()

        with cuda_timer() as ttft_timer:

            logits, cache = runtime.prefill(
                input_ids,
                attention_mask,
            )

        ttft_ms = ttft_timer.elapsed_ms

        # We only needed TTFT from this run.
        # Release the prefill result before starting the full measurement.
        del logits
        del cache

        _cleanup_cuda()
        reset_peak_memory()

        # Full generation measurement
        encoded = runtime.tokenize(
            prompt,
            config.system_prompt,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        generated = 0
        is_eos = False

        with cuda_timer() as full_timer:
            # Prefill
            logits, cache = runtime.prefill(
                input_ids,
                attention_mask,
            )

            next_token = sampler.sample(logits)

            input_ids, attention_mask = _append_token(
                input_ids,
                attention_mask,
                next_token,
            )

            generated = 1

            if stop_on_eos:
                is_eos = _is_eos(
                    next_token,
                    runtime,
                )

            # Decode steps
            if not is_eos:

                for _ in range(config.max_new_tokens - 1):

                    logits, cache = runtime.decode_step(
                        token_id=next_token.unsqueeze(-1),
                        attention_mask=attention_mask,
                        cache=cache,
                    )

                    next_token = sampler.sample(logits)

                    input_ids, attention_mask = _append_token(
                        input_ids,
                        attention_mask,
                        next_token,
                    )

                    generated += 1

                    if stop_on_eos and _is_eos(
                        next_token,
                        runtime,
                    ):
                        is_eos = True

                        logger.debug(
                            "KV-cache generation reached EOS at token %d",
                            generated,
                        )

                        break

        # Release final GPU tensors after timing has completed.
        del logits
        del cache
        del next_token

        peak_memory_mb = get_peak_memory_mb()

        result = BenchmarkResult.compute(
            engine_name="kv_cache",
            prompt_tokens=prompt_tokens,
            generated_tokens=generated,
            ttft_ms=ttft_ms,
            total_time_ms=full_timer.elapsed_ms,
            peak_memory_mb=peak_memory_mb,
        )

        return BenchmarkRun(result=result)

    except torch.cuda.OutOfMemoryError as exc:

        logger.warning(
            "KV-cache generation hit CUDA OOM. "
            "prompt_tokens=%d max_new_tokens=%d",
            prompt_tokens,
            config.max_new_tokens,
        )

        _cleanup_cuda()

        return BenchmarkRun(
            result=None,
            oom=True,
            error_message=str(exc),
        )

# Naive vs KV-cache scenario
def run_scenario(
    runtime: Runtime,
    sampler: Sampler,
    prompt: str,
    config: GenerationConfig,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
    stop_on_eos: bool = True,
) -> tuple[AggregatedResult | None, AggregatedResult | None]:
    """
    Run a complete naive-vs-KV-cache benchmark scenario.

    Warmup runs are discarded.

    Benchmark runs are aggregated into mean/std-dev statistics.

    Args:
        runtime:
            ForgeServe runtime.

        sampler:
            Sampling strategy.

        prompt:
            Input prompt.

        config:
            Generation configuration.

        warmup_runs:
            Number of warmup runs.

        benchmark_runs:
            Number of measured runs.

        stop_on_eos:
            Whether EOS should terminate generation.

    Returns:
        Tuple:
            (naive_result, kvcache_result)

        Either side can be None if no successful benchmark run was
        collected for that implementation.
    """

    logger.info(
        "Warming up with %d runs before benchmarking",
        warmup_runs,
    )

    # Warmup
    for i in range(warmup_runs):

        logger.debug(
            "Warmup run %d/%d",
            i + 1,
            warmup_runs,
        )

        naive = run_naive_once(
            runtime=runtime,
            sampler=sampler,
            prompt=prompt,
            config=config,
            stop_on_eos=stop_on_eos,
        )

        kvcache = run_kvcache_once(
            runtime=runtime,
            sampler=sampler,
            prompt=prompt,
            config=config,
            stop_on_eos=stop_on_eos,
        )

        if naive.oom or kvcache.oom:
            logger.warning(
                "CUDA OOM encountered during warmup. "
                "Stopping additional warmup runs."
            )
            break

    logger.info(
        "Warmup complete. Starting %d benchmark runs.",
        benchmark_runs,
    )

    # Benchmark runs
    naive_results: list[BenchmarkResult] = []
    kvcache_results: list[BenchmarkResult] = []

    for i in range(benchmark_runs):

        logger.debug(
            "Benchmark run %d/%d",
            i + 1,
            benchmark_runs,
        )

        naive = run_naive_once(
            runtime=runtime,
            sampler=sampler,
            prompt=prompt,
            config=config,
            stop_on_eos=stop_on_eos,
        )

        if naive.result is not None:
            naive_results.append(naive.result)

        kvcache = run_kvcache_once(
            runtime=runtime,
            sampler=sampler,
            prompt=prompt,
            config=config,
            stop_on_eos=stop_on_eos,
        )

        if kvcache.result is not None:
            kvcache_results.append(kvcache.result)

    # Aggregate results
    naive_aggregated = (
        AggregatedResult.from_results(naive_results)
        if naive_results
        else None
    )

    kvcache_aggregated = (
        AggregatedResult.from_results(kvcache_results)
        if kvcache_results
        else None
    )

    return (
        naive_aggregated,
        kvcache_aggregated,
    )

# KV-cache-only scenario
def run_kvcache_scenario(
    runtime: Runtime,
    sampler: Sampler,
    prompt: str,
    config: GenerationConfig,
    engine_name: str,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
    stop_on_eos: bool = True,
) -> AggregatedResult | None:
    """
    Benchmark a single KV-cache runtime.

    This is useful when comparing:

        EAGER
        SDPA
        FlashAttention

    while keeping KV-cache generation constant.

    Args:
        runtime:
            ForgeServe runtime.

        sampler:
            Sampling strategy.

        prompt:
            Input prompt.

        config:
            Generation configuration.

        engine_name:
            Label used in benchmark output.

        warmup_runs:
            Number of warmup runs.

        benchmark_runs:
            Number of benchmark runs.

        stop_on_eos:
            Whether EOS should terminate generation.
    """

    logger.info(
        "Warming up %s with %d runs",
        engine_name,
        warmup_runs,
    )

    for i in range(warmup_runs):

        logger.debug(
            "Warmup %d/%d",
            i + 1,
            warmup_runs,
        )

        run = run_kvcache_once(
            runtime=runtime,
            sampler=sampler,
            prompt=prompt,
            config=config,
            stop_on_eos=stop_on_eos,
        )

        if run.oom:
            logger.warning(
                "%s hit CUDA OOM during warmup.",
                engine_name,
            )
            return None

    logger.info(
        "Benchmarking %s with %d runs",
        engine_name,
        benchmark_runs,
    )

    results: list[BenchmarkResult] = []

    # Benchmark
    for i in range(benchmark_runs):

        logger.debug(
            "Run %d/%d",
            i + 1,
            benchmark_runs,
        )

        run = run_kvcache_once(
            runtime=runtime,
            sampler=sampler,
            prompt=prompt,
            config=config,
            stop_on_eos=stop_on_eos,
        )

        if run.oom or run.result is None:
            logger.warning(
                "%s benchmark run %d failed.",
                engine_name,
                i + 1,
            )
            continue

        result = BenchmarkResult.compute(
            engine_name=engine_name,
            prompt_tokens=run.result.prompt_tokens,
            generated_tokens=run.result.generated_tokens,
            ttft_ms=run.result.ttft_ms,
            total_time_ms=run.result.total_time_ms,
            peak_memory_mb=run.result.peak_memory_mb,
        )

        results.append(result)

    if not results:
        return None

    return AggregatedResult.from_results(results)


def _append_token(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    next_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Append one generated token to the sequence.

    Args:
        input_ids:
            Current token IDs with shape (batch_size, sequence_length).

        attention_mask:
            Current attention mask with shape
            (batch_size, sequence_length).

        next_token:
            Next token IDs with shape (batch_size,).

    Returns:
        Updated input_ids and attention_mask.
    """

    next_token = next_token.unsqueeze(-1)

    input_ids = torch.cat(
        (
            input_ids,
            next_token,
        ),
        dim=-1,
    )

    attention_mask = torch.cat(
        (
            attention_mask,
            torch.ones(
                (attention_mask.size(0), 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ),
        dim=-1,
    )

    return input_ids, attention_mask


def _is_eos(
    token: torch.Tensor,
    runtime: Runtime,
) -> bool:
    """
    Return True when all generated tokens are EOS.
    """

    return bool(
        (token == runtime.tokenizer.eos_token_id).all()
    )