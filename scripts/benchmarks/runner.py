"""
Core benchmark runner.
Design philosophy:
    The runner calls runtime methods directly, not engine.generate().
    This gives surgical control over timing boundaries.
    We can measure TTFT precisely because we control when prefill ends.
    Engine.generate() is a black box for the user.
    The runner is a white box for measurement.
"""
from __future__ import annotations

import torch

from forgeserve.logger import get_logger
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.base import Sampler
from forgeserve.engine.config import GenerationConfig

from metrics import BenchmarkResult, AggregatedResult
from timerrr import cuda_timer, reset_peak_memory, get_peak_memory_mb

logger = get_logger(__name__)


def run_naive_once(
    runtime: Runtime,
    sampler: Sampler,
    prompt: str,
    config: GenerationConfig,
) -> BenchmarkResult:
    """
    Benchmark one run of the naive engine (no KV cache).
    TTFT approximation for naive engine:
        Naive engine has no prefill/decode split.
        TTFT = time for first full forward pass over prompt.
        We measure this by running one forward pass separately.
    """
    encoded = runtime.tokenize(prompt, config.system_prompt)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_tokens = input_ids.shape[1]

    torch.cuda.synchronize()
    torch.cuda.empty_cache() 
    reset_peak_memory()

    # Measure TTFT (first forward pass over full prompt)
    with cuda_timer() as ttft_timer:
        output = runtime.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, -1, :]
        first_token = sampler.sample(logits)

    ttft_ms = ttft_timer.elapsed_ms

    # Decode loop 
    input_ids, attention_mask = _append_token(input_ids, attention_mask, first_token)
    generated = 1

    with cuda_timer() as total_timer:
        # Re-run first forward (total timer includes everything)
        pass

    # Restart full measurement cleanly
    torch.cuda.synchronize()
    torch.cuda.empty_cache() 
    reset_peak_memory()
    
    encoded = runtime.tokenize(prompt, config.system_prompt)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with cuda_timer() as full_timer:
        output = runtime.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, -1, :]
        next_token = sampler.sample(logits)
        input_ids, attention_mask = _append_token(input_ids, attention_mask, next_token)
        generated = 1

        for _ in range(config.max_new_tokens - 1):
            output = runtime.forward(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits[:, -1, :]
            next_token = sampler.sample(logits)
            input_ids, attention_mask = _append_token(input_ids, attention_mask, next_token)
            generated += 1

            if _is_eos(next_token, runtime):
                logger.debug("Eos token encountered")
                break

    peak_memory_mb = get_peak_memory_mb()

    return BenchmarkResult.compute(
        engine_name="naive",
        prompt_tokens=prompt_tokens,
        generated_tokens=generated,
        ttft_ms=ttft_ms,
        total_time_ms=full_timer.elapsed_ms,
        peak_memory_mb=peak_memory_mb,
    )


def run_kvcache_once(
    runtime: Runtime,
    sampler: Sampler,
    prompt: str,
    config: GenerationConfig,
) -> BenchmarkResult:
    """
    Benchmark one run of the KV cache engine.
    TTFT is precisely measurable here:
        We call runtime.prefill() and time it directly.
        This is the exact boundary between prefill and decode.
    """
    encoded = runtime.tokenize(prompt, config.system_prompt)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_tokens = input_ids.shape[1]

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    reset_peak_memory()
    
    # Measure TTFT precisely
    with cuda_timer() as ttft_timer:
        logits, cache = runtime.prefill(input_ids, attention_mask)
        first_token = sampler.sample(logits)

    ttft_ms = ttft_timer.elapsed_ms

    #Full generation measurement
    torch.cuda.synchronize()
    torch.cuda.empty_cache() 
    reset_peak_memory()

    encoded = runtime.tokenize(prompt, config.system_prompt)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with cuda_timer() as full_timer:
        logits, cache = runtime.prefill(input_ids, attention_mask)
        next_token = sampler.sample(logits)
        input_ids, attention_mask = _append_token(input_ids, attention_mask, next_token)
        generated = 1

        for _ in range(config.max_new_tokens - 1):
            logits, cache = runtime.decode_step(
                token_id=next_token.unsqueeze(-1),
                attention_mask=attention_mask,
                cache=cache,
            )
            next_token = sampler.sample(logits)
            input_ids, attention_mask = _append_token(input_ids, attention_mask, next_token)
            generated += 1

            if _is_eos(next_token, runtime):
                logger.debug("EOS token encountered")
                break

    peak_memory_mb = get_peak_memory_mb()

    return BenchmarkResult.compute(
        engine_name="kv_cache",
        prompt_tokens=prompt_tokens,
        generated_tokens=generated,
        ttft_ms=ttft_ms,
        total_time_ms=full_timer.elapsed_ms,
        peak_memory_mb=peak_memory_mb,
    )

def run_scenario(
    runtime: Runtime,
    sampler: Sampler,
    prompt: str,
    config: GenerationConfig,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> tuple[AggregatedResult, AggregatedResult]:
    """
    Run a full benchmark scenario: warmup then measure.
    Returns aggregated results for both naive and kv_cache engines.
    """
    logger.info(
        "Warming up with %d runs before benchmarking", warmup_runs
    )

    #Warmup discard results
    for i in range(warmup_runs):
        logger.debug("Warmup run %d/%d", i + 1, warmup_runs)
        run_naive_once(runtime, sampler, prompt, config)
        run_kvcache_once(runtime, sampler, prompt, config)

    logger.info("Warmup complete. Starting %d benchmark runs.", benchmark_runs)

    #Benchmark runs
    naive_results = []
    kvcache_results = []

    for i in range(benchmark_runs):
        logger.debug("Benchmark run %d/%d", i + 1, benchmark_runs)
        naive_results.append(run_naive_once(runtime, sampler, prompt, config))
        kvcache_results.append(run_kvcache_once(runtime, sampler, prompt, config))

    return (
        AggregatedResult.from_results(naive_results),
        AggregatedResult.from_results(kvcache_results),
    )

def run_kvcache_scenario(
    runtime: Runtime,
    sampler: Sampler,
    prompt: str,
    config: GenerationConfig,
    engine_name: str,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> AggregatedResult:
    """
    Benchmark a single runtime using KV cache engine.
    Used for Phase 3 where we compare attention backends,
    not naive vs kvcache.
    """
    logger.info("Warming up %s with %d runs", engine_name, warmup_runs)
    for i in range(warmup_runs):
        logger.debug("Warmup %d/%d", i + 1, warmup_runs)
        run_kvcache_once(runtime, sampler, prompt, config)

    logger.info("Benchmarking %s with %d runs", engine_name, benchmark_runs)
    results = []
    for i in range(benchmark_runs):
        logger.debug("Run %d/%d", i + 1, benchmark_runs)
        result = run_kvcache_once(runtime, sampler, prompt, config)
        # Override engine name for reporting
        result = BenchmarkResult.compute(
            engine_name=engine_name,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            ttft_ms=result.ttft_ms,
            total_time_ms=result.total_time_ms,
            peak_memory_mb=result.peak_memory_mb,
        )
        results.append(result)

    return AggregatedResult.from_results(results)

# Internal helpers 
def _append_token(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    next_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_token = next_token.unsqueeze(-1)
    input_ids = torch.cat((input_ids, next_token), dim=-1)
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


def _is_eos(token: torch.Tensor, runtime: Runtime) -> bool:
    return bool((token == runtime.tokenizer.eos_token_id).all())