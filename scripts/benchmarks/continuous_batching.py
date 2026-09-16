"""
Phase 5 Benchmark — Continuous Batching vs Sequential Serving.

What we measure
---------------
The fundamental question of Phase 5 is not "how fast is one request"
but "how efficiently can we serve many requests simultaneously."

Two serving strategies are compared:

    Sequential:
        Process each request fully before starting the next.
        Simple. GPU sits idle between requests.
        Baseline that any batching system must beat.

    Continuous batching:
        All admitted requests decode one token per step.
        GPU always busy. Short requests free slots for new ones.
        The strategy implemented by ForgeServe Phase 5.

Scenario design
---------------
Mixed-length requests reveal the true advantage of continuous batching.
Uniform-length requests show minimal benefit because no request finishes
early to free slots. We use both to make this visible in the numbers.

Metrics
-------
    throughput_rps      requests completed per second
    throughput_tps      tokens generated per second
    mean_latency_ms     average request latency
    p50_latency_ms      median latency
    p95_latency_ms      95th percentile latency
    p99_latency_ms      99th percentile latency
    mean_wait_ms        average time from submission to admission
    gpu_active_pct      percentage of wall time the GPU was decoding
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import torch

from forgeserve.page_attention.block_manager import BlockManager
from forgeserve.model.paged_runtime import PagedRuntime
from forgeserve.model.types import AttentionImplementation
from forgeserve.sampler.greedy import GreedySampler
from forgeserve.scheduler.continuous_batching import ContinuousBatching
from forgeserve.scheduler.request import RequestState
from forgeserve.logger import get_logger

logger = get_logger(__name__)


# Result dataclasses
@dataclass
class RequestMetrics:
    """
    Per-request timing collected after the request finishes.

    All times are in milliseconds.
    """
    request_id:      str
    prompt_tokens:   int
    generated_tokens: int
    finish_reason:   str
    wait_ms:         float    # time from submission to admission
    ttft_ms:         float    # time from submission to first token
    total_ms:        float    # time from submission to completion
    tokens_per_sec:  float    # generated_tokens / total_ms * 1000


@dataclass
class BenchmarkSummary:
    """
    Aggregate statistics across all requests in one scenario run.

    All latency values in milliseconds.
    All throughput values per second.
    """
    strategy:           str
    scenario_name:      str
    num_requests:       int
    total_wall_ms:      float

    # Throughput
    throughput_rps:     float   # requests per second
    throughput_tps:     float   # tokens per second

    # Latency distribution
    mean_latency_ms:    float
    p50_latency_ms:     float
    p95_latency_ms:     float
    p99_latency_ms:     float

    # Queue health
    mean_wait_ms:       float
    max_wait_ms:        float

    # GPU utilisation proxy
    total_tokens:       int
    total_steps:        int


# ── Sequential baseline ───────────────────────────────────────────────────────

def run_sequential(
    runtime: PagedRuntime,
    requests: list[tuple[str, str, int]],
) -> tuple[BenchmarkSummary, list[RequestMetrics]]:
    """
    Sequential baseline: process one request completely before the next.

    Uses ``PagedGenerationEngine`` directly — no scheduler involved.
    This is the simplest possible serving strategy and the one that
    any continuous batching system must outperform.

    Parameters
    ----------
    runtime:
        Fully initialised ``PagedRuntime`` with block manager attached.
    requests:
        List of ``(request_id, prompt, max_new_tokens)`` tuples.

    Returns
    -------
    summary:
        Aggregate metrics across all requests.
    per_request:
        One ``RequestMetrics`` per request in arrival order.
    """
    from forgeserve.engine.paged_generation import PagedGenerationEngine
    from forgeserve.engine.config import GenerationConfig

    sampler = GreedySampler()
    engine = PagedGenerationEngine(runtime=runtime, sampler=sampler)

    per_request: list[RequestMetrics] = []
    wall_start = time.perf_counter()

    for request_id, prompt, max_new_tokens in requests:
        config = GenerationConfig(max_new_tokens=max_new_tokens)
        req_start = time.perf_counter()

        response = engine.generate(
            prompt=prompt,
            config=config,
            request_id=request_id,
        )

        req_end = time.perf_counter()
        total_ms = (req_end - req_start) * 1_000

        per_request.append(RequestMetrics(
            request_id=request_id,
            prompt_tokens=0,            # not tracked in sequential engine
            generated_tokens=response.generated_tokens,
            finish_reason=response.finish_reason,
            wait_ms=0.0,                # no queue in sequential
            ttft_ms=0.0,                # not measured separately
            total_ms=total_ms,
            tokens_per_sec=(response.generated_tokens / total_ms * 1_000)
            if total_ms > 0 else 0.0,
        ))

    wall_end = time.perf_counter()
    wall_ms = (wall_end - wall_start) * 1_000

    return _summarise("sequential", "all", wall_ms, per_request), per_request


# ── Continuous batching ───────────────────────────────────────────────────────

def run_continuous_batching(
    runtime: PagedRuntime,
    requests: list[tuple[str, str, int]],
    max_batch_size: int | None = None,
) -> tuple[BenchmarkSummary, list[RequestMetrics]]:
    """
        Continuous batching: all requests decode one token per step.
    
        All requests are submitted at t=0 (simultaneous arrival).
        The scheduler runs ``step()`` in a loop until every request
        has finished.
    
        Parameters
        ----------
        runtime:
            Fully initialised ``PagedRuntime`` with block manager attached.
        requests:
            List of ``(request_id, prompt, max_new_tokens)`` tuples.
        max_batch_size:
            Optional cap on simultaneous running requests.
    
        Returns
        -------
        summary:
            Aggregate metrics across all requests.
        per_request:
            One ``RequestMetrics`` per request in arrival order.
     """
    sampler = GreedySampler()
    scheduler = ContinuousBatching(
        runtime=runtime,
        sampler=sampler,
        max_batch_size=max_batch_size,
    )

    submission_times: dict[str, float] = {}
    request_states: dict[str, RequestState] = {}

    # Submit all requests and store states with null check
    wall_start = time.perf_counter()

    for request_id, prompt, max_new_tokens in requests:
        t = time.perf_counter()
        state = scheduler.add_request(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        # Guard: add_request must return a RequestState
        if state is None:
            raise RuntimeError(
                f"scheduler.add_request returned None for '{request_id}'. "
                "Check that ContinuousBatchScheduler.add_request has "
                "a return statement."
            )

        submission_times[request_id] = t
        request_states[request_id] = state

    # Run until all requests are done
    total_steps = 0
    max_steps = sum(max_new_tokens for _, _, max_new_tokens in requests) * 2

    while scheduler.num_waiting > 0 or scheduler.num_running > 0:
        scheduler.step()
        total_steps += 1

        # Safety valve — prevents infinite loop if termination is broken
        if total_steps > max_steps:
            logger.error(
                "Benchmark exceeded max_steps=%d. "
                "Termination condition may be broken. "
                "running=%d waiting=%d",
                max_steps,
                scheduler.num_running,
                scheduler.num_waiting,
            )
            # Force finish all remaining requests
            for req in list(scheduler._running.values()):
                req.finish_reason = "force_stopped"
                scheduler._finish_request(req)
            break

        if total_steps % 10 == 0:
            snap = scheduler.snapshot()
            logger.debug(
                "Step %d: running=%d waiting=%d free_blocks=%d",
                total_steps,
                snap["running"],
                snap["waiting"],
                snap["free_blocks"],
            )

    wall_end = time.perf_counter()
    wall_ms = (wall_end - wall_start) * 1_000

    # Collect per-request metrics
    per_request: list[RequestMetrics] = []

    for request_id, _, _ in requests:
        state = request_states.get(request_id)

        # Defensive check with clear error message
        if state is None:
            logger.error(
                "No state found for request '%s' after benchmark. "
                "This is a bug in state collection.",
                request_id,
            )
            continue

        sub_time = submission_times[request_id]

        wait_ms = (
            (state.started_at - sub_time) * 1_000
            if state.started_at is not None
            else 0.0
        )

        ttft_ms = (
            (state.started_at - sub_time) * 1_000
            if state.started_at is not None
            else 0.0
        )

        total_ms = (
            state.total_latency_ms
            if state.total_latency_ms is not None
            else wall_ms
        )

        per_request.append(RequestMetrics(
            request_id=request_id,
            prompt_tokens=state.prompt_tokens,
            generated_tokens=state.generated_tokens,
            finish_reason=state.finish_reason,
            wait_ms=wait_ms,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_sec=(
                state.generated_tokens / total_ms * 1_000
            ) if total_ms > 0 else 0.0,
        ))

    summary = _summarise(
        "continuous_batching",
        "all",
        wall_ms,
        per_request,
        total_steps=total_steps,
    )

    return summary, per_request

def build_queue_pressure_scenario() -> list[tuple[str, str, int]]:
    """
    Simulate a burst of requests larger than initial batch capacity.
    Some requests must wait. As running requests finish, waiting
    requests are immediately admitted. This is continuous batching's
    core value proposition.
    """
    prompts = [
        ("What is 2+2?",                                          10),
        ("Write a haiku.",                                         20),
        ("Explain photosynthesis briefly.",                        40),
        ("What is the capital of Japan?",                          10),
        ("Describe how TCP works.",                               80),
        ("Write a short poem about rain.",                        30),
        ("What is machine learning?",                             50),
        ("Explain gravity in one sentence.",                      15),
        ("List three programming languages.",                     20),
        ("Describe the water cycle.",                             60),
        ("What is 100 divided by 4?",                            10),
        ("Explain what DNA is.",                                  45),
    ]
    return [
        (f"req_{i:02d}", prompt, tokens)
        for i, (prompt, tokens) in enumerate(prompts)
    ]


# ── Report ────────────────────────────────────────────────────────────────────

def print_summary(summary: BenchmarkSummary) -> None:
    """Print a formatted benchmark summary table."""
    print("\n" + "=" * 65)
    print(f"  {summary.strategy.upper().replace('_', ' ')}")
    print(f"  Scenario : {summary.scenario_name}")
    print(f"  Requests : {summary.num_requests}")
    print(f"  Wall time: {summary.total_wall_ms:.0f} ms")
    print("=" * 65)
    print(f"{'Metric':<35} {'Value':>20}")
    print("-" * 65)
    print(f"{'Throughput (req/s)':<35} {summary.throughput_rps:>19.2f}")
    print(f"{'Throughput (tok/s)':<35} {summary.throughput_tps:>19.1f}")
    print("-" * 65)
    print(f"{'Mean latency (ms)':<35} {summary.mean_latency_ms:>19.1f}")
    print(f"{'P50 latency (ms)':<35} {summary.p50_latency_ms:>19.1f}")
    print(f"{'P95 latency (ms)':<35} {summary.p95_latency_ms:>19.1f}")
    print(f"{'P99 latency (ms)':<35} {summary.p99_latency_ms:>19.1f}")
    print("-" * 65)
    print(f"{'Mean queue wait (ms)':<35} {summary.mean_wait_ms:>19.1f}")
    print(f"{'Max queue wait (ms)':<35} {summary.max_wait_ms:>19.1f}")
    print("-" * 65)
    print(f"{'Total tokens generated':<35} {summary.total_tokens:>20}")
    if summary.total_steps > 0:
        print(f"{'Total decode steps':<35} {summary.total_steps:>20}")
        avg_batch = summary.total_tokens / max(summary.total_steps, 1)
        print(f"{'Avg batch size per step':<35} {avg_batch:>19.2f}")
    print("=" * 65 + "\n")


def print_comparison(
    sequential: BenchmarkSummary,
    batched: BenchmarkSummary,
) -> None:
    """Print side-by-side comparison of sequential vs continuous batching."""
    tps_gain = batched.throughput_tps / max(sequential.throughput_tps, 1)
    rps_gain = batched.throughput_rps / max(sequential.throughput_rps, 1)
    latency_change = batched.mean_latency_ms / max(sequential.mean_latency_ms, 1)

    print("\n" + "=" * 70)
    print("  COMPARISON: Sequential vs Continuous Batching")
    print("=" * 70)
    print(f"{'Metric':<30} {'Sequential':>16} {'Cont. Batch':>16}")
    print("-" * 70)
    print(
        f"{'Throughput (req/s)':<30} "
        f"{sequential.throughput_rps:>15.2f} "
        f"{batched.throughput_rps:>15.2f}"
    )
    print(
        f"{'Throughput (tok/s)':<30} "
        f"{sequential.throughput_tps:>15.1f} "
        f"{batched.throughput_tps:>15.1f}"
    )
    print(
        f"{'Mean latency (ms)':<30} "
        f"{sequential.mean_latency_ms:>15.1f} "
        f"{batched.mean_latency_ms:>15.1f}"
    )
    print(
        f"{'P99 latency (ms)':<30} "
        f"{sequential.p99_latency_ms:>15.1f} "
        f"{batched.p99_latency_ms:>15.1f}"
    )
    print(
        f"{'Mean queue wait (ms)':<30} "
        f"{sequential.mean_wait_ms:>15.1f} "
        f"{batched.mean_wait_ms:>15.1f}"
    )
    print("-" * 70)
    print(f"{'Token throughput gain':<30} {tps_gain:>15.2f}x")
    print(f"{'Request throughput gain':<30} {rps_gain:>15.2f}x")
    print(f"{'Mean latency change':<30} {latency_change:>15.2f}x")
    print("=" * 70)

    # Interpretation
    print("\n  Interpretation:")
    if tps_gain > 1.5:
        print(f"  ✅ Strong throughput gain: {tps_gain:.2f}x more tokens/sec")
    elif tps_gain > 1.1:
        print(f"  ✅ Moderate throughput gain: {tps_gain:.2f}x")
    else:
        print(f"  ℹ  Small throughput gain: {tps_gain:.2f}x")
        print("     Expected with uniform-length requests or small batch.")

    if latency_change > 1.5:
        print(
            f"  ℹ  Individual latency increased {latency_change:.2f}x "
            "— expected trade-off for higher throughput."
        )
    print()


def print_per_request_table(metrics: list[RequestMetrics]) -> None:
    """Print a per-request breakdown table."""
    print(f"\n  {'ID':<12} {'Generated':>10} {'Reason':>8} "
          f"{'Wait(ms)':>10} {'Total(ms)':>10} {'tok/s':>8}")
    print("  " + "-" * 62)
    for m in metrics:
        print(
            f"  {m.request_id:<12} {m.generated_tokens:>10} "
            f"{m.finish_reason:>8} {m.wait_ms:>10.1f} "
            f"{m.total_ms:>10.1f} {m.tokens_per_sec:>8.1f}"
        )
    print()


# ── Scenario definitions ──────────────────────────────────────────────────────

def build_uniform_requests(n: int = 8) -> list[tuple[str, str, int]]:
    """
    All requests generate the same number of tokens.

    Continuous batching advantage: minimal.
    No request finishes early to free slots.
    Use this as a control scenario.
    """
    prompt = (
        "Explain in detail how transformer attention mechanisms work, "
        "covering queries, keys, values, and the mathematical operations."
    )
    return [
        (f"req_{i:02d}", prompt, 80)
        for i in range(n)
    ]


def build_mixed_requests(n: int = 8) -> list[tuple[str, str, int]]:
    """
    Requests with highly varied generation lengths.

    Continuous batching advantage: large.
    Short requests finish early and free slots for new arrivals.
    This is the scenario that makes continuous batching's value visible.
    """
    templates = [
        ("What is 2 + 2?",                                              10),
        ("Write a haiku about the ocean.",                              30),
        ("Explain what a transformer model is in one paragraph.",       80),
        ("Describe the water cycle in detail.",                        120),
        ("What is machine learning?",                                   20),
        ("Explain how GPUs accelerate neural network training.",       100),
        ("What is the capital of France?",                               5),
        ("Summarise the key ideas behind attention mechanisms.",        90),
    ]

    requests = []
    for i in range(n):
        prompt, max_tokens = templates[i % len(templates)]
        requests.append((f"req_{i:02d}", prompt, max_tokens))

    return requests


# ── Main entry point ──────────────────────────────────────────────────────────
def run(model_name: str, num_blocks: int = 512) -> None:
    print(f"\nForgeServe Phase 5 Benchmark — Continuous Batching")
    print(f"Model     : {model_name}")
    print(f"KV blocks : {num_blocks}")

    print("\nLoading model...")
    runtime = PagedRuntime(
        model_name=model_name,
        attention=AttentionImplementation.SDPA,
    )
    block_manager = BlockManager.from_model_config(
        num_blocks=num_blocks,
        block_size=16,
        model=runtime.model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    runtime.attach_block_manager(block_manager)
    print(f"Model loaded. Pool: {num_blocks} blocks × {block_manager.block_size} tokens")

    # ── Scenario 1: Uniform ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("SCENARIO 1 — Uniform request lengths (control)")
    print("Expected: small continuous batching advantage")
    print("─" * 65)

    uniform_requests = build_uniform_requests(n=4)
    seq_summary_u, seq_metrics_u = run_sequential(runtime, uniform_requests)
    cb_summary_u, cb_metrics_u   = run_continuous_batching(runtime, uniform_requests)
    print_summary(seq_summary_u)
    print_summary(cb_summary_u)
    print_comparison(seq_summary_u, cb_summary_u)

    # ── Scenario 2: Mixed ─────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("SCENARIO 2 — Mixed request lengths (real-world simulation)")
    print("Expected: moderate continuous batching advantage")
    print("─" * 65)

    mixed_requests = build_mixed_requests(n=4)
    seq_summary_m, seq_metrics_m = run_sequential(runtime, mixed_requests)
    cb_summary_m, cb_metrics_m   = run_continuous_batching(runtime, mixed_requests)
    print_summary(seq_summary_m)
    print_summary(cb_summary_m)
    print_comparison(seq_summary_m, cb_summary_m)
    print("  Per-request detail — Sequential:")
    print_per_request_table(seq_metrics_m)
    print("  Per-request detail — Continuous Batching:")
    print_per_request_table(cb_metrics_m)

    # ── Scenario 3: Throughput scaling ───────────────────────────────
    print("\n" + "─" * 65)
    print("SCENARIO 3 — Throughput scaling with concurrent requests")
    print("Expected: throughput grows as concurrency increases")
    print("─" * 65)
    print(f"\n  {'Concurrency':>12} {'Throughput (tok/s)':>20} {'Mean latency (ms)':>20}")
    print("  " + "-" * 55)

    for n_requests in [1, 2, 4]:
        requests = build_mixed_requests(n=n_requests)
        summary, _ = run_continuous_batching(runtime, requests)
        print(
            f"  {n_requests:>12} "
            f"{summary.throughput_tps:>20.1f} "
            f"{summary.mean_latency_ms:>20.1f}"
        )

    # ── Scenario 4: Queue pressure ────────────────────────────────────
    print("\n" + "─" * 65)
    print("SCENARIO 4 — Queue pressure (12 requests, batch cap=4)")
    print("Expected: largest continuous batching advantage")
    print("Freed slots immediately admit waiting requests")
    print("─" * 65)

    queue_requests = build_queue_pressure_scenario()
    seq_summary_q, seq_metrics_q = run_sequential(runtime, queue_requests)
    cb_summary_q, cb_metrics_q   = run_continuous_batching(
        runtime,
        queue_requests,
        max_batch_size=4,
    )
    print_summary(seq_summary_q)
    print_summary(cb_summary_q)
    print_comparison(seq_summary_q, cb_summary_q)
    print("  Per-request — Sequential:")
    print_per_request_table(seq_metrics_q)
    print("  Per-request — Continuous Batching:")
    print_per_request_table(cb_metrics_q)

    print()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _percentile(values: list[float], p: float) -> float:
    """
    Return the p-th percentile of values using linear interpolation.

    Parameters
    ----------
    values:
        Non-empty list of floats.
    p:
        Percentile in range [0, 100].
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _summarise(
    strategy: str,
    scenario_name: str,
    wall_ms: float,
    metrics: list[RequestMetrics],
    total_steps: int = 0,
) -> BenchmarkSummary:
    """
    Compute aggregate statistics from per-request metrics.

    Parameters
    ----------
    strategy:
        Label for the serving strategy (sequential / continuous_batching).
    scenario_name:
        Label for the scenario (uniform / mixed).
    wall_ms:
        Total wall-clock time for the entire run in milliseconds.
    metrics:
        Per-request metrics collected during the run.
    total_steps:
        Number of scheduler steps executed (0 for sequential).
    """
    if not metrics:
        return BenchmarkSummary(
            strategy=strategy, scenario_name=scenario_name,
            num_requests=0, total_wall_ms=wall_ms,
            throughput_rps=0, throughput_tps=0,
            mean_latency_ms=0, p50_latency_ms=0,
            p95_latency_ms=0, p99_latency_ms=0,
            mean_wait_ms=0, max_wait_ms=0,
            total_tokens=0, total_steps=total_steps,
        )

    latencies  = [m.total_ms  for m in metrics]
    waits      = [m.wait_ms   for m in metrics]
    total_toks = sum(m.generated_tokens for m in metrics)
    wall_sec   = wall_ms / 1_000

    return BenchmarkSummary(
        strategy=strategy,
        scenario_name=scenario_name,
        num_requests=len(metrics),
        total_wall_ms=wall_ms,
        throughput_rps=len(metrics) / wall_sec if wall_sec > 0 else 0,
        throughput_tps=total_toks / wall_sec   if wall_sec > 0 else 0,
        mean_latency_ms=statistics.mean(latencies),
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        p99_latency_ms=_percentile(latencies, 99),
        mean_wait_ms=statistics.mean(waits),
        max_wait_ms=max(waits),
        total_tokens=total_toks,
        total_steps=total_steps,
    )


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    blocks = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    run(model, blocks)