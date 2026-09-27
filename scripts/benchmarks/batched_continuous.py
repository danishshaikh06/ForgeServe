"""
Phase 5 Benchmark — Sequential vs Continuous Batching vs Batched Decode.

Three serving strategies compared:

    Sequential:
        One request at a time. Simplest possible.
        Baseline that everything must beat.

    Continuous batching (sequential loop):
        All requests decode simultaneously but each gets
        its own forward pass per step. N requests = N GPU calls.
        Phase 5 original implementation.

    Continuous batching (batched forward pass):
        All requests decoded in ONE forward pass per step.
        Weights loaded once for all N requests simultaneously.
        This is what production systems like vLLM do.

The gap between strategy 2 and strategy 3 is the value of
true GPU batching. Strategy 1 vs strategy 3 is the full
end-to-end value of continuous batching with batched decode.

Expected results
----------------
    Sequential → Batched:        3-4x throughput gain
    Seq loop   → Batched forward: 3-4x throughput gain
    Seq loop already showed ~1x so the batched forward
    is where the real gain materialises.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch

from forgeserve.engine.config import GenerationConfig
from forgeserve.engine.paged_generation import PagedGenerationEngine
from forgeserve.page_attention.block_manager import BlockManager
from forgeserve.logger import get_logger
from forgeserve.model.paged_runtime import PagedRuntime
from forgeserve.model.types import AttentionImplementation
from forgeserve.sampler.greedy import GreedySampler
from forgeserve.scheduler.continuous_batching import ContinuousBatchScheduler
from forgeserve.scheduler.request import RequestState

logger = get_logger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class RequestMetrics:
    request_id:       str
    generated_tokens: int
    finish_reason:    str
    wait_ms:          float
    total_ms:         float
    tokens_per_sec:   float


@dataclass
class ScenarioResult:
    strategy:           str
    num_requests:       int
    total_wall_ms:      float
    throughput_rps:     float
    throughput_tps:     float
    mean_latency_ms:    float
    p50_latency_ms:     float
    p95_latency_ms:     float
    p99_latency_ms:     float
    mean_wait_ms:       float
    max_wait_ms:        float
    total_tokens:       int
    total_steps:        int
    avg_batch_size:     float


# ── Strategies ────────────────────────────────────────────────────────────────

def run_sequential(
    runtime: PagedRuntime,
    requests: list[tuple[str, str, int]],
) -> tuple[ScenarioResult, list[RequestMetrics]]:
    """
    Process one request at a time using PagedGenerationEngine.
    Baseline — no concurrency.
    """
    engine  = PagedGenerationEngine(runtime=runtime, sampler=GreedySampler())
    metrics = []
    wall_start = time.perf_counter()

    for request_id, prompt, max_new_tokens in requests:
        config    = GenerationConfig(max_new_tokens=max_new_tokens)
        t0        = time.perf_counter()
        response  = engine.generate(prompt=prompt, config=config, request_id=request_id)
        total_ms  = (time.perf_counter() - t0) * 1_000

        metrics.append(RequestMetrics(
            request_id=request_id,
            generated_tokens=response.generated_tokens,
            finish_reason=response.finish_reason,
            wait_ms=0.0,
            total_ms=total_ms,
            tokens_per_sec=response.generated_tokens / total_ms * 1_000,
        ))

    wall_ms = (time.perf_counter() - wall_start) * 1_000
    return _summarise("sequential", requests, wall_ms, metrics, total_steps=0), metrics


def run_continuous_batching(
    runtime: PagedRuntime,
    requests: list[tuple[str, str, int]],
    max_batch_size: int | None = None,
) -> tuple[ScenarioResult, list[RequestMetrics]]:
    """
    Continuous batching with ONE batched forward pass per step.
    All running requests decode simultaneously in one GPU call.
    This is the Phase 5 Option B implementation.
    """
    scheduler = ContinuousBatchScheduler(
        runtime=runtime,
        sampler=GreedySampler(),
        max_batch_size=max_batch_size,
    )

    submission_times: dict[str, float] = {}
    states: dict[str, RequestState]    = {}

    wall_start = time.perf_counter()

    for request_id, prompt, max_new_tokens in requests:
        state = scheduler.add_request(request_id, prompt, max_new_tokens)
        submission_times[request_id] = time.perf_counter()
        states[request_id] = state

    total_steps  = 0
    total_tokens_decoded = 0
    safety_limit = sum(t for _, _, t in requests) * 3

    while scheduler.num_waiting > 0 or scheduler.num_running > 0:
        n = scheduler.step()
        total_steps += 1
        total_tokens_decoded += n   # n = batch size this step

        if total_steps > safety_limit:
            logger.error("Safety limit reached — forcing stop")
            for req in list(scheduler._running.values()):
                req.finish_reason = "force_stopped"
                scheduler._finish_request(req)
            break

    wall_ms = (time.perf_counter() - wall_start) * 1_000

    metrics = []
    for request_id, _, _ in requests:
        state   = states[request_id]
        sub_t   = submission_times[request_id]
        wait_ms = (state.started_at - sub_t) * 1_000 if state.started_at else 0.0
        tot_ms  = state.total_latency_ms or wall_ms

        metrics.append(RequestMetrics(
            request_id=request_id,
            generated_tokens=state.generated_tokens,
            finish_reason=state.finish_reason,
            wait_ms=wait_ms,
            total_ms=tot_ms,
            tokens_per_sec=state.generated_tokens / tot_ms * 1_000 if tot_ms > 0 else 0.0,
        ))

    avg_batch = total_tokens_decoded / total_steps if total_steps > 0 else 0.0
    result    = _summarise(
        "batched_forward", requests, wall_ms, metrics,
        total_steps=total_steps, avg_batch=avg_batch,
    )
    return result, metrics


# ── Scenarios ─────────────────────────────────────────────────────────────────

def build_uniform(n: int = 8) -> list[tuple[str, str, int]]:
    """Uniform lengths — hardest case for continuous batching to win."""
    prompt = (
        "Explain in detail how transformer attention mechanisms work, "
        "covering queries, keys, values, and why the mechanism is effective."
    )
    return [(f"req_{i:02d}", prompt, 80) for i in range(n)]


def build_mixed(n: int = 8) -> list[tuple[str, str, int]]:
    """Mixed lengths — realistic workload."""
    templates = [
        ("What is 2 + 2?",                                         10),
        ("Write a haiku about the ocean.",                         30),
        ("Explain what a transformer model is in one paragraph.",  80),
        ("Describe the water cycle in detail.",                   120),
        ("What is machine learning?",                              20),
        ("Explain how GPUs accelerate neural network training.",  100),
        ("What is the capital of France?",                          5),
        ("Summarise the key ideas behind attention mechanisms.",   90),
    ]
    return [
        (f"req_{i:02d}", templates[i % len(templates)][0], templates[i % len(templates)][1])
        for i in range(n)
    ]


def build_queue_pressure(n: int = 12) -> list[tuple[str, str, int]]:
    """Many short and medium requests — maximises queue churn."""
    templates = [
        ("What is 2+2?",                                10),
        ("Write a haiku.",                              20),
        ("Explain photosynthesis briefly.",             40),
        ("What is the capital of Japan?",               10),
        ("Describe how TCP works.",                     80),
        ("Write a short poem about rain.",              30),
        ("What is machine learning?",                   50),
        ("Explain gravity in one sentence.",            15),
        ("List three programming languages.",           20),
        ("Describe the water cycle.",                   60),
        ("What is 100 divided by 4?",                  10),
        ("Explain what DNA is.",                        45),
    ]
    return [(f"req_{i:02d}", t[0], t[1]) for i, t in enumerate(templates[:n])]


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_result(r: ScenarioResult) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {r.strategy.upper().replace('_', ' ')}")
    print(f"  Requests : {r.num_requests}  |  Wall time: {r.total_wall_ms:.0f} ms")
    print(f"{'=' * 65}")
    print(f"{'Metric':<35} {'Value':>20}")
    print(f"{'-' * 65}")
    print(f"{'Throughput (req/s)':<35} {r.throughput_rps:>19.2f}")
    print(f"{'Throughput (tok/s)':<35} {r.throughput_tps:>19.1f}")
    print(f"{'-' * 65}")
    print(f"{'Mean latency (ms)':<35} {r.mean_latency_ms:>19.1f}")
    print(f"{'P50  latency (ms)':<35} {r.p50_latency_ms:>19.1f}")
    print(f"{'P95  latency (ms)':<35} {r.p95_latency_ms:>19.1f}")
    print(f"{'P99  latency (ms)':<35} {r.p99_latency_ms:>19.1f}")
    print(f"{'-' * 65}")
    print(f"{'Mean queue wait (ms)':<35} {r.mean_wait_ms:>19.1f}")
    print(f"{'Max  queue wait (ms)':<35} {r.max_wait_ms:>19.1f}")
    print(f"{'-' * 65}")
    print(f"{'Total tokens generated':<35} {r.total_tokens:>20}")
    if r.total_steps > 0:
        print(f"{'Total decode steps':<35} {r.total_steps:>20}")
        print(f"{'Avg batch size / step':<35} {r.avg_batch_size:>19.2f}")
    print(f"{'=' * 65}")


def print_comparison(seq: ScenarioResult, batched: ScenarioResult) -> None:
    tps_gain     = batched.throughput_tps   / max(seq.throughput_tps,   0.001)
    rps_gain     = batched.throughput_rps   / max(seq.throughput_rps,   0.001)
    lat_change   = batched.mean_latency_ms  / max(seq.mean_latency_ms,  0.001)
    wall_gain    = seq.total_wall_ms        / max(batched.total_wall_ms, 0.001)

    print(f"\n{'=' * 70}")
    print(f"  COMPARISON  Sequential  →  Batched Forward")
    print(f"{'=' * 70}")
    print(f"{'Metric':<30} {'Sequential':>16} {'Batched':>16}")
    print(f"{'-' * 70}")
    print(f"{'Wall time (ms)':<30} {seq.total_wall_ms:>15.0f} {batched.total_wall_ms:>15.0f}")
    print(f"{'Throughput (tok/s)':<30} {seq.throughput_tps:>15.1f} {batched.throughput_tps:>15.1f}")
    print(f"{'Throughput (req/s)':<30} {seq.throughput_rps:>15.2f} {batched.throughput_rps:>15.2f}")
    print(f"{'Mean latency (ms)':<30} {seq.mean_latency_ms:>15.1f} {batched.mean_latency_ms:>15.1f}")
    print(f"{'P99 latency (ms)':<30} {seq.p99_latency_ms:>15.1f} {batched.p99_latency_ms:>15.1f}")
    print(f"{'Mean wait (ms)':<30} {seq.mean_wait_ms:>15.1f} {batched.mean_wait_ms:>15.1f}")
    if batched.total_steps > 0:
        print(f"{'Avg batch / step':<30} {'N/A':>16} {batched.avg_batch_size:>15.2f}")
    print(f"{'-' * 70}")
    print(f"{'Wall time speedup':<30} {wall_gain:>15.2f}x")
    print(f"{'Token throughput gain':<30} {tps_gain:>15.2f}x")
    print(f"{'Request throughput gain':<30} {rps_gain:>15.2f}x")
    print(f"{'Mean latency change':<30} {lat_change:>15.2f}x")
    print(f"{'=' * 70}")

    # Interpretation
    print("\n  Interpretation:")
    if tps_gain >= 3.0:
        print(f"  ✅ Strong batching gain: {tps_gain:.2f}x — GPU is bandwidth-bound")
        print(f"     Weight loading cost shared across {batched.avg_batch_size:.1f} requests/step")
    elif tps_gain >= 1.5:
        print(f"  ✅ Moderate batching gain: {tps_gain:.2f}x")
        print(f"     Partial compute overlap. Try larger batch or longer sequences.")
    else:
        print(f"  ℹ  Small gain: {tps_gain:.2f}x")
        print(f"     Sequences too short or batch too small to saturate GPU.")

    if lat_change > 2.0:
        print(f"  ℹ  Individual latency {lat_change:.2f}x worse — expected trade-off.")
        print(f"     Each request waits for batch-mates. Total wall time still wins.")
    print()


def print_per_request(label: str, metrics: list[RequestMetrics]) -> None:
    print(f"\n  {label}")
    print(f"  {'ID':<12} {'Tokens':>7} {'Reason':>8} "
          f"{'Wait(ms)':>10} {'Total(ms)':>10} {'tok/s':>8}")
    print("  " + "-" * 60)
    for m in metrics:
        print(
            f"  {m.request_id:<12} {m.generated_tokens:>7} "
            f"{m.finish_reason:>8} {m.wait_ms:>10.1f} "
            f"{m.total_ms:>10.1f} {m.tokens_per_sec:>8.1f}"
        )


def print_scaling_header() -> None:
    print(f"\n  {'Concurrency':>12} {'Seq tok/s':>12} "
          f"{'Batch tok/s':>12} {'Gain':>8} {'Avg batch':>10}")
    print("  " + "-" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(model_name: str, num_blocks: int = 512) -> None:
    """
    Run Phase 5 batched decode benchmark.

    Compares sequential serving against true batched forward pass
    continuous batching across four scenarios.
    """
    print(f"\nForgeServe Phase 5B — Batched Forward Pass Benchmark")
    print(f"Model     : {model_name}")
    print(f"KV blocks : {num_blocks}")

    # ── Initialise ────────────────────────────────────────────────────
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
    print(
        f"Ready. Pool: {num_blocks} blocks × {block_manager.block_size} tokens"
        f" = {num_blocks * block_manager.block_size:,} cached tokens max"
    )

    # ── Scenario 1: Uniform ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("SCENARIO 1 — Uniform lengths, N=4")
    print("Control: all same length, no early EOS")
    print("Batching should show clearest gain here")
    print("─" * 65)

    uniform = build_uniform(n=4)
    seq_u, seq_u_m   = run_sequential(runtime, uniform)
    bat_u, bat_u_m   = run_continuous_batching(runtime, uniform)

    print_result(seq_u)
    print_result(bat_u)
    print_comparison(seq_u, bat_u)

    # ── Scenario 2: Mixed ─────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("SCENARIO 2 — Mixed lengths, N=4")
    print("Realistic workload: short and long requests together")
    print("─" * 65)

    mixed = build_mixed(n=4)
    seq_m, seq_m_m   = run_sequential(runtime, mixed)
    bat_m, bat_m_m   = run_continuous_batching(runtime, mixed)

    print_result(seq_m)
    print_result(bat_m)
    print_comparison(seq_m, bat_m)
    print_per_request("Sequential per-request:", seq_m_m)
    print_per_request("Batched per-request:",    bat_m_m)

    # ── Scenario 3: Throughput scaling ───────────────────────────────
    print("\n" + "─" * 65)
    print("SCENARIO 3 — Throughput scaling")
    print("Key test: does tok/s grow with concurrency?")
    print("Sequential loop showed flat/declining — batched should rise")
    print("─" * 65)

    print_scaling_header()
    for n in [1, 2, 4, 8]:
        reqs = build_mixed(n=n)

        seq_s, _  = run_sequential(runtime, reqs)
        bat_s, _  = run_continuous_batching(runtime, reqs)

        gain = bat_s.throughput_tps / max(seq_s.throughput_tps, 0.001)
        print(
            f"  {n:>12} {seq_s.throughput_tps:>12.1f} "
            f"{bat_s.throughput_tps:>12.1f} {gain:>7.2f}x "
            f"{bat_s.avg_batch_size:>10.2f}"
        )

    # ── Scenario 4: Queue pressure ────────────────────────────────────
    print("\n" + "─" * 65)
    print("SCENARIO 4 — Queue pressure, 12 requests, batch cap=4")
    print("Expected: largest wall-time improvement")
    print("Short requests free slots immediately for waiting requests")
    print("─" * 65)

    queue = build_queue_pressure(n=12)
    seq_q, seq_q_m   = run_sequential(runtime, queue)
    bat_q, bat_q_m   = run_continuous_batching(runtime, queue, max_batch_size=4)

    print_result(seq_q)
    print_result(bat_q)
    print_comparison(seq_q, bat_q)
    print_per_request("Sequential per-request:", seq_q_m)
    print_per_request("Batched per-request:",    bat_q_m)

    print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s   = sorted(values)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)


def _summarise(
    strategy: str,
    requests: list[tuple[str, str, int]],
    wall_ms: float,
    metrics: list[RequestMetrics],
    total_steps: int = 0,
    avg_batch: float = 0.0,
) -> ScenarioResult:
    if not metrics:
        return ScenarioResult(
            strategy=strategy, num_requests=0,
            total_wall_ms=wall_ms, throughput_rps=0, throughput_tps=0,
            mean_latency_ms=0, p50_latency_ms=0, p95_latency_ms=0,
            p99_latency_ms=0, mean_wait_ms=0, max_wait_ms=0,
            total_tokens=0, total_steps=total_steps, avg_batch_size=avg_batch,
        )

    lats       = [m.total_ms         for m in metrics]
    waits      = [m.wait_ms          for m in metrics]
    total_toks = sum(m.generated_tokens for m in metrics)
    wall_sec   = wall_ms / 1_000

    return ScenarioResult(
        strategy=strategy,
        num_requests=len(metrics),
        total_wall_ms=wall_ms,
        throughput_rps=len(metrics) / wall_sec if wall_sec > 0 else 0,
        throughput_tps=total_toks   / wall_sec if wall_sec > 0 else 0,
        mean_latency_ms=statistics.mean(lats),
        p50_latency_ms=_pct(lats, 50),
        p95_latency_ms=_pct(lats, 95),
        p99_latency_ms=_pct(lats, 99),
        mean_wait_ms=statistics.mean(waits),
        max_wait_ms=max(waits),
        total_tokens=total_toks,
        total_steps=total_steps,
        avg_batch_size=avg_batch,
    )


if __name__ == "__main__":
    import sys
    model  = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    blocks = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    run(model, blocks)