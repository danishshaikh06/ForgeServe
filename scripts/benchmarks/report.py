"""
Results formatting and output.
Prints a clean comparison table and speedup analysis.
"""

from __future__ import annotations
from metrics import AggregatedResult

def print_comparison(
    naive: AggregatedResult,
    kvcache: AggregatedResult,
) -> None:
    """Print a side-by-side comparison of naive vs kv_cache."""
    speedup_total = naive.mean_total_ms / kvcache.mean_total_ms
    speedup_tpot = naive.mean_tpot_ms / kvcache.mean_tpot_ms
    memory_overhead = kvcache.mean_peak_memory_mb - naive.mean_peak_memory_mb

    print("\n" + "=" * 65)
    print(f"  BENCHMARK RESULTS")
    print(f"  Prompt tokens : {naive.prompt_tokens}")
    print(f"  Generated     : {naive.generated_tokens} (naive) / {kvcache.generated_tokens} (kv_cache)")
    print(f"  Runs          : {naive.runs}")
    print("=" * 65)
    print(f"{'Metric':<28} {'Naive':>14} {'KV Cache':>14}")
    print("-" * 65)
    print(f"{'TTFT (ms)':<28} {naive.mean_ttft_ms:>13.1f} {kvcache.mean_ttft_ms:>13.1f}")
    print(f"{'TPOT (ms)':<28} {naive.mean_tpot_ms:>13.1f} {kvcache.mean_tpot_ms:>13.1f}  ±{kvcache.std_tpot_ms:.1f}")
    print(f"{'Total time (ms)':<28} {naive.mean_total_ms:>13.1f} {kvcache.mean_total_ms:>13.1f}")
    print(f"{'Throughput (tok/s)':<28} {naive.mean_tokens_per_second:>13.1f} {kvcache.mean_tokens_per_second:>13.1f}")
    print(f"{'Peak memory (MB)':<28} {naive.mean_peak_memory_mb:>13.1f} {kvcache.mean_peak_memory_mb:>13.1f}")
    print("-" * 65)
    print(f"{'Total speedup':<28} {speedup_total:>13.2f}x")
    print(f"{'TPOT speedup':<28} {speedup_tpot:>13.2f}x")
    print(f"{'Memory overhead (MB)':<28} {memory_overhead:>+13.1f}")
    print("=" * 65 + "\n")

def print_comparison_eager_sdpa(
    first: AggregatedResult,
    second: AggregatedResult,
) -> None:
    """Print a side-by-side comparison of any two aggregated results."""

    speedup_total = first.mean_total_ms / second.mean_total_ms
    speedup_tpot  = first.mean_tpot_ms  / second.mean_tpot_ms
    memory_delta  = second.mean_peak_memory_mb - first.mean_peak_memory_mb

    first_label  = first.engine_name.upper()
    second_label = second.engine_name.upper()

    print("\n" + "=" * 65)
    print(f"  BENCHMARK RESULTS")
    print(f"  Prompt tokens : {first.prompt_tokens}")
    print(f"  Generated     : {first.generated_tokens} ({first_label}) / {second.generated_tokens} ({second_label})")
    print(f"  Runs          : {first.runs}")
    print("=" * 65)
    print(f"{'Metric':<28} {first_label:>14} {second_label:>14}")
    print("-" * 65)
    print(f"{'TTFT (ms)':<28} {first.mean_ttft_ms:>13.1f} {second.mean_ttft_ms:>13.1f}")
    print(f"{'TPOT (ms)':<28} {first.mean_tpot_ms:>13.1f} {second.mean_tpot_ms:>13.1f}  ±{second.std_tpot_ms:.1f}")
    print(f"{'Total time (ms)':<28} {first.mean_total_ms:>13.1f} {second.mean_total_ms:>13.1f}")
    print(f"{'Throughput (tok/s)':<28} {first.mean_tokens_per_second:>13.1f} {second.mean_tokens_per_second:>13.1f}")
    print(f"{'Peak memory (MB)':<28} {first.mean_peak_memory_mb:>13.1f} {second.mean_peak_memory_mb:>13.1f}")
    print("-" * 65)
    print(f"{'Total speedup':<28} {speedup_total:>13.2f}x")
    print(f"{'TPOT speedup':<28} {speedup_tpot:>13.2f}x")
    print(f"{'Memory delta (MB)':<28} {memory_delta:>+13.1f}")
    print("=" * 65 + "\n")


def print_scenario_header(prompt_tokens: int, max_new_tokens: int) -> None:
    print(f"\n>>> Scenario: prompt={prompt_tokens} tokens, generate={max_new_tokens} tokens")