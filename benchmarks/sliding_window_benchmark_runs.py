"""
Sliding window KV eviction benchmark configs.

Usage in nanogpt-sliding-window.py:

    from benchmarks.sliding_window_benchmark_runs import (
        run_sliding_window_benchmark_suite,
    )

    run_sliding_window_benchmark_suite(
        m, vocab_size=vocab_size, device=device, block_size=block_size,
    )
"""

from benchmarks.sliding_window import run_window_vs_full_benchmark, run_quality_sweep


def run_sliding_window_benchmark_suite(model, *, vocab_size, device, block_size):
    print("\n" + "=" * 60)
    print("  Sliding Window KV Eviction — Benchmark Suite")
    print("=" * 60)
    print(f"block_size={block_size}, device={device}\n")

    # ── Benchmark 1: Memory savings ──────────────────────────────
    print("\n--- Benchmark 1: Memory Savings (window=20 vs full cache) ---")
    run_window_vs_full_benchmark(
        model,
        vocab_size=vocab_size,
        device=device,
        workload_name="long_generation",
        sliding_window=20,
        max_batch_size=3,
        token_budget=16,
        prefill_chunk_size=8,
        max_kv_tokens=200,
        seed=1337,
    )

    # ── Benchmark 2: Preemption reduction ────────────────────────
    print("\n\n--- Benchmark 2: Preemption Reduction Under Tight Memory ---")
    run_window_vs_full_benchmark(
        model,
        vocab_size=vocab_size,
        device=device,
        workload_name="tight_memory",
        sliding_window=20,
        max_batch_size=3,
        token_budget=12,
        prefill_chunk_size=8,
        max_kv_tokens=50,
        seed=1337,
    )

    # ── Benchmark 3: Quality sweep ───────────────────────────────
    # Keep max_new within block_size to avoid position embedding OOB
    max_new = min(40, block_size - 12)
    print(f"\n\n--- Benchmark 3: Quality vs Window Size (max_new={max_new}) ---")
    run_quality_sweep(
        model,
        vocab_size=vocab_size,
        device=device,
        window_sizes=(8, 16, 32, 48),
        prompt_len=10,
        max_new=max_new,
        max_kv_tokens=200,
        seed=1337,
    )

    # ── Benchmark 4: Batch capacity ──────────────────────────────
    print("\n\n--- Benchmark 4: Batch Capacity (window=16 vs full cache) ---")
    run_window_vs_full_benchmark(
        model,
        vocab_size=vocab_size,
        device=device,
        workload_name="batch_capacity",
        sliding_window=16,
        max_batch_size=8,
        token_budget=16,
        prefill_chunk_size=8,
        max_kv_tokens=80,
        seed=1337,
    )

    print("\n" + "=" * 60)
    print("  Benchmark suite complete.")
    print("=" * 60)
