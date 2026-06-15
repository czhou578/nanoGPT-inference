"""
Disaggregated prefill benchmark runner.

Usage:
1. Import near the bottom of nanogpt-disaggregated-prefill.py:

       from benchmarks.disaggregated_prefill_benchmark_runs import (
           run_disaggregated_prefill_benchmark_suite,
       )

2. After training, call:

       run_disaggregated_prefill_benchmark_suite(
           m,
           vocab_size=vocab_size,
           device=device,
           block_size=block_size,
       )

Compares: monolithic (chunked-prefill scheduler) vs disaggregated
(separate prefill/decode workers with KV cache handoff) across workloads
that stress different request counts, prompt lengths, and batch pressures.
"""

from benchmarks.disaggregated_prefill import run_monolithic_vs_disaggregated_benchmark


def run_disaggregated_prefill_benchmark_suite(
    model,
    *,
    vocab_size,
    device,
    block_size,
):
    """
    Run monolithic vs disaggregated comparisons across multiple
    workload scenarios.
    """

    print("\n=== Disaggregated Prefill Benchmark Suite ===")
    print(f"block_size={block_size}, device={device}")

    if block_size <= 32:
        configs = [
            {
                "name": "smoke_test",
                "kwargs": dict(
                    num_requests=4,
                    prompt_len=8,
                    max_new_tokens=4,
                    max_batch_size=4,
                    token_budget=12,
                    max_kv_tokens=128,
                ),
            },
            {
                "name": "staggered_arrivals",
                "kwargs": dict(
                    num_requests=6,
                    prompt_len=12,
                    max_new_tokens=6,
                    max_batch_size=4,
                    token_budget=16,
                    max_kv_tokens=256,
                ),
            },
            {
                "name": "batch_pressure",
                "kwargs": dict(
                    num_requests=8,
                    prompt_len=8,
                    max_new_tokens=4,
                    max_batch_size=2,
                    token_budget=8,
                    max_kv_tokens=128,
                ),
            },
            {
                "name": "long_decode",
                "kwargs": dict(
                    num_requests=4,
                    prompt_len=8,
                    max_new_tokens=16,
                    max_batch_size=4,
                    token_budget=16,
                    max_kv_tokens=256,
                ),
            },
        ]
    else:
        configs = [
            {
                "name": "smoke_test",
                "kwargs": dict(
                    num_requests=4,
                    prompt_len=16,
                    max_new_tokens=8,
                    max_batch_size=4,
                    token_budget=16,
                    max_kv_tokens=256,
                ),
            },
            {
                "name": "staggered_arrivals",
                "kwargs": dict(
                    num_requests=8,
                    prompt_len=24,
                    max_new_tokens=10,
                    max_batch_size=4,
                    token_budget=16,
                    max_kv_tokens=512,
                ),
            },
            {
                "name": "batch_pressure",
                "kwargs": dict(
                    num_requests=12,
                    prompt_len=16,
                    max_new_tokens=8,
                    max_batch_size=2,
                    token_budget=12,
                    max_kv_tokens=256,
                ),
            },
            {
                "name": "long_prompts",
                "kwargs": dict(
                    num_requests=6,
                    prompt_len=40,
                    max_new_tokens=8,
                    max_batch_size=4,
                    token_budget=24,
                    max_kv_tokens=512,
                ),
            },
            {
                "name": "long_decode",
                "kwargs": dict(
                    num_requests=6,
                    prompt_len=16,
                    max_new_tokens=32,
                    max_batch_size=4,
                    token_budget=16,
                    max_kv_tokens=512,
                ),
            },
            {
                "name": "stress_test",
                "kwargs": dict(
                    num_requests=16,
                    prompt_len=16,
                    max_new_tokens=10,
                    max_batch_size=4,
                    token_budget=16,
                    max_kv_tokens=512,
                ),
            },
        ]

    results = {}

    for config in configs:
        name = config["name"]
        kwargs = config["kwargs"]

        print(f"\n\n=== {name} ===")
        print(kwargs)

        results[name] = run_monolithic_vs_disaggregated_benchmark(
            model,
            vocab_size=vocab_size,
            device=device,
            seed=1337,
            temperature=1.0,
            **kwargs,
        )

    return results
