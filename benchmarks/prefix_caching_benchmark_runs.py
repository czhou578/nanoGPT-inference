"""
Suggested prefix-caching benchmark runs.

Use this with the benchmark file that defines:

    run_no_prefix_vs_prefix_cache_benchmark(...)

How to use:
1. Import this file near the bottom of nanogpt-prefix-caching.py:

       from benchmarks.prefix_caching_benchmark_runs import (
           run_prefix_caching_benchmark_suite,
       )

2. After training/loading your model, call:

       run_prefix_caching_benchmark_suite(
           m,
           vocab_size=vocab_size,
           device=device,
           block_size=block_size,
       )

These runs compare full prompt prefill against prefix caching for repeated
block-aligned prompt prefixes.
"""

from benchmarks.prefix_caching import run_no_prefix_vs_prefix_cache_benchmark


def run_prefix_caching_benchmark_suite(
    model,
    *,
    vocab_size,
    device,
    block_size,
):
    """
    Run no-prefix-cache vs prefix-cache comparisons.

    All prompt lengths are kept below the model block_size. Every shared-prefix
    workload includes a unique suffix so first-token logits are produced by a
    real suffix forward pass after cached prefix blocks are loaded.
    """

    print("\n=== Prefix Caching Benchmark Suite ===")
    print(f"block_size={block_size}, device={device}")

    if block_size <= 32:
        configs = [
            {
                "name": "shared_prefix_basic",
                "kwargs": dict(
                    workload_name="shared_prefix",
                    num_requests=8,
                    shared_prefix_len=12,
                    unique_suffix_len=4,
                    max_new_tokens=4,
                    num_groups=1,
                    prefix_block_size=4,
                    max_cache_blocks=32,
                ),
            },
            {
                "name": "high_reuse_many_requests",
                "kwargs": dict(
                    workload_name="shared_prefix",
                    num_requests=16,
                    shared_prefix_len=16,
                    unique_suffix_len=4,
                    max_new_tokens=3,
                    num_groups=1,
                    prefix_block_size=4,
                    max_cache_blocks=32,
                ),
            },
            {
                "name": "multi_prefix_groups",
                "kwargs": dict(
                    workload_name="shared_prefix",
                    num_requests=16,
                    shared_prefix_len=12,
                    unique_suffix_len=4,
                    max_new_tokens=3,
                    num_groups=4,
                    prefix_block_size=4,
                    max_cache_blocks=32,
                ),
            },
            {
                "name": "low_reuse_control",
                "kwargs": dict(
                    workload_name="low_reuse",
                    num_requests=8,
                    prompt_len=16,
                    max_new_tokens=4,
                    prefix_block_size=4,
                    max_cache_blocks=32,
                ),
            },
            {
                "name": "eviction_pressure",
                "kwargs": dict(
                    workload_name="shared_prefix",
                    num_requests=16,
                    shared_prefix_len=16,
                    unique_suffix_len=4,
                    max_new_tokens=3,
                    num_groups=4,
                    prefix_block_size=4,
                    max_cache_blocks=6,
                ),
            },
        ]
    else:
        configs = [
            {
                "name": "shared_prefix_basic",
                "kwargs": dict(
                    workload_name="shared_prefix",
                    num_requests=8,
                    shared_prefix_len=16,
                    unique_suffix_len=8,
                    max_new_tokens=6,
                    num_groups=1,
                    prefix_block_size=4,
                    max_cache_blocks=64,
                ),
            },
            {
                "name": "high_reuse_many_requests",
                "kwargs": dict(
                    workload_name="shared_prefix",
                    num_requests=24,
                    shared_prefix_len=24,
                    unique_suffix_len=4,
                    max_new_tokens=4,
                    num_groups=1,
                    prefix_block_size=4,
                    max_cache_blocks=64,
                ),
            },
            {
                "name": "multi_prefix_groups",
                "kwargs": dict(
                    workload_name="shared_prefix",
                    num_requests=24,
                    shared_prefix_len=16,
                    unique_suffix_len=8,
                    max_new_tokens=4,
                    num_groups=4,
                    prefix_block_size=4,
                    max_cache_blocks=64,
                ),
            },
            {
                "name": "low_reuse_control",
                "kwargs": dict(
                    workload_name="low_reuse",
                    num_requests=8,
                    prompt_len=24,
                    max_new_tokens=6,
                    prefix_block_size=4,
                    max_cache_blocks=64,
                ),
            },
            {
                "name": "eviction_pressure",
                "kwargs": dict(
                    workload_name="shared_prefix",
                    num_requests=24,
                    shared_prefix_len=16,
                    unique_suffix_len=8,
                    max_new_tokens=4,
                    num_groups=6,
                    prefix_block_size=4,
                    max_cache_blocks=8,
                ),
            },
        ]

    results = {}

    for config in configs:
        name = config["name"]
        kwargs = config["kwargs"]

        print(f"\n\n=== {name} ===")
        print(kwargs)

        results[name] = run_no_prefix_vs_prefix_cache_benchmark(
            model,
            vocab_size=vocab_size,
            device=device,
            seed=1337,
            temperature=1.0,
            **kwargs,
        )

    return results


# Minimal direct example if you do not want to run the whole suite:
#
# run_no_prefix_vs_prefix_cache_benchmark(
#     m,
#     vocab_size=vocab_size,
#     workload_name="shared_prefix",
#     num_requests=8,
#     shared_prefix_len=12,
#     unique_suffix_len=4,
#     max_new_tokens=4,
#     prefix_block_size=4,
#     max_cache_blocks=32,
#     device=device,
# )
