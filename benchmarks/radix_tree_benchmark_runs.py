"""
Radix tree benchmark runner.

Usage:
1. Import near the bottom of nanogpt-radix-tree-.py:

       from benchmarks.radix_tree_benchmark_runs import (
           run_radix_tree_benchmark_suite,
       )

2. After training, call:

       run_radix_tree_benchmark_suite(
           m,
           vocab_size=vocab_size,
           device=device,
           block_size=block_size,
       )

Compares: no_cache vs flat_cache vs radix_tree across workloads
that exercise shared prefixes, branching conversations, eviction
pressure, and a low-reuse control case.
"""

from benchmarks.radix_tree_caching import run_radix_vs_flat_benchmark


def run_radix_tree_benchmark_suite(
    model,
    *,
    vocab_size,
    device,
    block_size,
):
    """
    Run no_cache / flat_cache / radix_tree comparisons across multiple
    workload scenarios.
    """

    print("\n=== Radix Tree Benchmark Suite ===")
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
                    block_size=4,
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
                    block_size=4,
                    max_cache_blocks=32,
                ),
            },
            {
                "name": "branching_conversation",
                "kwargs": dict(
                    workload_name="branching",
                    trunk_len=16,
                    branch_suffix_len=4,
                    num_branches=6,
                    max_new_tokens=4,
                    block_size=4,
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
                    block_size=4,
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
                    block_size=4,
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
                    block_size=4,
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
                    block_size=4,
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
                    block_size=4,
                    max_cache_blocks=64,
                ),
            },
            {
                "name": "branching_conversation",
                "kwargs": dict(
                    workload_name="branching",
                    trunk_len=24,
                    branch_suffix_len=8,
                    num_branches=8,
                    max_new_tokens=4,
                    block_size=4,
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
                    block_size=4,
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
                    block_size=4,
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
                    block_size=4,
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

        results[name] = run_radix_vs_flat_benchmark(
            model,
            vocab_size=vocab_size,
            device=device,
            seed=1337,
            temperature=1.0,
            **kwargs,
        )

    return results
