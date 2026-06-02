"""
Suggested paged-attention benchmark runs.

Use this with the benchmark file that defines:

    run_contiguous_vs_paged_attention_benchmark(...)

How to use:
1. Import this file near the bottom of nanogpt-paged-attention.py:

       from benchmarks.paged_attention_benchmark_runs import (
           run_paged_attention_benchmark_suite,
       )

2. After training/loading your model, call:

       run_paged_attention_benchmark_suite(
           m,
           vocab_size=vocab_size,
           device=device,
           block_size=block_size,
       )

These runs compare normal contiguous per-request KV caches against a paged KV
block pool with per-request block tables.
"""

from benchmarks.paged_attention import run_contiguous_vs_paged_attention_benchmark


def run_paged_attention_benchmark_suite(
    model,
    *,
    vocab_size,
    device,
    block_size,
):
    """
    Run contiguous-KV vs paged-KV comparisons.

    Workloads are sized so prompt_len + max_new_tokens stays below the model's
    position embedding block_size.
    """

    print("\n=== Paged Attention Benchmark Suite ===")
    print(f"block_size={block_size}, device={device}")

    if block_size <= 32:
        configs = [
            {
                "name": "uniform_short",
                "kwargs": dict(
                    workload_name="uniform",
                    num_requests=8,
                    prompt_len=8,
                    max_new_tokens=12,
                    arrival_gap=0,
                    max_batch_size=4,
                    page_block_size=4,
                    num_physical_blocks=64,
                ),
            },
            {
                "name": "mixed_lengths",
                "kwargs": dict(
                    workload_name="mixed_lengths",
                    num_requests=12,
                    prompt_lens=(4, 7, 8, 11),
                    output_lens=(4, 8, 12),
                    arrival_gap=0,
                    max_batch_size=4,
                    page_block_size=4,
                    num_physical_blocks=64,
                ),
            },
            {
                "name": "block_boundary_pressure",
                "kwargs": dict(
                    workload_name="block_boundary",
                    num_requests=10,
                    max_new_tokens=8,
                    max_batch_size=4,
                    page_block_size=4,
                    num_physical_blocks=64,
                ),
            },
            {
                "name": "limited_pool",
                "kwargs": dict(
                    workload_name="mixed_lengths",
                    num_requests=12,
                    prompt_lens=(4, 7, 8, 11),
                    output_lens=(4, 8, 12),
                    arrival_gap=0,
                    max_batch_size=3,
                    page_block_size=4,
                    num_physical_blocks=18,
                ),
            },
        ]
    else:
        configs = [
            {
                "name": "uniform_short",
                "kwargs": dict(
                    workload_name="uniform",
                    num_requests=12,
                    prompt_len=12,
                    max_new_tokens=16,
                    arrival_gap=0,
                    max_batch_size=4,
                    page_block_size=4,
                    num_physical_blocks=96,
                ),
            },
            {
                "name": "mixed_lengths",
                "kwargs": dict(
                    workload_name="mixed_lengths",
                    num_requests=16,
                    prompt_lens=(8, 12, 16, 20),
                    output_lens=(8, 12, 16),
                    arrival_gap=0,
                    max_batch_size=4,
                    page_block_size=4,
                    num_physical_blocks=128,
                ),
            },
            {
                "name": "block_boundary_pressure",
                "kwargs": dict(
                    workload_name="block_boundary",
                    num_requests=12,
                    max_new_tokens=12,
                    max_batch_size=4,
                    page_block_size=4,
                    num_physical_blocks=128,
                ),
            },
            {
                "name": "limited_pool",
                "kwargs": dict(
                    workload_name="mixed_lengths",
                    num_requests=16,
                    prompt_lens=(8, 12, 16, 20),
                    output_lens=(8, 12, 16),
                    arrival_gap=0,
                    max_batch_size=3,
                    page_block_size=4,
                    num_physical_blocks=28,
                ),
            },
        ]

    results = {}

    for config in configs:
        name = config["name"]
        kwargs = config["kwargs"]

        print(f"\n\n=== {name} ===")
        print(kwargs)

        results[name] = run_contiguous_vs_paged_attention_benchmark(
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
# run_contiguous_vs_paged_attention_benchmark(
#     m,
#     vocab_size=vocab_size,
#     workload_name="uniform",
#     num_requests=8,
#     prompt_len=8,
#     max_new_tokens=12,
#     max_batch_size=4,
#     page_block_size=4,
#     num_physical_blocks=64,
#     device=device,
# )
