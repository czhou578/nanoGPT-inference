"""
Suggested continuous-batching benchmark runs.

Use this with the benchmark file that defines:

    run_single_vs_continuous_batching_benchmark(...)

How to use:
1. Put your benchmark file next to your nanoGPT script, for example:

       benchmark_single_vs_continuous_batching.py

2. Put this file next to your nanoGPT script too:

       continuous_batching_benchmark_runs.py

3. Import `run_continuous_batching_benchmark_suite` near the bottom of nanoGPT.

4. After training/loading your model, call:

       run_continuous_batching_benchmark_suite(
           m,
           vocab_size=vocab_size,
           device=device,
           block_size=block_size,
       )

These runs assume CPU and a small nanoGPT. They are designed to show:
- throughput changes as max_batch_size increases
- effect of longer generations
- effect of staggered arrivals
- effect of heavier prompt lengths
"""

from benchmarks.single_req_cont_batching import (
    run_single_vs_continuous_batching_benchmark,
)


def run_continuous_batching_benchmark_suite(
    model,
    *,
    vocab_size,
    device,
    block_size,
):
    """
    Run several single-request vs continuous-batching comparisons.

    The configs are split for block_size=32 vs block_size>=64 so you do not
    accidentally exceed your model's position embedding range.
    """

    print("\n=== Continuous Batching Benchmark Suite ===")
    print(f"block_size={block_size}, device={device}")

    if block_size <= 32:
        configs = [
            {
                "name": "small_smoke_test",
                "kwargs": dict(
                    num_requests=8,
                    prompt_len=8,
                    max_new_tokens=16,
                    max_batch_size=4,
                    arrival_gap=0,
                ),
            },
            {
                "name": "more_requests_small_batch",
                "kwargs": dict(
                    num_requests=16,
                    prompt_len=8,
                    max_new_tokens=24,
                    max_batch_size=4,
                    arrival_gap=0,
                ),
            },
            {
                "name": "more_requests_larger_batch",
                "kwargs": dict(
                    num_requests=16,
                    prompt_len=8,
                    max_new_tokens=24,
                    max_batch_size=8,
                    arrival_gap=0,
                ),
            },
            {
                "name": "stress_batch_capacity",
                "kwargs": dict(
                    num_requests=32,
                    prompt_len=8,
                    max_new_tokens=24,
                    max_batch_size=8,
                    arrival_gap=0,
                ),
            },
            {
                "name": "staggered_arrivals",
                "kwargs": dict(
                    num_requests=16,
                    prompt_len=8,
                    max_new_tokens=24,
                    max_batch_size=8,
                    arrival_gap=1,
                ),
            },
        ]
    else:
        configs = [
            {
                "name": "small_smoke_test",
                "kwargs": dict(
                    num_requests=8,
                    prompt_len=8,
                    max_new_tokens=16,
                    max_batch_size=4,
                    arrival_gap=0,
                ),
            },
            {
                "name": "more_requests_small_batch",
                "kwargs": dict(
                    num_requests=16,
                    prompt_len=8,
                    max_new_tokens=24,
                    max_batch_size=4,
                    arrival_gap=0,
                ),
            },
            {
                "name": "more_requests_larger_batch",
                "kwargs": dict(
                    num_requests=16,
                    prompt_len=8,
                    max_new_tokens=24,
                    max_batch_size=8,
                    arrival_gap=0,
                ),
            },
            {
                "name": "longer_generations",
                "kwargs": dict(
                    num_requests=16,
                    prompt_len=8,
                    max_new_tokens=48,
                    max_batch_size=8,
                    arrival_gap=0,
                ),
            },
            # {
            #     "name": "staggered_arrivals",
            #     "kwargs": dict(
            #         num_requests=16,
            #         prompt_len=8,
            #         max_new_tokens=32,
            #         max_batch_size=8,
            #         arrival_gap=1,
            #     ),
            # },
            {
                "name": "heavier_prompt",
                "kwargs": dict(
                    num_requests=16,
                    prompt_len=16,
                    max_new_tokens=32,
                    max_batch_size=8,
                    arrival_gap=0,
                ),
            },
            {
                "name": "stress_batch_capacity",
                "kwargs": dict(
                    num_requests=32,
                    prompt_len=8,
                    max_new_tokens=32,
                    max_batch_size=8,
                    arrival_gap=0,
                ),
            },
        ]

    results = {}

    for config in configs:
        name = config["name"]
        kwargs = config["kwargs"]

        print(f"\n\n=== {name} ===")
        print(kwargs)

        results[name] = run_single_vs_continuous_batching_benchmark(
            model,
            vocab_size=vocab_size,
            device=device,
            seed=1337,
            temperature=1.0,
            **kwargs,
        )

    print("\n\n=== Batch Size Sweep ===")
    sweep_results = {}

    if block_size <= 32:
        sweep_max_new_tokens = 24
    else:
        sweep_max_new_tokens = 32

    for max_batch_size in [1, 2, 4, 8, 16]:
        print(f"\n\n=== max_batch_size={max_batch_size} ===")

        sweep_results[max_batch_size] = run_single_vs_continuous_batching_benchmark(
            model,
            vocab_size=vocab_size,
            num_requests=32,
            prompt_len=8,
            max_new_tokens=sweep_max_new_tokens,
            max_batch_size=max_batch_size,
            arrival_gap=0,
            device=device,
            seed=1337,
            temperature=1.0,
        )

    results["batch_size_sweep"] = sweep_results
    return results


# Minimal direct example if you do not want to run the whole suite:
#
# run_single_vs_continuous_batching_benchmark(
#     m,
#     vocab_size=vocab_size,
#     num_requests=16,
#     prompt_len=8,
#     max_new_tokens=24,
#     max_batch_size=8,
#     arrival_gap=0,
#     device=device,
# )
