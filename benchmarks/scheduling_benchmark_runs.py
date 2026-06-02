"""
Suggested scheduling benchmark runs.

Use this with the benchmark file that defines:

    run_fcfs_vs_priority_scheduling_benchmark(...)

How to use:
1. Import this file near the bottom of nanogpt-scheduling.py:

       from benchmarks.scheduling_benchmark_runs import (
           run_scheduling_benchmark_suite,
       )

2. After training/loading your model, call:

       run_scheduling_benchmark_suite(
           m,
           vocab_size=vocab_size,
           device=device,
           block_size=block_size,
       )

These runs compare FCFS scheduling against priority scheduling. They are
designed to show:
- priority inversion, where a short high-priority request arrives behind long
  low-priority work
- memory pressure, where priority scheduling may preempt lower-priority active
  requests and recompute them later
- a control case where all priorities are equal
"""

from benchmarks.scheduling_policy import (
    run_fcfs_vs_priority_scheduling_benchmark,
)


def run_scheduling_benchmark_suite(
    model,
    *,
    vocab_size,
    device,
    block_size,
):
    """
    Run FCFS vs priority scheduling comparisons.

    The configs keep prompt + generated length within block_size so position
    embeddings stay in range for the current NanoGPT model.
    """

    print("\n=== Scheduling Benchmark Suite ===")
    print(f"block_size={block_size}, device={device}")

    if block_size <= 32:
        configs = [
            {
                "name": "priority_inversion_serial",
                "kwargs": dict(
                    workload_name="priority_inversion",
                    max_batch_size=1,
                    token_budget=8,
                    prefill_chunk_size=8,
                    max_kv_tokens=32,
                ),
            },
            {
                "name": "priority_mix_small_batch",
                "kwargs": dict(
                    workload_name="priority_mix",
                    max_batch_size=2,
                    token_budget=8,
                    prefill_chunk_size=6,
                    max_kv_tokens=36,
                ),
            },
            {
                "name": "memory_pressure_preemption",
                "kwargs": dict(
                    workload_name="memory_pressure",
                    max_batch_size=2,
                    token_budget=8,
                    prefill_chunk_size=6,
                    max_kv_tokens=24,
                ),
            },
            {
                "name": "equal_priority_control",
                "kwargs": dict(
                    workload_name="equal_priority_control",
                    max_batch_size=2,
                    token_budget=8,
                    prefill_chunk_size=6,
                    max_kv_tokens=36,
                ),
            },
        ]
    else:
        configs = [
            {
                "name": "priority_inversion_serial",
                "kwargs": dict(
                    workload_name="priority_inversion",
                    max_batch_size=1,
                    token_budget=12,
                    prefill_chunk_size=10,
                    max_kv_tokens=40,
                ),
            },
            {
                "name": "priority_mix_small_batch",
                "kwargs": dict(
                    workload_name="priority_mix",
                    max_batch_size=4,
                    token_budget=16,
                    prefill_chunk_size=8,
                    max_kv_tokens=64,
                ),
            },
            {
                "name": "memory_pressure_preemption",
                "kwargs": dict(
                    workload_name="memory_pressure",
                    max_batch_size=3,
                    token_budget=12,
                    prefill_chunk_size=8,
                    max_kv_tokens=32,
                ),
            },
            {
                "name": "equal_priority_control",
                "kwargs": dict(
                    workload_name="equal_priority_control",
                    max_batch_size=4,
                    token_budget=16,
                    prefill_chunk_size=8,
                    max_kv_tokens=64,
                ),
            },
        ]

    results = {}
    for config in configs:
        name = config["name"]
        kwargs = config["kwargs"]

        print(f"\n\n=== {name} ===")
        print(kwargs)

        results[name] = run_fcfs_vs_priority_scheduling_benchmark(
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
# run_fcfs_vs_priority_scheduling_benchmark(
#     m,
#     vocab_size=vocab_size,
#     workload_name="priority_inversion",
#     max_batch_size=1,
#     token_budget=8,
#     prefill_chunk_size=8,
#     max_kv_tokens=32,
#     device=device,
# )
