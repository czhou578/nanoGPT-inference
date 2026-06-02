"""
Suggested decode/prefill interleaving benchmark runs.

Use this with the benchmark file that defines:

    run_separate_vs_interleaved_benchmark(...)

How to use:
1. Import this file near the bottom of nanogpt-interleaving.py:

       from benchmarks.interleaving_benchmark_runs import (
           run_interleaving_benchmark_suite,
       )

2. After training/loading your model, call:

       run_interleaving_benchmark_suite(
           m,
           vocab_size=vocab_size,
           device=device,
           block_size=block_size,
       )

These runs compare separate decode/prefill forward calls against a fused
interleaved forward pass under the same token budget.
"""

from benchmarks.interleaving import run_separate_vs_interleaved_benchmark


def run_interleaving_benchmark_suite(
    model,
    *,
    vocab_size,
    device,
    block_size,
):
    """
    Run separate-calls vs interleaved-fused comparisons.

    The configs keep prompt + generated length within block_size so position
    embeddings stay in range for the current NanoGPT model.
    """

    print("\n=== Decode/Prefill Interleaving Benchmark Suite ===")
    print(f"block_size={block_size}, device={device}")

    if block_size <= 32:
        configs = [
            {
                "name": "small_interleave_smoke",
                "kwargs": dict(
                    num_decode_heavy_requests=4,
                    decode_prompt_len=8,
                    decode_max_new_tokens=16,
                    num_prefill_heavy_requests=2,
                    prefill_prompt_len=20,
                    prefill_max_new_tokens=6,
                    prefill_arrival_step=2,
                    max_batch_size=4,
                    token_budget=12,
                    chunk_size=6,
                ),
            },
            {
                "name": "decode_prefill_overlap",
                "kwargs": dict(
                    num_decode_heavy_requests=5,
                    decode_prompt_len=8,
                    decode_max_new_tokens=18,
                    num_prefill_heavy_requests=3,
                    prefill_prompt_len=20,
                    prefill_max_new_tokens=6,
                    prefill_arrival_step=2,
                    max_batch_size=4,
                    token_budget=12,
                    chunk_size=6,
                ),
            },
            {
                "name": "smaller_chunks",
                "kwargs": dict(
                    num_decode_heavy_requests=5,
                    decode_prompt_len=8,
                    decode_max_new_tokens=18,
                    num_prefill_heavy_requests=3,
                    prefill_prompt_len=20,
                    prefill_max_new_tokens=6,
                    prefill_arrival_step=2,
                    max_batch_size=4,
                    token_budget=12,
                    chunk_size=4,
                ),
            },
        ]
    else:
        configs = [
            {
                "name": "small_interleave_smoke",
                "kwargs": dict(
                    num_decode_heavy_requests=4,
                    decode_prompt_len=8,
                    decode_max_new_tokens=24,
                    num_prefill_heavy_requests=2,
                    prefill_prompt_len=32,
                    prefill_max_new_tokens=8,
                    prefill_arrival_step=2,
                    max_batch_size=4,
                    token_budget=16,
                    chunk_size=8,
                ),
            },
            {
                "name": "decode_prefill_overlap",
                "kwargs": dict(
                    num_decode_heavy_requests=6,
                    decode_prompt_len=8,
                    decode_max_new_tokens=32,
                    num_prefill_heavy_requests=3,
                    prefill_prompt_len=32,
                    prefill_max_new_tokens=8,
                    prefill_arrival_step=2,
                    max_batch_size=4,
                    token_budget=16,
                    chunk_size=8,
                ),
            },
            {
                "name": "smaller_chunks_more_mixing",
                "kwargs": dict(
                    num_decode_heavy_requests=6,
                    decode_prompt_len=8,
                    decode_max_new_tokens=32,
                    num_prefill_heavy_requests=3,
                    prefill_prompt_len=32,
                    prefill_max_new_tokens=8,
                    prefill_arrival_step=2,
                    max_batch_size=4,
                    token_budget=16,
                    chunk_size=4,
                ),
            },
            {
                "name": "larger_chunks_less_overhead",
                "kwargs": dict(
                    num_decode_heavy_requests=6,
                    decode_prompt_len=8,
                    decode_max_new_tokens=32,
                    num_prefill_heavy_requests=3,
                    prefill_prompt_len=40,
                    prefill_max_new_tokens=8,
                    prefill_arrival_step=2,
                    max_batch_size=4,
                    token_budget=24,
                    chunk_size=12,
                ),
            },
            {
                "name": "staggered_prefill_arrivals",
                "kwargs": dict(
                    num_decode_heavy_requests=8,
                    decode_prompt_len=8,
                    decode_max_new_tokens=40,
                    num_prefill_heavy_requests=4,
                    prefill_prompt_len=32,
                    prefill_max_new_tokens=8,
                    prefill_arrival_step=3,
                    stagger_prefill_arrivals=True,
                    max_batch_size=4,
                    token_budget=16,
                    chunk_size=8,
                ),
            },
        ]

    results = {}

    for config in configs:
        name = config["name"]
        kwargs = config["kwargs"]

        print(f"\n\n=== {name} ===")
        print(kwargs)

        results[name] = run_separate_vs_interleaved_benchmark(
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
# run_separate_vs_interleaved_benchmark(
#     m,
#     vocab_size=vocab_size,
#     num_decode_heavy_requests=6,
#     decode_prompt_len=8,
#     decode_max_new_tokens=32,
#     num_prefill_heavy_requests=3,
#     prefill_prompt_len=32,
#     prefill_max_new_tokens=8,
#     prefill_arrival_step=2,
#     max_batch_size=4,
#     token_budget=16,
#     chunk_size=8,
#     device=device,
# )
