"""
Suggested chunked-prefill benchmark runs.

Use this with your real benchmark file, which defines:

    run_normal_vs_chunked_prefill_benchmark(...)

How to use:
1. Put your benchmark file next to your nanoGPT script, for example:

       benchmark_normal_vs_chunked_prefill.py

2. Import `run_chunked_prefill_benchmark_suite` near the bottom of nanoGPT.

3. After training/loading your model, call:

       run_chunked_prefill_benchmark_suite(
           m,
           vocab_size=vocab_size,
           device=device,
           block_size=block_size,
       )

These runs assume CPU and a small nanoGPT. They are designed to show:
- decode starvation under normal full-prompt prefill
- smoother streaming with chunked prefill
- the tradeoff between chunk size, token budget, TTFT, and inter-token gaps
"""

from benchmarks.normal_chunked_prefill import (
    run_normal_vs_chunked_prefill_benchmark,
)

def run_chunked_prefill_benchmark_suite(
    model,
    *,
    vocab_size,
    device,
    block_size,
):
    """
    Run several normal-prefill vs chunked-prefill comparisons.

    The configs are split for block_size=32 vs block_size>=64 so you do not
    accidentally exceed your model's position embedding range.
    """

    print("\n=== Chunked Prefill Benchmark Suite ===")
    print(f"block_size={block_size}, device={device}")

    if block_size <= 32:
        configs = [
            {
                "name": "small_smoke_test",
                "kwargs": dict(
                    num_short_requests=4,
                    short_prompt_len=8,
                    short_max_new_tokens=16,
                    num_long_requests=2,
                    long_prompt_len=24,
                    long_max_new_tokens=8,
                    long_arrival_step=2,
                    token_budget=16,
                    chunk_size=8,
                ),
            },
            {
                "name": "more_long_prompts",
                "kwargs": dict(
                    num_short_requests=4,
                    short_prompt_len=8,
                    short_max_new_tokens=16,
                    num_long_requests=4,
                    long_prompt_len=24,
                    long_max_new_tokens=8,
                    long_arrival_step=2,
                    token_budget=16,
                    chunk_size=8,
                ),
            },
            {
                "name": "smaller_chunks_smoother_decode",
                "kwargs": dict(
                    num_short_requests=4,
                    short_prompt_len=8,
                    short_max_new_tokens=16,
                    num_long_requests=4,
                    long_prompt_len=24,
                    long_max_new_tokens=8,
                    long_arrival_step=2,
                    token_budget=16,
                    chunk_size=4,
                ),
            },
            {
                "name": "larger_budget_larger_chunks",
                "kwargs": dict(
                    num_short_requests=4,
                    short_prompt_len=8,
                    short_max_new_tokens=16,
                    num_long_requests=4,
                    long_prompt_len=24,
                    long_max_new_tokens=8,
                    long_arrival_step=2,
                    token_budget=24,
                    chunk_size=12,
                ),
            },
        ]
    else:
        configs = [
            {
                "name": "small_smoke_test",
                "kwargs": dict(
                    num_short_requests=4,
                    short_prompt_len=8,
                    short_max_new_tokens=24,
                    num_long_requests=2,
                    long_prompt_len=32,
                    long_max_new_tokens=8,
                    long_arrival_step=2,
                    token_budget=16,
                    chunk_size=8,
                ),
            },
            {
                "name": "more_long_prompts",
                "kwargs": dict(
                    num_short_requests=4,
                    short_prompt_len=8,
                    short_max_new_tokens=32,
                    num_long_requests=4,
                    long_prompt_len=32,
                    long_max_new_tokens=8,
                    long_arrival_step=2,
                    token_budget=16,
                    chunk_size=8,
                ),
            },
            {
                "name": "smaller_chunks_smoother_decode",
                "kwargs": dict(
                    num_short_requests=4,
                    short_prompt_len=8,
                    short_max_new_tokens=32,
                    num_long_requests=4,
                    long_prompt_len=32,
                    long_max_new_tokens=8,
                    long_arrival_step=2,
                    token_budget=16,
                    chunk_size=4,
                ),
            },
            {
                "name": "larger_chunks_less_overhead",
                "kwargs": dict(
                    num_short_requests=4,
                    short_prompt_len=8,
                    short_max_new_tokens=32,
                    num_long_requests=4,
                    long_prompt_len=32,
                    long_max_new_tokens=8,
                    long_arrival_step=2,
                    token_budget=32,
                    chunk_size=16,
                ),
            },
            {
                "name": "decode_heavy_pressure",
                "kwargs": dict(
                    num_short_requests=8,
                    short_prompt_len=8,
                    short_max_new_tokens=48,
                    num_long_requests=4,
                    long_prompt_len=32,
                    long_max_new_tokens=8,
                    long_arrival_step=4,
                    token_budget=16,
                    chunk_size=8,
                ),
            },
            {
                "name": "late_long_prompt_interruptions",
                "kwargs": dict(
                    num_short_requests=8,
                    short_prompt_len=8,
                    short_max_new_tokens=48,
                    num_long_requests=4,
                    long_prompt_len=40,
                    long_max_new_tokens=8,
                    long_arrival_step=8,
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

        results[name] = run_normal_vs_chunked_prefill_benchmark(
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
# run_normal_vs_chunked_prefill_benchmark(
#     m,
#     vocab_size=vocab_size,
#     num_short_requests=4,
#     short_prompt_len=8,
#     short_max_new_tokens=24,
#     num_long_requests=2,
#     long_prompt_len=32,
#     long_max_new_tokens=8,
#     long_arrival_step=2,
#     token_budget=16,
#     chunk_size=8,
#     device=device,
# )
