"""
Suggested trigram speculative-decoding benchmark runs.

Use this with the benchmark file that defines:

    run_kv_vs_trigram_speculative_decoding_benchmark(...)

How to use:
1. Import this file near the bottom of nanogpt-trigram-spec-decode.py:

       from benchmarks.trigram_speculative_decoding_benchmark_runs import (
           run_trigram_speculative_decoding_benchmark_suite,
       )

2. After training/loading your model, call:

       run_trigram_speculative_decoding_benchmark_suite(
           m,
           vocab_size=vocab_size,
           device=device,
           block_size=block_size,
           training_token_ids=train_data,
           prompt_source_tokens=val_data,
       )

These runs compare normal KV-cache decoding against speculative decoding with
a trigram draft model at different draft depths and draft quality levels.
"""

from benchmarks.trigram_speculative_decoding import (
    run_kv_vs_trigram_speculative_decoding_benchmark,
)


def run_trigram_speculative_decoding_benchmark_suite(
    model,
    *,
    vocab_size,
    device,
    block_size,
    training_token_ids=None,
    prompt_source_tokens=None,
):
    """
    Run KV-cache vs trigram speculative-decoding comparisons.

    The configs keep prompt + generated length within block_size so position
    embeddings stay in range for the current NanoGPT model.
    """

    print("\n=== Trigram Speculative Decoding Benchmark Suite ===")
    print(f"block_size={block_size}, device={device}")

    if block_size <= 32:
        configs = [
            {
                "name": "k2_trigram_short_outputs",
                "kwargs": dict(
                    num_requests=6,
                    prompt_len=12,
                    max_new_tokens=8,
                    speculation_len=2,
                    draft_noise=0.0,
                ),
            },
            {
                "name": "k4_trigram_short_outputs",
                "kwargs": dict(
                    num_requests=6,
                    prompt_len=12,
                    max_new_tokens=8,
                    speculation_len=4,
                    draft_noise=0.0,
                ),
            },
            {
                "name": "k4_noisy_trigram",
                "kwargs": dict(
                    num_requests=6,
                    prompt_len=12,
                    max_new_tokens=8,
                    speculation_len=4,
                    draft_noise=0.50,
                ),
            },
            {
                "name": "longer_outputs_k4_trigram",
                "kwargs": dict(
                    num_requests=4,
                    prompt_len=16,
                    max_new_tokens=10,
                    speculation_len=4,
                    draft_noise=0.0,
                ),
            },
        ]
    else:
        configs = [
            {
                "name": "k2_trigram_draft",
                "kwargs": dict(
                    num_requests=8,
                    prompt_len=24,
                    max_new_tokens=16,
                    speculation_len=2,
                    draft_noise=0.0,
                ),
            },
            {
                "name": "k4_trigram_draft",
                "kwargs": dict(
                    num_requests=8,
                    prompt_len=24,
                    max_new_tokens=16,
                    speculation_len=4,
                    draft_noise=0.0,
                ),
            },
            {
                "name": "k6_trigram_draft",
                "kwargs": dict(
                    num_requests=8,
                    prompt_len=24,
                    max_new_tokens=16,
                    speculation_len=6,
                    draft_noise=0.0,
                ),
            },
            {
                "name": "k4_noisy_trigram",
                "kwargs": dict(
                    num_requests=8,
                    prompt_len=24,
                    max_new_tokens=16,
                    speculation_len=4,
                    draft_noise=0.50,
                ),
            },
            {
                "name": "longer_outputs_k4_trigram",
                "kwargs": dict(
                    num_requests=6,
                    prompt_len=24,
                    max_new_tokens=24,
                    speculation_len=4,
                    draft_noise=0.0,
                ),
            },
        ]

    results = {}

    for config in configs:
        name = config["name"]
        kwargs = config["kwargs"]

        print(f"\n\n=== {name} ===")
        print(kwargs)

        results[name] = run_kv_vs_trigram_speculative_decoding_benchmark(
            model,
            vocab_size=vocab_size,
            training_token_ids=training_token_ids,
            prompt_source_tokens=prompt_source_tokens,
            device=device,
            seed=1337,
            temperature=1.0,
            **kwargs,
        )

    return results


# Minimal direct example if you do not want to run the whole suite:
#
# run_kv_vs_trigram_speculative_decoding_benchmark(
#     m,
#     vocab_size=vocab_size,
#     training_token_ids=train_data,
#     prompt_source_tokens=val_data,
#     num_requests=8,
#     prompt_len=24,
#     max_new_tokens=16,
#     speculation_len=4,
#     draft_noise=0.0,
#     device=device,
# )
