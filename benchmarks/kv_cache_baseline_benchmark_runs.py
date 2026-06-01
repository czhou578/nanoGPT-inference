"""
Suggested baseline-vs-KV-cache benchmark runs.

Use this with the benchmark file that defines:

    run_baseline_vs_kv_benchmark(...)

How to use:
1. Put your benchmark file next to your nanoGPT script, for example:

       benchmark_baseline_vs_kv.py

2. Put this file next to your nanoGPT script too:

       kv_cache_baseline_benchmark_runs.py

3. Import `run_kv_cache_baseline_benchmark_suite` near the bottom of nanoGPT.

4. After training/loading your model, call:

       run_kv_cache_baseline_benchmark_suite(
           m,
           train_data=train_data,
           device=device,
           block_size=block_size,
       )

These runs compare:
- no-cache generation, which recomputes the context every token
- KV-cache generation, which prefills once then decodes one token at a time
"""

from benchmarks.kv_cache_baseline import run_baseline_vs_kv_benchmark


def _prompt_from_train_data(train_data, prompt_len):
    """
    Return a deterministic prompt from the start of train_data.

    `train_data` can be a torch tensor or a plain list.
    """
    prompt = train_data[:prompt_len]

    if hasattr(prompt, "tolist"):
        return prompt.tolist()

    return list(prompt)


def run_kv_cache_baseline_benchmark_suite(
    model,
    *,
    train_data,
    device,
    block_size,
):
    """
    Run several baseline-vs-KV-cache comparisons.

    The configs are split for block_size=32 vs block_size>=64 so you do not
    accidentally exceed your model's position embedding range.
    """

    print("\n=== KV Cache Baseline Benchmark Suite ===")
    print(f"block_size={block_size}, device={device}")

    if block_size <= 32:
        configs = [
            {
                "name": "small_smoke_test",
                "prompt_len": 8,
                "N": 16,
            },
            {
                "name": "longer_generation",
                "prompt_len": 8,
                "N": 24,
            },
            {
                "name": "heavier_prompt",
                "prompt_len": 16,
                "N": 16,
            },
            {
                "name": "near_context_limit",
                "prompt_len": 4,
                "N": 28,
            },
        ]
        sweep_prompt_len = 8
        sweep_values = [4, 8, 16, 24]
    else:
        configs = [
            {
                "name": "small_smoke_test",
                "prompt_len": 8,
                "N": 16,
            },
            {
                "name": "medium_generation",
                "prompt_len": 16,
                "N": 32,
            },
            {
                "name": "longer_generation",
                "prompt_len": 16,
                "N": 48,
            },
            {
                "name": "heavier_prompt",
                "prompt_len": 32,
                "N": 32,
            },
            {
                "name": "near_context_limit",
                "prompt_len": 8,
                "N": 56,
            },
        ]
        sweep_prompt_len = 8
        sweep_values = [8, 16, 32, 48, 56]

    results = {}

    for config in configs:
        name = config["name"]
        prompt_len = config["prompt_len"]
        N = config["N"]

        print(f"\n\n=== {name} ===")
        print({"prompt_len": prompt_len, "N": N})

        prompt_tokens = _prompt_from_train_data(train_data, prompt_len)

        results[name] = run_baseline_vs_kv_benchmark(
            model,
            prompt_tokens,
            N=N,
            device=device,
            block_size=block_size,
            seed=1337,
            temperature=1.0,
        )

    print("\n\n=== Generation Length Sweep ===")
    sweep_results = {}

    for N in sweep_values:
        print(f"\n\n=== N={N} ===")
        print({"prompt_len": sweep_prompt_len, "N": N})

        prompt_tokens = _prompt_from_train_data(train_data, sweep_prompt_len)

        sweep_results[N] = run_baseline_vs_kv_benchmark(
            model,
            prompt_tokens,
            N=N,
            device=device,
            block_size=block_size,
            seed=1337,
            temperature=1.0,
        )

    results["generation_length_sweep"] = sweep_results
    return results


# Minimal direct example if you do not want to run the whole suite:
#
# prompt_tokens = train_data[:8].tolist()
# run_baseline_vs_kv_benchmark(
#     m,
#     prompt_tokens,
#     N=24,
#     device=device,
#     block_size=block_size,
# )
