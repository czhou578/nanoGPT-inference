"""
Benchmark suite: Eager KV-cache vs CUDA Graph decode.

Runs several configurations comparing decode step latency
and end-to-end throughput with and without CUDA graph replay.

Import and call from nanogpt-cuda-graph.py:

    from benchmarks.cuda_graph_benchmark_runs import run_cuda_graph_benchmark_suite

    run_cuda_graph_benchmark_suite(
        m,
        train_data=train_data,
        clear_cache_fn=clear_kv_cache,
        device=device,
        block_size=block_size,
    )
"""

from benchmarks.cuda_graph import run_eager_vs_cuda_graph_benchmark


def _prompt_from_train_data(train_data, prompt_len):
    prompt = train_data[:prompt_len]
    if hasattr(prompt, "tolist"):
        return prompt.tolist()
    return list(prompt)


def run_cuda_graph_benchmark_suite(
    model,
    *,
    train_data,
    clear_cache_fn,
    device,
    block_size,
):
    """
    Run the full eager-vs-CUDA-graph benchmark suite.

    Configs:
      1. Smoke test         — short prompt, few tokens (sanity check)
      2. Medium generation   — typical decode workload
      3. Long generation     — measure sustained throughput
      4. Heavy prompt        — longer prefill, shorter decode
      5. Near context limit  — stress the pre-allocated cache
      6. Generation sweep    — vary N to show how graph overhead amortizes
    """

    print("\n" + "=" * 60)
    print("  CUDA Graph Benchmark Suite")
    print(f"  block_size={block_size}, device={device}")
    print("=" * 60)

    configs = [
        {
            "name": "smoke_test",
            "prompt_len": 4,
            "N": 8,
        },
        {
            "name": "medium_generation",
            "prompt_len": 8,
            "N": 32,
        },
        {
            "name": "long_generation",
            "prompt_len": 8,
            "N": 48,
        },
        {
            "name": "heavy_prompt",
            "prompt_len": 32,
            "N": 16,
        },
        {
            "name": "near_context_limit",
            "prompt_len": 4,
            "N": 56,
        },
    ]

    results = {}

    for config in configs:
        name = config["name"]
        prompt_len = config["prompt_len"]
        N = config["N"]

        # Safety: don't exceed context window
        if prompt_len + N > block_size:
            N = block_size - prompt_len

        print(f"\n\n{'─' * 50}")
        print(f"  {name}  (prompt_len={prompt_len}, N={N})")
        print(f"{'─' * 50}")

        prompt_tokens = _prompt_from_train_data(train_data, prompt_len)

        results[name] = run_eager_vs_cuda_graph_benchmark(
            model,
            prompt_tokens,
            N=N,
            clear_cache_fn=clear_cache_fn,
            block_size=block_size,
            device=device,
            seed=1337,
        )

    # ── Generation length sweep ──
    print(f"\n\n{'═' * 50}")
    print("  Generation Length Sweep (prompt_len=8)")
    print(f"{'═' * 50}")

    sweep_prompt_len = 8
    sweep_values = [8, 16, 32, 48, 56]
    sweep_results = {}

    for N in sweep_values:
        if sweep_prompt_len + N > block_size:
            N = block_size - sweep_prompt_len

        print(f"\n  N={N}")

        prompt_tokens = _prompt_from_train_data(train_data, sweep_prompt_len)

        sweep_results[N] = run_eager_vs_cuda_graph_benchmark(
            model,
            prompt_tokens,
            N=N,
            clear_cache_fn=clear_cache_fn,
            block_size=block_size,
            device=device,
            seed=1337,
        )

    results["generation_length_sweep"] = sweep_results

    # ── Summary ──
    print(f"\n\n{'═' * 50}")
    print("  Summary: Per-step decode latency (median, ms)")
    print(f"{'═' * 50}")
    print(f"  {'config':<25} {'eager':>10} {'graph':>10} {'speedup':>10}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 10} {'─' * 10}")

    for name, result in results.items():
        if name == "generation_length_sweep":
            continue
        eager = result["eager_kv"]
        graph = result["cuda_graph"]
        speedup = (
            eager.median_decode_step_ms / graph.median_decode_step_ms
            if graph.median_decode_step_ms > 0 else float("inf")
        )
        print(
            f"  {name:<25} "
            f"{eager.median_decode_step_ms:>9.3f} "
            f"{graph.median_decode_step_ms:>9.3f} "
            f"{speedup:>9.2f}x"
        )

    return results
