"""
Pre-built simulation scenarios for the NanoGPT request simulator.

Each scenario demonstrates a different traffic pattern and scheduler behavior.
Run from the main NanoGPT script after training:

    from benchmarks.simulation_benchmark_runs import run_simulation_benchmarks
    run_simulation_benchmarks(model, vocab_size=vocab_size, device=device)
"""

from benchmarks.request_simulator import (
    SimulatorConfig,
    ArrivalPattern,
    SchedulerBackend,
    run_simulation,
)
from benchmarks.simulation_plots import (
    print_simulation_timeline,
    print_simulation_summary,
    plot_simulation,
)


SCENARIOS = {
    "steady_stream": SimulatorConfig(
        num_requests=16,
        arrival_pattern=ArrivalPattern.POISSON,
        arrival_rate=2.0,
        prompt_len_range=(4, 12),
        max_new_tokens_range=(6, 16),
        backend=SchedulerBackend.SCHEDULING_POLICY,
        policy="fcfs",
        max_batch_size=4,
        token_budget=16,
        prefill_chunk_size=8,
        max_kv_tokens=48,
    ),
    "burst_traffic": SimulatorConfig(
        num_requests=16,
        arrival_pattern=ArrivalPattern.BURSTY,
        burst_size=4,
        burst_gap=6,
        prompt_len_range=(4, 12),
        max_new_tokens_range=(6, 16),
        backend=SchedulerBackend.SCHEDULING_POLICY,
        policy="fcfs",
        max_batch_size=4,
        token_budget=16,
        prefill_chunk_size=8,
        max_kv_tokens=48,
    ),
    "priority_under_load": SimulatorConfig(
        num_requests=16,
        arrival_pattern=ArrivalPattern.POISSON,
        arrival_rate=3.0,
        prompt_len_range=(4, 10),
        max_new_tokens_range=(6, 14),
        priority_weights={0: 0.2, 5: 0.6, 9: 0.2},
        backend=SchedulerBackend.SCHEDULING_POLICY,
        policy="priority",
        max_batch_size=4,
        token_budget=16,
        prefill_chunk_size=8,
        max_kv_tokens=48,
    ),
    "cancellation_chaos": SimulatorConfig(
        num_requests=16,
        arrival_pattern=ArrivalPattern.POISSON,
        arrival_rate=2.0,
        prompt_len_range=(4, 12),
        max_new_tokens_range=(6, 16),
        cancellation_rate=0.08,
        backend=SchedulerBackend.SCHEDULING_POLICY,
        policy="fcfs",
        max_batch_size=4,
        token_budget=16,
        prefill_chunk_size=8,
        max_kv_tokens=48,
    ),
    "interleaved_steady": SimulatorConfig(
        num_requests=12,
        arrival_pattern=ArrivalPattern.POISSON,
        arrival_rate=2.0,
        prompt_len_range=(4, 12),
        max_new_tokens_range=(6, 16),
        backend=SchedulerBackend.INTERLEAVING,
        max_batch_size=4,
        token_budget=16,
        prefill_chunk_size=8,
    ),
    "interleaved_bursty": SimulatorConfig(
        num_requests=12,
        arrival_pattern=ArrivalPattern.BURSTY,
        burst_size=4,
        burst_gap=5,
        prompt_len_range=(4, 12),
        max_new_tokens_range=(6, 16),
        backend=SchedulerBackend.INTERLEAVING,
        max_batch_size=4,
        token_budget=16,
        prefill_chunk_size=8,
    ),
}


def run_simulation_benchmarks(
    model,
    *,
    vocab_size,
    device,
    seed=1337,
    scenarios=None,
    save_plots=True,
    max_timeline_rows=40,
):
    """
    Run all (or selected) simulation scenarios and print results.

    Args:
        model:              Trained NanoGPT model.
        vocab_size:         Vocabulary size.
        device:             torch device.
        seed:               Random seed.
        scenarios:          List of scenario names to run (default: all).
        save_plots:         Whether to save matplotlib plots as PNGs.
        max_timeline_rows:  Max rows to show in the text timeline (None = all).
    """
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    print("\n" + "=" * 60)
    print("  Request Simulator Benchmarks")
    print("=" * 60)

    results = {}
    for name in scenarios:
        if name not in SCENARIOS:
            print(f"  Unknown scenario: {name}")
            continue

        config = SCENARIOS[name]
        print(f"\n{'─' * 60}")
        print(f"  Scenario: {name}")
        print(f"  Backend: {config.backend.value} | Pattern: {config.arrival_pattern.value}")
        print(f"{'─' * 60}")

        result = run_simulation(model, config, vocab_size, device, seed=seed)
        results[name] = result

        print_simulation_timeline(result.snapshots, max_rows=max_timeline_rows)
        print_simulation_summary(result)

        if save_plots:
            plot_path = f"simulation_{name}.png"
            plot_simulation(
                result.snapshots,
                save_path=plot_path,
                title=f"Simulation: {name}",
            )

    print("\n" + "=" * 60)
    print(f"  {len(results)} simulation(s) completed.")
    print("=" * 60)

    return results
