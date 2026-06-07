"""
Visualization and reporting for request simulator results.

Provides:
  - print_simulation_timeline: step-by-step text log
  - print_simulation_summary: aggregate statistics
  - plot_simulation: multi-panel matplotlib figure
"""

from statistics import mean


def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


# ──────────────────────────────────────────────────────────────────────
# Text timeline
# ──────────────────────────────────────────────────────────────────────

def print_simulation_timeline(snapshots, max_rows=None):
    """
    Print a step-by-step timeline table.

    Events are encoded as:
      +R3      = request 3 arrived
      ✓R3      = request 3 completed
      ✗R3      = request 3 cancelled
      P=2      = 2 preemptions this step
    """
    headers = ["step", "pending", "waiting", "prefill", "active", "batch", "d_tok", "p_tok", "events"]

    rows = []
    for snap in snapshots:
        events = []
        for rid in snap.arrivals:
            events.append(f"+R{rid}")
        for rid in snap.completions:
            events.append(f"✓R{rid}")
        for rid in snap.cancellations:
            events.append(f"✗R{rid}")
        if snap.preemptions > 0:
            events.append(f"P={snap.preemptions}")

        rows.append([
            str(snap.step),
            str(snap.pending_count),
            str(snap.waiting_count),
            str(snap.prefilling_count),
            str(snap.active_count),
            str(snap.decode_batch_size),
            str(snap.decode_tokens),
            str(snap.prefill_tokens),
            " ".join(events) if events else "",
        ])

    if max_rows is not None and len(rows) > max_rows:
        # Show first max_rows/2 and last max_rows/2.
        half = max_rows // 2
        rows = rows[:half] + [["...", "...", "...", "...", "...", "...", "...", "...", "..."]] + rows[-half:]

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]

    def fmt(values):
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

def print_simulation_summary(result):
    """Print aggregate statistics for a simulation run."""
    print()
    print("Simulation Summary")
    print("=" * 50)
    print(f"  Backend:          {result.config.backend.value}")
    print(f"  Arrival pattern:  {result.config.arrival_pattern.value}")
    if result.config.backend.value == "scheduling_policy":
        print(f"  Policy:           {result.config.policy}")
    print(f"  Requests:         {result.total_requests} total, "
          f"{result.completed_requests} completed, "
          f"{result.cancelled_requests} cancelled")
    print(f"  Generated tokens: {result.total_generated_tokens}")
    print(f"  Wall time:        {result.total_seconds:.4f}s")
    print(f"  Throughput:       {result.tokens_per_second:.2f} tokens/sec")

    if result.request_latencies_s:
        print(f"  Avg latency:      {result.avg_latency_s * 1000:.2f} ms")
        print(f"  P95 latency:      {_percentile(result.request_latencies_s, 0.95) * 1000:.2f} ms")

    if result.ttft_s:
        print(f"  Avg TTFT:         {result.avg_ttft_s * 1000:.2f} ms")
        print(f"  P95 TTFT:         {_percentile(result.ttft_s, 0.95) * 1000:.2f} ms")

    if result.total_preemptions > 0:
        print(f"  Preemptions:      {result.total_preemptions}")

    if result.snapshots:
        batch_sizes = [s.decode_batch_size for s in result.snapshots if s.decode_batch_size > 0]
        if batch_sizes:
            print(f"  Avg batch size:   {mean(batch_sizes):.2f}")
            print(f"  Max batch size:   {max(batch_sizes)}")

    print("=" * 50)


# ──────────────────────────────────────────────────────────────────────
# Matplotlib plots
# ──────────────────────────────────────────────────────────────────────

def plot_simulation(snapshots, save_path=None, title=None):
    """
    Generate a multi-panel matplotlib figure showing simulation behavior.

    Panel 1: Queue depths over time (stacked area)
    Panel 2: Batch utilization (line chart)
    Panel 3: Cumulative throughput (line chart)
    Panel 4: Event timeline (scatter)

    Args:
        snapshots:  List of SimulationStepSnapshot from a simulation run.
        save_path:  If provided, save the figure to this path (PNG).
        title:      Optional figure title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot generation.")
        return

    steps = [s.step for s in snapshots]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Panel 1: Queue depths (stacked area) ---
    ax1 = axes[0]
    pending = [s.pending_count for s in snapshots]
    waiting = [s.waiting_count for s in snapshots]
    prefilling = [s.prefilling_count for s in snapshots]
    active = [s.active_count for s in snapshots]

    ax1.stackplot(
        steps, pending, waiting, prefilling, active,
        labels=["Pending", "Waiting", "Prefilling", "Active"],
        colors=["#e0e0e0", "#ffcc80", "#90caf9", "#66bb6a"],
        alpha=0.85,
    )
    ax1.set_ylabel("Request Count")
    ax1.set_title("Queue Depths Over Time")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # --- Panel 2: Batch utilization ---
    ax2 = axes[1]
    batch_sizes = [s.decode_batch_size for s in snapshots]
    ax2.plot(steps, batch_sizes, color="#1976d2", linewidth=1.5, label="Decode batch size")
    ax2.fill_between(steps, batch_sizes, alpha=0.2, color="#1976d2")
    ax2.set_ylabel("Batch Size")
    ax2.set_title("Decode Batch Utilization")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # --- Panel 3: Cumulative throughput ---
    ax3 = axes[2]
    cum_tokens = [s.total_generated_tokens for s in snapshots]
    wall_times = [s.wall_time_s for s in snapshots]
    throughput = []
    for ct, wt in zip(cum_tokens, wall_times):
        throughput.append(ct / wt if wt > 0 else 0)
    ax3.plot(steps, throughput, color="#388e3c", linewidth=1.5)
    ax3.set_ylabel("Tokens/sec")
    ax3.set_title("Cumulative Throughput")
    ax3.grid(axis="y", alpha=0.3)

    # --- Panel 4: Event timeline ---
    ax4 = axes[3]
    for snap in snapshots:
        for _ in snap.arrivals:
            ax4.scatter(snap.step, 0.8, color="#4caf50", marker="o", s=20, zorder=3)
        for _ in snap.completions:
            ax4.scatter(snap.step, 0.5, color="#1976d2", marker="s", s=20, zorder=3)
        for _ in snap.cancellations:
            ax4.scatter(snap.step, 0.2, color="#d32f2f", marker="x", s=30, zorder=3)

    ax4.set_yticks([0.2, 0.5, 0.8])
    ax4.set_yticklabels(["Cancelled", "Completed", "Arrived"])
    ax4.set_xlabel("Scheduler Step")
    ax4.set_title("Events")
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
