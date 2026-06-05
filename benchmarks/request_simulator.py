"""
Request simulator for the NanoGPT inference engine.

Generates realistic, staggered workloads with configurable arrival patterns
(Poisson, uniform, bursty), variable prompt/generation lengths, priorities,
and mid-flight cancellation. Feeds them into the existing schedulers and
captures per-step telemetry snapshots.

Supports two scheduler backends:
  - scheduling_policy.py (FCFS/priority with preemption)
  - interleaving.py (fused decode-prefill batching)

Usage:
    from benchmarks.request_simulator import SimulatorConfig, ArrivalPattern, run_simulation
    config = SimulatorConfig(num_requests=20, arrival_pattern=ArrivalPattern.POISSON)
    result = run_simulation(model, config, vocab_size=65, device="cpu")
"""

from dataclasses import dataclass, field
from enum import Enum
import heapq
import math
import random
import time
import torch
import torch.nn.functional as F

from benchmarks.scheduling_policy import (
    SchedulingRequestSpec,
    SchedulingRequestState,
    SchedulingStepMetrics,
    _sort_key,
    _admit_waiting,
    _prefill_chunk,
    _decode_batch,
    _sync_if_cuda,
)
from benchmarks.interleaving import (
    InterleaveRequestSpec,
    InterleaveRequestState,
    InterleaveStepMetrics,
    _arrive_requests as _interleave_arrive,
    _prefill_chunk as _interleave_prefill_chunk,
    _decode_batch as _interleave_decode_batch,
    _run_fused_step,
    _make_generator,
    _sync_if_cuda as _interleave_sync,
)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

class ArrivalPattern(Enum):
    POISSON = "poisson"
    UNIFORM = "uniform"
    BURSTY = "bursty"


class SchedulerBackend(Enum):
    SCHEDULING_POLICY = "scheduling_policy"
    INTERLEAVING = "interleaving"


@dataclass
class SimulatorConfig:
    num_requests: int = 20
    arrival_pattern: ArrivalPattern = ArrivalPattern.POISSON
    arrival_rate: float = 2.0           # requests per step (Poisson λ)
    arrival_gap: int = 1                # steps between arrivals (uniform)
    burst_size: int = 4                 # requests per burst (bursty)
    burst_gap: int = 5                  # steps between bursts (bursty)

    prompt_len_range: tuple[int, int] = (4, 16)
    max_new_tokens_range: tuple[int, int] = (8, 32)
    priority_weights: dict[int, float] | None = None  # e.g. {0: 0.3, 5: 0.5, 9: 0.2}
    cancellation_rate: float = 0.0      # probability of cancelling one active request per step

    # Scheduler backend
    backend: SchedulerBackend = SchedulerBackend.SCHEDULING_POLICY

    # Scheduler knobs (passed through to the backend)
    policy: str = "fcfs"                # "fcfs" or "priority" (scheduling_policy backend)
    max_batch_size: int = 4
    token_budget: int = 16
    prefill_chunk_size: int = 8
    max_kv_tokens: int = 64
    temperature: float = 1.0
    max_steps: int = 10000


# ──────────────────────────────────────────────────────────────────────
# Telemetry
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SimulationStepSnapshot:
    step: int
    wall_time_s: float

    # Queue state
    pending_count: int
    waiting_count: int
    prefilling_count: int
    active_count: int

    # Work done this step
    prefill_tokens: int
    decode_tokens: int
    decode_batch_size: int

    # Events
    arrivals: list[int] = field(default_factory=list)
    completions: list[int] = field(default_factory=list)
    cancellations: list[int] = field(default_factory=list)
    preemptions: int = 0

    # Cumulative
    total_completed: int = 0
    total_generated_tokens: int = 0


@dataclass
class SimulationResult:
    config: SimulatorConfig
    snapshots: list[SimulationStepSnapshot]
    total_requests: int
    completed_requests: int
    cancelled_requests: int
    total_generated_tokens: int
    total_seconds: float
    request_latencies_s: list[float]
    ttft_s: list[float]
    total_preemptions: int

    @property
    def tokens_per_second(self) -> float:
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_generated_tokens / self.total_seconds

    @property
    def avg_latency_s(self) -> float:
        return sum(self.request_latencies_s) / len(self.request_latencies_s) if self.request_latencies_s else 0.0

    @property
    def avg_ttft_s(self) -> float:
        return sum(self.ttft_s) / len(self.ttft_s) if self.ttft_s else 0.0


# ──────────────────────────────────────────────────────────────────────
# Workload generation
# ──────────────────────────────────────────────────────────────────────

def _generate_arrival_steps(config: SimulatorConfig, rng: random.Random) -> list[int]:
    """Generate arrival_step for each request based on the arrival pattern."""
    n = config.num_requests
    steps = []

    if config.arrival_pattern == ArrivalPattern.UNIFORM:
        for i in range(n):
            steps.append(i * config.arrival_gap)

    elif config.arrival_pattern == ArrivalPattern.POISSON:
        # Poisson process: exponential inter-arrival times.
        # arrival_rate = expected requests per step → inter-arrival = 1/rate.
        current_step = 0.0
        for _ in range(n):
            steps.append(int(current_step))
            inter_arrival = rng.expovariate(config.arrival_rate)
            current_step += inter_arrival

    elif config.arrival_pattern == ArrivalPattern.BURSTY:
        burst_step = 0
        remaining = n
        while remaining > 0:
            count = min(config.burst_size, remaining)
            for _ in range(count):
                steps.append(burst_step)
            remaining -= count
            burst_step += config.burst_gap

    return steps


def _sample_priority(config: SimulatorConfig, rng: random.Random) -> int:
    """Draw a priority value from the configured weights."""
    if config.priority_weights is None:
        return 0
    priorities = list(config.priority_weights.keys())
    weights = list(config.priority_weights.values())
    return rng.choices(priorities, weights=weights, k=1)[0]


def generate_workload(
    config: SimulatorConfig,
    vocab_size: int,
    seed: int = 1337,
) -> list[SchedulingRequestSpec] | list[InterleaveRequestSpec]:
    """
    Generate a workload of request specs with realistic arrival patterns.

    Returns SchedulingRequestSpec for scheduling_policy backend,
    or InterleaveRequestSpec for interleaving backend.
    """
    rng = random.Random(seed)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(seed)

    arrival_steps = _generate_arrival_steps(config, rng)

    specs = []
    for i in range(config.num_requests):
        prompt_len = rng.randint(*config.prompt_len_range)
        max_new_tokens = rng.randint(*config.max_new_tokens_range)
        priority = _sample_priority(config, rng)
        prompt_tokens = torch.randint(
            0, vocab_size, (prompt_len,), generator=torch_rng
        ).tolist()

        if config.backend == SchedulerBackend.SCHEDULING_POLICY:
            specs.append(SchedulingRequestSpec(
                id=i,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                arrival_step=arrival_steps[i],
                priority=priority,
                group=f"priority_{priority}",
            ))
        else:
            specs.append(InterleaveRequestSpec(
                id=i,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                arrival_step=arrival_steps[i],
                group=f"priority_{priority}",
            ))

    return sorted(specs, key=lambda s: (s.arrival_step, s.id))


# ──────────────────────────────────────────────────────────────────────
# Simulation loop — scheduling_policy backend
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_scheduling_policy_simulation(
    model,
    config: SimulatorConfig,
    workload: list[SchedulingRequestSpec],
    device,
    seed: int,
) -> SimulationResult:
    """Run the scheduling_policy scheduler with per-step snapshot capture."""
    model.eval()
    cancel_rng = random.Random(seed + 999)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    pending = [
        SchedulingRequestState(spec=req)
        for req in sorted(workload, key=lambda r: (r.arrival_step, r.id))
    ]
    all_states = list(pending)
    waiting: list[tuple] = []
    prefilling: list[SchedulingRequestState] = []
    active: list[SchedulingRequestState] = []
    completed: list[SchedulingRequestState] = []
    cancelled: list[SchedulingRequestState] = []
    snapshots: list[SimulationStepSnapshot] = []

    _sync_if_cuda(device)
    run_start = time.perf_counter()
    step = 0
    total_preemptions = 0

    while pending or waiting or prefilling or active:
        if step > config.max_steps:
            break

        step_start = time.perf_counter()
        prefill_tokens = 0
        decode_tokens = 0
        step_preemptions = 0
        tokens_used = 0
        step_arrivals = []
        step_completions = []
        step_cancellations = []

        # --- Arrivals ---
        while pending and pending[0].spec.arrival_step <= step:
            req = pending.pop(0)
            req.arrived_at_s = time.perf_counter()
            heapq.heappush(waiting, (_sort_key(config.policy, req), req))
            step_arrivals.append(req.spec.id)

        # --- Admission ---
        step_preemptions += _admit_waiting(
            waiting, prefilling, active, completed,
            policy=config.policy,
            max_batch_size=config.max_batch_size,
            max_kv_tokens=config.max_kv_tokens,
        )

        # --- Decode ---
        decode_batch = active[:min(config.max_batch_size, config.token_budget)]
        if decode_batch:
            decode_tokens = _decode_batch(
                model, decode_batch, device, config.temperature, generator,
            )
            tokens_used += decode_tokens

            for req in list(active):
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    active.remove(req)
                    completed.append(req)
                    step_completions.append(req.spec.id)

        # --- Prefill ---
        if prefilling and tokens_used < config.token_budget:
            req = prefilling[0]
            chunk = min(config.prefill_chunk_size, config.token_budget - tokens_used)
            if chunk <= 0:
                chunk = 1

            p_tokens, emitted = _prefill_chunk(
                model, req, chunk, device, config.temperature, generator,
            )
            prefill_tokens += p_tokens
            decode_tokens += emitted

            if req.is_prefill_done:
                prefilling.remove(req)
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    completed.append(req)
                    step_completions.append(req.spec.id)
                else:
                    active.append(req)

        # --- Cancellation ---
        if config.cancellation_rate > 0 and active:
            if cancel_rng.random() < config.cancellation_rate:
                victim = cancel_rng.choice(active)
                active.remove(victim)
                victim.completed_at_s = time.perf_counter()
                cancelled.append(victim)
                step_cancellations.append(victim.spec.id)

        total_preemptions += step_preemptions

        total_gen = sum(len(r.generated_tokens) for r in completed) + \
                    sum(len(r.generated_tokens) for r in active) + \
                    sum(len(r.generated_tokens) for r in cancelled)

        snapshots.append(SimulationStepSnapshot(
            step=step,
            wall_time_s=time.perf_counter() - run_start,
            pending_count=len(pending),
            waiting_count=len(waiting),
            prefilling_count=len(prefilling),
            active_count=len(active),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            decode_batch_size=len(decode_batch),
            arrivals=step_arrivals,
            completions=step_completions,
            cancellations=step_cancellations,
            preemptions=step_preemptions,
            total_completed=len(completed),
            total_generated_tokens=total_gen,
        ))

        step += 1

    _sync_if_cuda(device)
    run_end = time.perf_counter()

    latencies = [
        req.completed_at_s - req.arrived_at_s
        for req in all_states
        if req.completed_at_s is not None and req.arrived_at_s is not None
        and req not in cancelled
    ]
    ttfts = [
        req.first_token_at_s - req.arrived_at_s
        for req in all_states
        if req.first_token_at_s is not None and req.arrived_at_s is not None
    ]

    return SimulationResult(
        config=config,
        snapshots=snapshots,
        total_requests=len(all_states),
        completed_requests=len(completed),
        cancelled_requests=len(cancelled),
        total_generated_tokens=sum(len(r.generated_tokens) for r in all_states),
        total_seconds=run_end - run_start,
        request_latencies_s=latencies,
        ttft_s=ttfts,
        total_preemptions=total_preemptions,
    )


# ──────────────────────────────────────────────────────────────────────
# Simulation loop — interleaving backend
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_interleaving_simulation(
    model,
    config: SimulatorConfig,
    workload: list[InterleaveRequestSpec],
    device,
    seed: int,
) -> SimulationResult:
    """Run the interleaving (fused decode-prefill) scheduler with snapshot capture."""
    model.eval()
    cancel_rng = random.Random(seed + 999)

    generator = _make_generator(device, seed)

    pending = [
        InterleaveRequestState(spec=req)
        for req in sorted(workload, key=lambda r: (r.arrival_step, r.id))
    ]
    all_states = list(pending)
    prefilling: list[InterleaveRequestState] = []
    active: list[InterleaveRequestState] = []
    completed: list[InterleaveRequestState] = []
    cancelled: list[InterleaveRequestState] = []
    snapshots: list[SimulationStepSnapshot] = []

    _interleave_sync(device)
    run_start = time.perf_counter()
    step = 0

    while pending or prefilling or active:
        if step > config.max_steps:
            break

        step_start = time.perf_counter()
        prefill_tokens = 0
        decode_tokens = 0
        step_arrivals = []
        step_completions = []
        step_cancellations = []

        # --- Arrivals ---
        arrivals = _interleave_arrive(pending, step)
        prefilling.extend(arrivals)
        step_arrivals = [r.spec.id for r in arrivals]

        # --- Fused decode + prefill ---
        decode_batch = active[:min(config.max_batch_size, config.token_budget)]
        remaining_budget = config.token_budget - len(decode_batch)
        prefill_req = prefilling[0] if prefilling and remaining_budget > 0 else None
        prefill_chunk = min(config.prefill_chunk_size, remaining_budget) if prefill_req else 0

        if decode_batch or prefill_req is not None:
            p_tokens, d_tokens, _ = _run_fused_step(
                model, decode_batch, prefill_req, prefill_chunk,
                device, config.temperature, generator,
            )
            prefill_tokens += p_tokens
            decode_tokens += d_tokens

        # --- Check completions in active ---
        for req in list(active):
            if req.is_done:
                req.completed_at_s = time.perf_counter()
                active.remove(req)
                completed.append(req)
                step_completions.append(req.spec.id)

        # --- Promote finished prefill ---
        if prefill_req is not None and prefill_req.is_prefill_done:
            prefilling.pop(0)
            if prefill_req.is_done:
                prefill_req.completed_at_s = time.perf_counter()
                completed.append(prefill_req)
                step_completions.append(prefill_req.spec.id)
            else:
                active.append(prefill_req)

        # --- Cancellation ---
        if config.cancellation_rate > 0 and active:
            if cancel_rng.random() < config.cancellation_rate:
                victim = cancel_rng.choice(active)
                active.remove(victim)
                victim.completed_at_s = time.perf_counter()
                cancelled.append(victim)
                step_cancellations.append(victim.spec.id)

        total_gen = sum(len(r.generated_tokens) for r in completed) + \
                    sum(len(r.generated_tokens) for r in active) + \
                    sum(len(r.generated_tokens) for r in cancelled)

        snapshots.append(SimulationStepSnapshot(
            step=step,
            wall_time_s=time.perf_counter() - run_start,
            pending_count=len(pending),
            waiting_count=0,  # interleaving backend has no explicit waiting queue
            prefilling_count=len(prefilling),
            active_count=len(active),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            decode_batch_size=len(decode_batch),
            arrivals=step_arrivals,
            completions=step_completions,
            cancellations=step_cancellations,
            preemptions=0,
            total_completed=len(completed),
            total_generated_tokens=total_gen,
        ))

        step += 1

    _interleave_sync(device)
    run_end = time.perf_counter()

    latencies = [
        req.completed_at_s - req.arrived_at_s
        for req in all_states
        if req.completed_at_s is not None and req.arrived_at_s is not None
        and req not in cancelled
    ]
    ttfts = [
        req.first_token_at_s - req.arrived_at_s
        for req in all_states
        if req.first_token_at_s is not None and req.arrived_at_s is not None
    ]

    return SimulationResult(
        config=config,
        snapshots=snapshots,
        total_requests=len(all_states),
        completed_requests=len(completed),
        cancelled_requests=len(cancelled),
        total_generated_tokens=sum(len(r.generated_tokens) for r in all_states),
        total_seconds=run_end - run_start,
        request_latencies_s=latencies,
        ttft_s=ttfts,
        total_preemptions=0,
    )


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def run_simulation(
    model,
    config: SimulatorConfig,
    vocab_size: int,
    device,
    seed: int = 1337,
) -> SimulationResult:
    """
    Generate a workload and run a full simulation with per-step telemetry.

    Args:
        model:      Trained NanoGPT model.
        config:     SimulatorConfig with arrival pattern, scheduler knobs, etc.
        vocab_size: Vocabulary size for generating random prompts.
        device:     torch device.
        seed:       Random seed for reproducibility.

    Returns:
        SimulationResult with snapshots, latencies, and summary stats.
    """
    workload = generate_workload(config, vocab_size, seed=seed)

    if config.backend == SchedulerBackend.SCHEDULING_POLICY:
        return _run_scheduling_policy_simulation(
            model, config, workload, device, seed,
        )
    elif config.backend == SchedulerBackend.INTERLEAVING:
        return _run_interleaving_simulation(
            model, config, workload, device, seed,
        )
    else:
        raise ValueError(f"Unknown backend: {config.backend}")
