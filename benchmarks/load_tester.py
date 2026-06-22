"""
Concurrent load tester with p99 latency analysis for NanoGPT inference.

Simulates realistic concurrent load against the in-process scheduler
and computes percentile latency distributions (p50, p90, p95, p99)
for TTFT, inter-token latency (ITL), end-to-end (E2E), and queue wait.

Supports four load patterns:
  - constant:  N requests at fixed intervals
  - burst:     all N requests arrive at once
  - ramp:      linearly increasing arrival rate
  - poisson:   exponentially distributed inter-arrival times

All runs are CPU-only. The in-process mode directly calls a full-context-recompute
scheduler loop; the HTTP mode (for later GPU use) is stubbed out.

Usage:
    from benchmarks.load_tester import LoadTester

    tester = LoadTester(model, vocab_size=65, device="cpu")
    report = tester.run_load_test(pattern="constant", concurrency=4, num_requests=10)
    tester.print_report(report)
"""

import json
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PercentileStats:
    """Latency distribution summary."""
    p50: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float
    mean: float
    std: float

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_values(values: list[float]) -> "PercentileStats":
        """Compute percentile stats from a list of raw values."""
        if not values:
            return PercentileStats(
                p50=0.0, p90=0.0, p95=0.0, p99=0.0,
                min=0.0, max=0.0, mean=0.0, std=0.0,
            )
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return PercentileStats(
            p50=sorted_vals[int(n * 0.50)] if n > 0 else 0.0,
            p90=sorted_vals[min(int(n * 0.90), n - 1)],
            p95=sorted_vals[min(int(n * 0.95), n - 1)],
            p99=sorted_vals[min(int(n * 0.99), n - 1)],
            min=sorted_vals[0],
            max=sorted_vals[-1],
            mean=mean(sorted_vals),
            std=stdev(sorted_vals) if n > 1 else 0.0,
        )


@dataclass
class LatencyReport:
    """Full latency report for one load test run."""
    scenario_name: str
    pattern: str                    # "constant", "burst", "ramp", "poisson"
    concurrency: int                # number of concurrent clients
    num_requests: int               # total requests submitted
    completed_requests: int         # requests that finished generation

    ttft: PercentileStats           # time-to-first-token (ms)
    itl: PercentileStats            # inter-token latency (ms)
    e2e: PercentileStats            # end-to-end request latency (ms)
    queue_wait: PercentileStats     # time in waiting queue (ms)

    throughput_tok_s: float         # total generated tokens / wall time
    total_duration_s: float         # wall-clock duration
    goodput: float                  # completed / submitted

    total_tokens_generated: int
    total_prompt_tokens: int
    timestamp: str                  # ISO 8601

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoadRequest:
    """Internal request state tracked during a load test."""
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_step: int                 # scheduler step at which this request arrives
    generated_tokens: list[int] = field(default_factory=list)
    status: str = "waiting"           # waiting → active → done

    # Timing bookmarks (seconds, perf_counter)
    created_at_s: float | None = None      # when the request was created
    submitted_at_s: float | None = None    # when admitted to the scheduler
    first_token_at_s: float | None = None  # when first generated token emitted
    completed_at_s: float | None = None    # when generation finished
    token_times_s: list[float] = field(default_factory=list)  # timestamp of each token

    @property
    def is_done(self) -> bool:
        return len(self.generated_tokens) >= self.max_new_tokens

    @property
    def tokens_so_far(self) -> list[int]:
        return self.prompt_tokens + self.generated_tokens


# ──────────────────────────────────────────────────────────────────────
# Arrival pattern generators
# ──────────────────────────────────────────────────────────────────────

def _generate_arrival_steps(
    pattern: str,
    num_requests: int,
    concurrency: int,
    total_steps: int = 200,
    seed: int = 42,
) -> list[int]:
    """
    Generate arrival step assignments for each request.

    Returns a list of length `num_requests` where each element is the
    scheduler step at which that request arrives.
    """
    rng = random.Random(seed)

    if pattern == "burst":
        # All requests arrive at step 0
        return [0] * num_requests

    elif pattern == "constant":
        # Requests arrive evenly spaced, `concurrency` at a time
        steps = []
        interval = max(1, total_steps // (num_requests // max(concurrency, 1)))
        for i in range(num_requests):
            steps.append((i // concurrency) * interval)
        return steps

    elif pattern == "ramp":
        # Linearly increasing: early requests are spaced far apart,
        # later requests arrive closer together
        steps = []
        for i in range(num_requests):
            # Quadratic spacing: t = total_steps * (i/N)^2
            frac = i / max(num_requests - 1, 1)
            step = int(total_steps * (1.0 - (1.0 - frac) ** 2))
            steps.append(step)
        return steps

    elif pattern == "poisson":
        # Exponentially distributed inter-arrival times
        rate = num_requests / total_steps  # avg arrivals per step
        steps = []
        current = 0.0
        for _ in range(num_requests):
            gap = rng.expovariate(rate) if rate > 0 else 1.0
            current += gap
            steps.append(int(current))
        return steps

    else:
        raise ValueError(f"Unknown arrival pattern: {pattern}")


# ──────────────────────────────────────────────────────────────────────
# In-process scheduler (chunked prefill + batched decode)
# ──────────────────────────────────────────────────────────────────────

def _generate_next_token(model, tokens, *, device, block_size, temperature=1.0):
    """Run a full-context forward pass and sample the next token."""
    _clear_kv_cache(model)
    seq = torch.tensor([tokens], dtype=torch.long, device=device)
    seq = seq[:, -block_size:]  # clamp to positional embedding range
    logits, _ = model(seq, start_pos=0)
    logits = logits[:, -1, :]
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


@torch.no_grad()
def _run_scheduled_load_test(
    model,
    requests: list[LoadRequest],
    *,
    device: str,
    block_size: int,
    token_budget: int = 16,
    max_batch_size: int = 4,
    temperature: float = 1.0,
    seed: int = 42,
    max_steps: int = 5000,
) -> list[LoadRequest]:
    """
    Run a simplified scheduler loop with full-context recompute.

    Requests flow through two queues:
      pending → waiting → active → done

    Active requests get one decode step per scheduler step (full recompute,
    no KV cache). New requests are admitted from the waiting queue when
    capacity is available, getting their first token via a full forward pass.

    Each request's timing fields are filled in as it progresses.
    """
    model.eval()
    torch.manual_seed(seed)

    pending = sorted(requests, key=lambda r: r.arrival_step)
    waiting = []
    active = []

    step = 0

    while (pending or waiting or active) and step < max_steps:
        now = time.perf_counter()

        # --- Arrivals: move pending → waiting ---
        while pending and pending[0].arrival_step <= step:
            req = pending.pop(0)
            req.submitted_at_s = now
            req.status = "waiting"
            waiting.append(req)

        tokens_used = 0

        # --- Decode active requests (priority over new admissions) ---
        for req in active[:min(max_batch_size, token_budget)]:
            token = _generate_next_token(
                model, req.tokens_so_far,
                device=device, block_size=block_size, temperature=temperature,
            )
            now_token = time.perf_counter()
            req.generated_tokens.append(token)
            req.token_times_s.append(now_token)
            tokens_used += 1

        # Retire completed requests
        active = [r for r in active if not r.is_done]
        for req in requests:
            if req.is_done and req.status != "done":
                req.completed_at_s = time.perf_counter()
                req.status = "done"

        # --- Admit new requests from waiting → active ---
        while waiting and len(active) < max_batch_size and tokens_used < token_budget:
            req = waiting.pop(0)
            token = _generate_next_token(
                model, req.prompt_tokens,
                device=device, block_size=block_size, temperature=temperature,
            )
            now_token = time.perf_counter()
            req.generated_tokens.append(token)
            req.token_times_s.append(now_token)
            req.first_token_at_s = now_token
            tokens_used += 1

            if req.is_done:
                req.completed_at_s = now_token
                req.status = "done"
            else:
                req.status = "active"
                active.append(req)

        step += 1

    # Mark any unfinished requests
    for req in waiting + active:
        if req.completed_at_s is None:
            req.completed_at_s = time.perf_counter()
            req.status = "timeout"

    model.train()
    return requests


def _clear_kv_cache(model):
    """Clear KV caches in all Head modules."""
    for module in model.modules():
        if hasattr(module, "key_cache"):
            module.key_cache = None
        if hasattr(module, "value_cache"):
            module.value_cache = None


# ──────────────────────────────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────────────────────────────

def _build_latency_report(
    requests: list[LoadRequest],
    scenario_name: str,
    pattern: str,
    concurrency: int,
    run_start: float,
    run_end: float,
) -> LatencyReport:
    """Build a LatencyReport from completed LoadRequests."""

    # Collect per-request latencies
    ttft_values = []
    e2e_values = []
    queue_wait_values = []
    itl_values = []

    completed = [r for r in requests if r.status == "done"]

    for req in completed:
        # TTFT: time from submission to first token
        if req.first_token_at_s is not None and req.submitted_at_s is not None:
            ttft_ms = (req.first_token_at_s - req.submitted_at_s) * 1000
            ttft_values.append(ttft_ms)

        # E2E: time from submission to completion
        if req.completed_at_s is not None and req.submitted_at_s is not None:
            e2e_ms = (req.completed_at_s - req.submitted_at_s) * 1000
            e2e_values.append(e2e_ms)

        # Queue wait: time from creation to submission (time spent waiting)
        if req.submitted_at_s is not None and req.created_at_s is not None:
            wait_ms = (req.submitted_at_s - req.created_at_s) * 1000
            queue_wait_values.append(wait_ms)

        # ITL: inter-token latency from token timestamps
        if len(req.token_times_s) >= 2:
            for j in range(1, len(req.token_times_s)):
                gap_ms = (req.token_times_s[j] - req.token_times_s[j - 1]) * 1000
                itl_values.append(gap_ms)

    # Compute throughput
    total_duration = run_end - run_start
    total_tokens = sum(len(r.generated_tokens) for r in requests)
    throughput = total_tokens / total_duration if total_duration > 0 else 0.0

    return LatencyReport(
        scenario_name=scenario_name,
        pattern=pattern,
        concurrency=concurrency,
        num_requests=len(requests),
        completed_requests=len(completed),
        ttft=PercentileStats.from_values(ttft_values),
        itl=PercentileStats.from_values(itl_values),
        e2e=PercentileStats.from_values(e2e_values),
        queue_wait=PercentileStats.from_values(queue_wait_values),
        throughput_tok_s=throughput,
        total_duration_s=total_duration,
        goodput=len(completed) / len(requests) if requests else 0.0,
        total_tokens_generated=total_tokens,
        total_prompt_tokens=sum(len(r.prompt_tokens) for r in requests),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ──────────────────────────────────────────────────────────────────────
# LoadTester — the main class
# ──────────────────────────────────────────────────────────────────────

class LoadTester:
    """
    Concurrent load testing framework for NanoGPT inference.

    In-process mode: runs a chunked-prefill scheduler loop directly.
    HTTP mode: (stubbed for later GPU use) sends requests to server.py.
    """

    def __init__(
        self,
        model,
        *,
        vocab_size: int,
        device: str = "cpu",
        block_size: int = 64,
        mode: str = "in_process",
    ):
        self.model = model
        self.vocab_size = vocab_size
        self.device = device
        self.block_size = block_size
        self.mode = mode

    def _make_prompts(
        self,
        num_requests: int,
        prompt_len: int,
        val_data: torch.Tensor | None = None,
        seed: int = 42,
    ) -> list[list[int]]:
        """Generate deterministic prompt token lists."""
        torch.manual_seed(seed)
        prompts = []

        if val_data is not None and len(val_data) > prompt_len:
            # Draw prompts from validation data
            max_start = len(val_data) - prompt_len
            for _ in range(num_requests):
                start = torch.randint(0, max_start, (1,)).item()
                prompts.append(val_data[start : start + prompt_len].tolist())
        else:
            # Random tokens
            for _ in range(num_requests):
                prompts.append(
                    torch.randint(0, self.vocab_size, (prompt_len,)).tolist()
                )

        return prompts

    def run_load_test(
        self,
        *,
        scenario_name: str = "default",
        pattern: str = "constant",
        concurrency: int = 4,
        num_requests: int = 10,
        prompt_len: int = 16,
        max_tokens: int = 10,
        token_budget: int = 16,
        max_batch_size: int = 4,
        val_data: torch.Tensor | None = None,
        seed: int = 42,
    ) -> LatencyReport:
        """
        Run a single load test scenario.

        Args:
            pattern: "constant", "burst", "ramp", or "poisson"
            concurrency: max concurrent requests for arrival pattern
            num_requests: total requests to submit
            prompt_len: length of each prompt in tokens
            max_tokens: max new tokens to generate per request
            token_budget: scheduler's per-step token budget
            max_batch_size: scheduler's max decode batch size
            val_data: optional validation data to draw prompts from

        Returns:
            LatencyReport with p50/p90/p95/p99 for TTFT, ITL, E2E, queue wait
        """
        if self.mode != "in_process":
            raise NotImplementedError(
                f"HTTP mode not yet implemented. Use mode='in_process'."
            )

        # Generate prompts
        prompts = self._make_prompts(num_requests, prompt_len, val_data, seed)

        # Generate arrival steps
        arrival_steps = _generate_arrival_steps(
            pattern, num_requests, concurrency, seed=seed,
        )

        # Build request objects
        now = time.perf_counter()
        requests = []
        for i, (prompt, arrival) in enumerate(zip(prompts, arrival_steps)):
            req = LoadRequest(
                id=i,
                prompt_tokens=prompt,
                max_new_tokens=max_tokens,
                arrival_step=arrival,
                created_at_s=now,
            )
            requests.append(req)

        # Run the scheduler
        run_start = time.perf_counter()
        completed_requests = _run_scheduled_load_test(
            self.model,
            requests,
            device=self.device,
            block_size=self.block_size,
            token_budget=token_budget,
            max_batch_size=max_batch_size,
            seed=seed,
        )
        run_end = time.perf_counter()

        return _build_latency_report(
            completed_requests, scenario_name, pattern,
            concurrency, run_start, run_end,
        )

    def run_sweep(
        self,
        concurrency_values: list[int] | None = None,
        **kwargs,
    ) -> list[LatencyReport]:
        """Run load tests at multiple concurrency levels."""
        if concurrency_values is None:
            concurrency_values = [1, 2, 4, 8]

        reports = []
        for c in concurrency_values:
            report = self.run_load_test(
                scenario_name=f"concurrency_{c}",
                concurrency=c,
                **kwargs,
            )
            reports.append(report)

        return reports

    # ──────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(report: LatencyReport):
        """Pretty-print a single latency report."""
        print(f"\n  📊 Load Test: {report.scenario_name}")
        print(f"  {'─' * 60}")
        print(f"  Pattern: {report.pattern}  |  Concurrency: {report.concurrency}")
        print(f"  Requests: {report.completed_requests}/{report.num_requests} completed")
        print(f"  Duration: {report.total_duration_s:.2f}s  |  "
              f"Throughput: {report.throughput_tok_s:.1f} tok/s  |  "
              f"Goodput: {report.goodput:.0%}")
        print()

        # Latency table
        headers = ["metric", "p50", "p90", "p95", "p99", "min", "max", "mean", "std"]
        rows = [
            ("TTFT (ms)", report.ttft),
            ("ITL (ms)", report.itl),
            ("E2E (ms)", report.e2e),
            ("Queue (ms)", report.queue_wait),
        ]

        rendered = []
        for label, stats in rows:
            rendered.append([
                label,
                f"{stats.p50:.2f}",
                f"{stats.p90:.2f}",
                f"{stats.p95:.2f}",
                f"{stats.p99:.2f}",
                f"{stats.min:.2f}",
                f"{stats.max:.2f}",
                f"{stats.mean:.2f}",
                f"{stats.std:.2f}",
            ])

        widths = [
            max(len(headers[i]), *(len(r[i]) for r in rendered))
            for i in range(len(headers))
        ]

        def fmt(vals):
            return " | ".join(v.rjust(widths[i]) for i, v in enumerate(vals))

        print(f"  {fmt(headers)}")
        print(f"  {'-+-'.join('-' * w for w in widths)}")
        for row in rendered:
            print(f"  {fmt(row)}")
        print()

    @staticmethod
    def print_sweep_summary(reports: list[LatencyReport]):
        """Print a concurrency sweep comparison table."""
        if not reports:
            return

        print(f"\n  {'=' * 70}")
        print(f"  Concurrency Sweep Summary")
        print(f"  {'=' * 70}")

        headers = ["scenario", "conc", "reqs", "done", "tok/s",
                    "ttft_p50", "ttft_p99", "itl_p50", "itl_p99",
                    "e2e_p50", "e2e_p99", "goodput"]
        rows = []
        for r in reports:
            rows.append([
                r.scenario_name,
                str(r.concurrency),
                str(r.num_requests),
                str(r.completed_requests),
                f"{r.throughput_tok_s:.1f}",
                f"{r.ttft.p50:.1f}",
                f"{r.ttft.p99:.1f}",
                f"{r.itl.p50:.1f}",
                f"{r.itl.p99:.1f}",
                f"{r.e2e.p50:.1f}",
                f"{r.e2e.p99:.1f}",
                f"{r.goodput:.0%}",
            ])

        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rows))
            for i in range(len(headers))
        ]

        def fmt(vals):
            return " | ".join(v.rjust(widths[i]) for i, v in enumerate(vals))

        print(f"  {fmt(headers)}")
        print(f"  {'-+-'.join('-' * w for w in widths)}")
        for row in rows:
            print(f"  {fmt(row)}")
        print()

    @staticmethod
    def save_results(reports: list[LatencyReport], path: str):
        """Save reports to JSON for historical comparison."""
        data = [r.to_dict() for r in reports]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
