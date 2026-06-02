"""
FCFS vs priority scheduling benchmark for nanogpt-scheduling.py.

This benchmark is intentionally standalone. Importing nanogpt-scheduling.py
would immediately train the model, so this file only assumes the model API:

    logits, loss, new_kvs = model(idx, targets=None, pos=None,
                                  past_kvs=None, attn_mask=None)

The benchmark compares:
- fcfs: requests are admitted in arrival order
- priority: lower priority value is served first, with recompute preemption
  when a higher-priority request is blocked behind lower-priority active work
"""

from dataclasses import dataclass, field
import heapq
from statistics import mean
import time
import torch
import torch.nn.functional as F


@dataclass
class SchedulingRequestSpec:
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_step: int = 0
    priority: int = 0
    group: str = "default"


@dataclass
class SchedulingRequestState:
    spec: SchedulingRequestSpec
    generated_tokens: list[int] = field(default_factory=list)
    past_kvs: object = None
    last_token: torch.Tensor | None = None
    prefill_cursor: int = 0
    arrived_at_s: float | None = None
    admitted_at_s: float | None = None
    first_token_at_s: float | None = None
    completed_at_s: float | None = None
    token_times_s: list[float] = field(default_factory=list)
    preemptions: int = 0
    prefill_tokens_processed: int = 0

    @property
    def is_prefill_done(self) -> bool:
        return self.prefill_cursor >= len(self.spec.prompt_tokens)

    @property
    def is_done(self) -> bool:
        return len(self.generated_tokens) >= self.spec.max_new_tokens

    @property
    def cache_tokens(self) -> int:
        if self.past_kvs is None:
            return self.prefill_cursor
        return self.past_kvs[0][0][0].shape[1]

    @property
    def reserved_kv_tokens(self) -> int:
        return len(self.spec.prompt_tokens) + len(self.generated_tokens)

    def reset_for_recompute(self):
        self.generated_tokens.clear()
        self.past_kvs = None
        self.last_token = None
        self.prefill_cursor = 0
        self.first_token_at_s = None
        self.token_times_s.clear()
        self.preemptions += 1


@dataclass
class SchedulingStepMetrics:
    step: int
    waiting: int
    prefilling: int
    active: int
    prefill_tokens: int
    decode_tokens: int
    decode_batch_size: int
    preemptions: int
    forward_seconds: float
    step_seconds: float


@dataclass
class SchedulingRunMetrics:
    name: str
    total_requests: int
    completed_requests: int
    total_prompt_tokens: int
    total_prefill_tokens_processed: int
    total_generated_tokens: int
    total_seconds: float
    request_latencies_s: list[float]
    ttft_s: list[float]
    high_priority_latencies_s: list[float]
    low_priority_latencies_s: list[float]
    total_preemptions: int
    step_metrics: list[SchedulingStepMetrics]

    @property
    def generated_tokens_per_second(self) -> float:
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_generated_tokens / self.total_seconds

    @property
    def avg_latency_s(self) -> float:
        return mean(self.request_latencies_s) if self.request_latencies_s else 0.0

    @property
    def avg_ttft_s(self) -> float:
        return mean(self.ttft_s) if self.ttft_s else 0.0

    @property
    def avg_high_priority_latency_s(self) -> float:
        values = self.high_priority_latencies_s
        return mean(values) if values else 0.0

    @property
    def avg_low_priority_latency_s(self) -> float:
        values = self.low_priority_latencies_s
        return mean(values) if values else 0.0

    @property
    def avg_decode_batch_size(self) -> float:
        values = [s.decode_batch_size for s in self.step_metrics if s.decode_batch_size > 0]
        return mean(values) if values else 0.0

    @property
    def max_decode_batch_size(self) -> int:
        return max((s.decode_batch_size for s in self.step_metrics), default=0)

    @property
    def forward_seconds(self) -> float:
        return sum(s.forward_seconds for s in self.step_metrics)


def _sync_if_cuda(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _sample_next_token(logits, temperature=1.0, generator=None):
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _record_generated_token(req, token):
    now = time.perf_counter()
    req.last_token = token
    req.generated_tokens.append(int(token.item()))
    req.token_times_s.append(now)
    if req.first_token_at_s is None:
        req.first_token_at_s = now


def _stack_request_kvs(requests, device):
    lengths = [req.past_kvs[0][0][0].shape[1] for req in requests]
    max_len = max(lengths)
    pad_lengths = [max_len - length for length in lengths]

    attn_mask = torch.zeros(
        (len(requests), 1, max_len),
        dtype=torch.bool,
        device=device,
    )
    for i, pad in enumerate(pad_lengths):
        attn_mask[i, :, pad:] = True

    n_layer = len(requests[0].past_kvs)
    batched = []
    for layer_idx in range(n_layer):
        n_head = len(requests[0].past_kvs[layer_idx])
        layer = []
        for head_idx in range(n_head):
            keys = []
            values = []
            for req, pad in zip(requests, pad_lengths):
                k, v = req.past_kvs[layer_idx][head_idx]
                if pad > 0:
                    hs = k.shape[-1]
                    k_pad = torch.zeros((1, pad, hs), dtype=k.dtype, device=device)
                    v_pad = torch.zeros((1, pad, hs), dtype=v.dtype, device=device)
                    k = torch.cat([k_pad, k], dim=1)
                    v = torch.cat([v_pad, v], dim=1)
                keys.append(k)
                values.append(v)
            layer.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        batched.append(layer)

    return batched, attn_mask, pad_lengths


def _unstack_request_kvs(requests, batched_kvs, pad_lengths):
    for req_idx, req in enumerate(requests):
        req_kvs = []
        for layer in batched_kvs:
            req_layer = []
            for k, v in layer:
                pad = pad_lengths[req_idx]
                req_layer.append((
                    k[req_idx:req_idx + 1, pad:, :].contiguous(),
                    v[req_idx:req_idx + 1, pad:, :].contiguous(),
                ))
            req_kvs.append(req_layer)
        req.past_kvs = req_kvs


def _sort_key(policy, req):
    if policy == "fcfs":
        return (req.spec.arrival_step, req.spec.id)
    if policy == "priority":
        return (req.spec.priority, req.spec.arrival_step, req.spec.id)
    raise ValueError(f"Unknown scheduling policy: {policy}")


def _kv_tokens_in_memory(active, prefilling):
    return sum(req.reserved_kv_tokens for req in active) + sum(
        len(req.spec.prompt_tokens) for req in prefilling
    )


def _select_preemption_victim(active, candidate, policy):
    if not active:
        return None

    if policy == "priority":
        lower_priority = [req for req in active if req.spec.priority > candidate.spec.priority]
        if not lower_priority:
            return None
        return max(lower_priority, key=lambda r: (r.spec.priority, -r.spec.arrival_step, -r.spec.id))

    return None


def _admit_waiting(
    waiting,
    prefilling,
    active,
    completed,
    *,
    policy,
    max_batch_size,
    max_kv_tokens,
):
    preemptions = 0

    while waiting and not prefilling and len(active) < max_batch_size:
        _, candidate = waiting[0]
        projected = _kv_tokens_in_memory(active, prefilling) + len(candidate.spec.prompt_tokens)

        while projected > max_kv_tokens:
            victim = _select_preemption_victim(active, candidate, policy)
            if victim is None:
                break

            active.remove(victim)
            victim.reset_for_recompute()
            heapq.heappush(waiting, (_sort_key(policy, victim), victim))
            preemptions += 1
            projected = _kv_tokens_in_memory(active, prefilling) + len(candidate.spec.prompt_tokens)

        if projected > max_kv_tokens:
            break

        heapq.heappop(waiting)
        candidate.admitted_at_s = time.perf_counter()
        prefilling.append(candidate)

    # Allow a high-priority request to preempt a full active batch.
    if policy == "priority" and waiting and not prefilling and len(active) >= max_batch_size:
        _, candidate = waiting[0]
        victim = _select_preemption_victim(active, candidate, policy)
        if victim is not None:
            active.remove(victim)
            victim.reset_for_recompute()
            heapq.heappush(waiting, (_sort_key(policy, victim), victim))
            preemptions += 1

            projected = _kv_tokens_in_memory(active, prefilling) + len(candidate.spec.prompt_tokens)
            if projected <= max_kv_tokens:
                heapq.heappop(waiting)
                candidate.admitted_at_s = time.perf_counter()
                prefilling.append(candidate)

    return preemptions


@torch.no_grad()
def _prefill_chunk(model, req, chunk_size, device, temperature, generator):
    start = req.prefill_cursor
    end = min(start + chunk_size, len(req.spec.prompt_tokens))
    tokens = req.spec.prompt_tokens[start:end]

    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    pos = torch.arange(start, end, device=device).unsqueeze(0)

    logits, _, new_kvs = model(idx, pos=pos, past_kvs=req.past_kvs)
    req.prefill_cursor = end
    req.past_kvs = new_kvs
    req.prefill_tokens_processed += len(tokens)

    emitted = 0
    if req.is_prefill_done:
        next_token = _sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            generator=generator,
        )
        _record_generated_token(req, next_token)
        emitted = 1

    return len(tokens), emitted


@torch.no_grad()
def _decode_batch(model, requests, device, temperature, generator):
    if not requests:
        return 0

    input_tokens = torch.cat([req.last_token for req in requests], dim=0)
    positions = torch.tensor(
        [[len(req.spec.prompt_tokens) + len(req.generated_tokens) - 1] for req in requests],
        dtype=torch.long,
        device=device,
    )
    past_kvs, attn_mask, pad_lengths = _stack_request_kvs(requests, device)

    logits, _, new_kvs = model(
        input_tokens,
        pos=positions,
        past_kvs=past_kvs,
        attn_mask=attn_mask,
    )
    next_tokens = _sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        generator=generator,
    )

    _unstack_request_kvs(requests, new_kvs, pad_lengths)

    for i, req in enumerate(requests):
        _record_generated_token(req, next_tokens[i:i + 1])

    return len(requests)


def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


@torch.no_grad()
def run_scheduling_policy(
    model,
    workload,
    *,
    policy,
    device,
    max_batch_size=4,
    token_budget=16,
    prefill_chunk_size=8,
    max_kv_tokens=64,
    temperature=1.0,
    seed=1337,
    max_steps=10000,
):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    pending = [
        SchedulingRequestState(spec=req)
        for req in sorted(workload, key=lambda r: (r.arrival_step, r.id))
    ]
    all_states = list(pending)
    waiting = []
    prefilling = []
    active = []
    completed = []
    step_metrics = []

    _sync_if_cuda(device)
    run_start = time.perf_counter()
    step = 0

    while pending or waiting or prefilling or active:
        if step > max_steps:
            raise RuntimeError(
                f"Scheduling benchmark exceeded max_steps={max_steps}. "
                "Check token budget, max_kv_tokens, and workload sizes."
            )

        step_start = time.perf_counter()
        prefill_tokens = 0
        decode_tokens = 0
        forward_seconds = 0.0
        step_preemptions = 0
        tokens_used = 0

        while pending and pending[0].spec.arrival_step <= step:
            req = pending.pop(0)
            req.arrived_at_s = time.perf_counter()
            heapq.heappush(waiting, (_sort_key(policy, req), req))

        step_preemptions += _admit_waiting(
            waiting,
            prefilling,
            active,
            completed,
            policy=policy,
            max_batch_size=max_batch_size,
            max_kv_tokens=max_kv_tokens,
        )

        decode_batch = active[:min(max_batch_size, token_budget)]
        if decode_batch:
            fwd_start = time.perf_counter()
            decode_tokens = _decode_batch(model, decode_batch, device, temperature, generator)
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start
            tokens_used += decode_tokens

            for req in list(active):
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    active.remove(req)
                    completed.append(req)

        if prefilling and tokens_used < token_budget:
            req = prefilling[0]
            chunk = min(prefill_chunk_size, token_budget - tokens_used)
            if chunk <= 0:
                chunk = 1

            fwd_start = time.perf_counter()
            p_tokens, emitted = _prefill_chunk(
                model,
                req,
                chunk,
                device,
                temperature,
                generator,
            )
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start

            prefill_tokens += p_tokens
            decode_tokens += emitted

            if req.is_prefill_done:
                prefilling.remove(req)
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    completed.append(req)
                else:
                    active.append(req)

        if not decode_batch and not prefilling and waiting and not active:
            _, blocked = waiting[0]
            raise RuntimeError(
                "Scheduling benchmark is blocked: top waiting request cannot be "
                f"admitted. request_id={blocked.spec.id}, prompt_len="
                f"{len(blocked.spec.prompt_tokens)}, max_kv_tokens={max_kv_tokens}"
            )

        step_metrics.append(SchedulingStepMetrics(
            step=step,
            waiting=len(waiting) + len(pending),
            prefilling=len(prefilling),
            active=len(active),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            decode_batch_size=len(decode_batch),
            preemptions=step_preemptions,
            forward_seconds=forward_seconds,
            step_seconds=time.perf_counter() - step_start,
        ))

        step += 1

    _sync_if_cuda(device)
    run_end = time.perf_counter()

    latencies = [
        req.completed_at_s - req.arrived_at_s
        for req in all_states
        if req.completed_at_s is not None and req.arrived_at_s is not None
    ]
    ttfts = [
        req.first_token_at_s - req.arrived_at_s
        for req in all_states
        if req.first_token_at_s is not None and req.arrived_at_s is not None
    ]
    high_priority_latencies = [
        req.completed_at_s - req.arrived_at_s
        for req in all_states
        if req.spec.priority == 0 and req.completed_at_s is not None and req.arrived_at_s is not None
    ]
    low_priority_latencies = [
        req.completed_at_s - req.arrived_at_s
        for req in all_states
        if req.spec.priority > 0 and req.completed_at_s is not None and req.arrived_at_s is not None
    ]

    return SchedulingRunMetrics(
        name=policy,
        total_requests=len(all_states),
        completed_requests=len(completed),
        total_prompt_tokens=sum(len(req.spec.prompt_tokens) for req in all_states),
        total_prefill_tokens_processed=sum(req.prefill_tokens_processed for req in all_states),
        total_generated_tokens=sum(len(req.generated_tokens) for req in all_states),
        total_seconds=run_end - run_start,
        request_latencies_s=latencies,
        ttft_s=ttfts,
        high_priority_latencies_s=high_priority_latencies,
        low_priority_latencies_s=low_priority_latencies,
        total_preemptions=sum(req.preemptions for req in all_states),
        step_metrics=step_metrics,
    )


def make_priority_inversion_workload(*, vocab_size, seed=1337):
    rng = torch.Generator()
    rng.manual_seed(seed)

    def toks(n):
        return torch.randint(0, vocab_size, (n,), generator=rng).tolist()

    return [
        SchedulingRequestSpec(0, toks(10), 18, arrival_step=0, priority=8, group="low_long"),
        SchedulingRequestSpec(1, toks(4), 6, arrival_step=2, priority=0, group="high_short"),
        SchedulingRequestSpec(2, toks(5), 8, arrival_step=3, priority=0, group="high_short"),
        SchedulingRequestSpec(3, toks(8), 10, arrival_step=4, priority=5, group="medium"),
    ]


def make_priority_mix_workload(
    *,
    vocab_size,
    num_requests=12,
    prompt_len=6,
    max_new_tokens=10,
    seed=1337,
):
    rng = torch.Generator()
    rng.manual_seed(seed)
    requests = []
    for i in range(num_requests):
        high_priority = i in {3, 4, 7}
        requests.append(SchedulingRequestSpec(
            id=i,
            prompt_tokens=torch.randint(0, vocab_size, (prompt_len,), generator=rng).tolist(),
            max_new_tokens=max_new_tokens if not high_priority else 5,
            arrival_step=i // 2,
            priority=0 if high_priority else 6,
            group="high" if high_priority else "low",
        ))
    return requests


def make_memory_pressure_workload(*, vocab_size, seed=1337):
    rng = torch.Generator()
    rng.manual_seed(seed)
    specs = [
        (0, 8, 14, 0, 9, "low"),
        (1, 8, 14, 1, 8, "low"),
        (2, 5, 6, 3, 0, "high"),
        (3, 5, 6, 4, 0, "high"),
        (4, 8, 10, 5, 6, "medium"),
    ]
    return [
        SchedulingRequestSpec(
            id=req_id,
            prompt_tokens=torch.randint(0, vocab_size, (prompt_len,), generator=rng).tolist(),
            max_new_tokens=max_new,
            arrival_step=arrival,
            priority=priority,
            group=group,
        )
        for req_id, prompt_len, max_new, arrival, priority, group in specs
    ]


def make_equal_priority_control_workload(*, vocab_size, seed=1337):
    rng = torch.Generator()
    rng.manual_seed(seed)
    return [
        SchedulingRequestSpec(
            id=i,
            prompt_tokens=torch.randint(0, vocab_size, (6,), generator=rng).tolist(),
            max_new_tokens=8,
            arrival_step=i,
            priority=0,
            group="equal_priority",
        )
        for i in range(8)
    ]


def print_scheduling_comparison_table(rows):
    headers = [
        "policy",
        "reqs",
        "done",
        "gen_tok",
        "wall_s",
        "tok/s",
        "avg_ttft_ms",
        "p95_ttft_ms",
        "avg_lat_ms",
        "p95_lat_ms",
        "hi_lat_ms",
        "low_lat_ms",
        "preempt",
        "avg_batch",
        "max_batch",
        "forward_s",
    ]

    rendered = []
    for row in rows:
        rendered.append([
            row.name,
            str(row.total_requests),
            str(row.completed_requests),
            str(row.total_generated_tokens),
            f"{row.total_seconds:.4f}",
            f"{row.generated_tokens_per_second:.2f}",
            f"{row.avg_ttft_s * 1000:.2f}",
            f"{_percentile(row.ttft_s, 0.95) * 1000:.2f}",
            f"{row.avg_latency_s * 1000:.2f}",
            f"{_percentile(row.request_latencies_s, 0.95) * 1000:.2f}",
            f"{row.avg_high_priority_latency_s * 1000:.2f}",
            f"{row.avg_low_priority_latency_s * 1000:.2f}",
            str(row.total_preemptions),
            f"{row.avg_decode_batch_size:.2f}",
            str(row.max_decode_batch_size),
            f"{row.forward_seconds:.4f}",
        ])

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rendered))
        for i in range(len(headers))
    ]

    def fmt(values):
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rendered:
        print(fmt(row))

    if len(rows) == 2:
        fcfs, priority = rows
        print()
        print(
            "Priority throughput ratio: "
            f"{priority.generated_tokens_per_second / fcfs.generated_tokens_per_second:.2f}x"
        )
        if fcfs.avg_high_priority_latency_s > 0:
            print(
                "High-priority latency ratio: "
                f"{priority.avg_high_priority_latency_s / fcfs.avg_high_priority_latency_s:.2f}x"
            )
        if fcfs.avg_latency_s > 0:
            print(
                "Average latency ratio: "
                f"{priority.avg_latency_s / fcfs.avg_latency_s:.2f}x"
            )
        print(f"Priority preemptions: {priority.total_preemptions}")


def run_fcfs_vs_priority_scheduling_benchmark(
    model,
    *,
    vocab_size,
    workload_name="priority_inversion",
    max_batch_size=1,
    token_budget=8,
    prefill_chunk_size=8,
    max_kv_tokens=32,
    device=None,
    seed=1337,
    temperature=1.0,
):
    if device is None:
        device = next(model.parameters()).device

    if workload_name == "priority_inversion":
        workload = make_priority_inversion_workload(vocab_size=vocab_size, seed=seed)
    elif workload_name == "priority_mix":
        workload = make_priority_mix_workload(vocab_size=vocab_size, seed=seed)
    elif workload_name == "memory_pressure":
        workload = make_memory_pressure_workload(vocab_size=vocab_size, seed=seed)
    elif workload_name == "equal_priority_control":
        workload = make_equal_priority_control_workload(vocab_size=vocab_size, seed=seed)
    else:
        raise ValueError(f"Unknown scheduling workload: {workload_name}")

    fcfs = run_scheduling_policy(
        model,
        workload,
        policy="fcfs",
        device=device,
        max_batch_size=max_batch_size,
        token_budget=token_budget,
        prefill_chunk_size=prefill_chunk_size,
        max_kv_tokens=max_kv_tokens,
        temperature=temperature,
        seed=seed,
    )

    priority = run_scheduling_policy(
        model,
        workload,
        policy="priority",
        device=device,
        max_batch_size=max_batch_size,
        token_budget=token_budget,
        prefill_chunk_size=prefill_chunk_size,
        max_kv_tokens=max_kv_tokens,
        temperature=temperature,
        seed=seed,
    )

    print_scheduling_comparison_table([fcfs, priority])

    return {
        "fcfs": fcfs,
        "priority": priority,
    }
