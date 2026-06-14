"""
Sliding window KV eviction benchmark for nanogpt-sliding-window.py.

Standalone benchmark — importing nanogpt-sliding-window.py would trigger
training, so this file re-implements the inference engine with sliding
window support and measures:

1. Memory savings (peak KV cache size with vs without window)
2. Preemption reduction (fewer evictions under tight memory budgets)
3. Quality sweep (token agreement vs full-cache baseline at varying W)
4. Batch capacity (concurrent requests under fixed memory budget)
"""

from dataclasses import dataclass, field
import heapq
from statistics import mean
import time
import torch
import torch.nn.functional as F


# ── Request / Metrics dataclasses ────────────────────────────────────

@dataclass
class SWRequestSpec:
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_step: int = 0
    priority: int = 0
    group: str = "default"


@dataclass
class SWRequestState:
    spec: SWRequestSpec
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
    peak_kv_tokens: int = 0  # track max cache size this request ever held

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

    def reserved_kv_tokens(self, sliding_window=None) -> int:
        total = len(self.spec.prompt_tokens) + len(self.generated_tokens)
        if sliding_window is not None:
            return min(total, sliding_window)
        return total

    def reset_for_recompute(self):
        self.generated_tokens.clear()
        self.past_kvs = None
        self.last_token = None
        self.prefill_cursor = 0
        self.first_token_at_s = None
        self.token_times_s.clear()
        self.preemptions += 1


@dataclass
class SWStepMetrics:
    step: int
    waiting: int
    prefilling: int
    active: int
    prefill_tokens: int
    decode_tokens: int
    decode_batch_size: int
    preemptions: int
    total_kv_tokens: int  # sum of all active cache sizes this step


@dataclass
class SWRunMetrics:
    name: str
    sliding_window: int | None
    total_requests: int
    completed_requests: int
    total_generated_tokens: int
    total_seconds: float
    total_preemptions: int
    avg_peak_kv_per_request: float
    max_peak_kv_per_request: int
    request_latencies_s: list[float]
    ttft_s: list[float]
    step_metrics: list[SWStepMetrics]
    generated_token_sequences: dict[int, list[int]] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
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
    def avg_decode_batch_size(self) -> float:
        vals = [s.decode_batch_size for s in self.step_metrics if s.decode_batch_size > 0]
        return mean(vals) if vals else 0.0

    @property
    def max_decode_batch_size(self) -> int:
        return max((s.decode_batch_size for s in self.step_metrics), default=0)

    @property
    def max_total_kv_tokens(self) -> int:
        return max((s.total_kv_tokens for s in self.step_metrics), default=0)


# ── Engine helpers ───────────────────────────────────────────────────

def _sync_if_cuda(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _sample(logits, temperature=1.0, generator=None):
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _record_token(req, token):
    now = time.perf_counter()
    req.last_token = token
    req.generated_tokens.append(int(token.item()))
    req.token_times_s.append(now)
    if req.first_token_at_s is None:
        req.first_token_at_s = now


def _evict_kv_cache(req, window_size):
    """Trim per-request KV tensors to last W entries."""
    if req.past_kvs is None or window_size is None:
        return
    trimmed = []
    for layer in req.past_kvs:
        trimmed_layer = []
        for k, v in layer:
            T = k.shape[1]
            if T > window_size:
                k = k[:, -window_size:, :]
                v = v[:, -window_size:, :]
            trimmed_layer.append((k, v))
        trimmed.append(trimmed_layer)
    req.past_kvs = trimmed


def _update_peak_kv(req):
    """Track the largest cache this request has ever held."""
    ct = req.cache_tokens
    if ct > req.peak_kv_tokens:
        req.peak_kv_tokens = ct


def _stack_kvs(requests, device):
    lengths = [req.past_kvs[0][0][0].shape[1] for req in requests]
    max_len = max(lengths)
    pad_lengths = [max_len - l for l in lengths]

    attn_mask = torch.zeros((len(requests), 1, max_len), dtype=torch.bool, device=device)
    for i, pad in enumerate(pad_lengths):
        attn_mask[i, :, pad:] = True

    n_layer = len(requests[0].past_kvs)
    batched = []
    for layer_idx in range(n_layer):
        n_head = len(requests[0].past_kvs[layer_idx])
        layer = []
        for head_idx in range(n_head):
            keys, values = [], []
            for req, pad in zip(requests, pad_lengths):
                k, v = req.past_kvs[layer_idx][head_idx]
                if pad > 0:
                    hs = k.shape[-1]
                    k = torch.cat([torch.zeros(1, pad, hs, dtype=k.dtype, device=device), k], dim=1)
                    v = torch.cat([torch.zeros(1, pad, hs, dtype=v.dtype, device=device), v], dim=1)
                keys.append(k)
                values.append(v)
            layer.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        batched.append(layer)

    return batched, attn_mask, pad_lengths


def _unstack_kvs(requests, batched_kvs, pad_lengths):
    for idx, req in enumerate(requests):
        req_kvs = []
        for layer in batched_kvs:
            req_layer = []
            for k, v in layer:
                pad = pad_lengths[idx]
                req_layer.append((
                    k[idx:idx + 1, pad:, :].contiguous(),
                    v[idx:idx + 1, pad:, :].contiguous(),
                ))
            req_kvs.append(req_layer)
        req.past_kvs = req_kvs


# ── Scheduling helpers ───────────────────────────────────────────────

def _sort_key(policy, req):
    if policy == "fcfs":
        return (req.spec.arrival_step, req.spec.id)
    if policy == "priority":
        return (req.spec.priority, req.spec.arrival_step, req.spec.id)
    raise ValueError(f"Unknown policy: {policy}")


def _kv_in_memory(active, prefilling, sliding_window):
    return (
        sum(r.reserved_kv_tokens(sliding_window) for r in active)
        + sum(len(r.spec.prompt_tokens) for r in prefilling)
    )


def _admit_waiting(waiting, prefilling, active, *, policy, max_batch_size,
                   max_kv_tokens, sliding_window):
    preemptions = 0
    while waiting and not prefilling and len(active) < max_batch_size:
        _, candidate = waiting[0]
        projected = _kv_in_memory(active, prefilling, sliding_window) + len(candidate.spec.prompt_tokens)
        if projected > max_kv_tokens:
            break
        heapq.heappop(waiting)
        candidate.admitted_at_s = time.perf_counter()
        prefilling.append(candidate)
    return preemptions


# ── Forward pass wrappers ────────────────────────────────────────────

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
        next_token = _sample(logits[:, -1, :], temperature=temperature, generator=generator)
        _record_token(req, next_token)
        emitted = 1
    return len(tokens), emitted


@torch.no_grad()
def _decode_batch(model, requests, device, temperature, generator, sliding_window):
    if not requests:
        return 0
    input_tokens = torch.cat([req.last_token for req in requests], dim=0)
    positions = torch.tensor(
        [[len(req.spec.prompt_tokens) + len(req.generated_tokens) - 1] for req in requests],
        dtype=torch.long, device=device,
    )
    past_kvs, attn_mask, pad_lengths = _stack_kvs(requests, device)
    logits, _, new_kvs = model(input_tokens, pos=positions, past_kvs=past_kvs, attn_mask=attn_mask)
    next_tokens = _sample(logits[:, -1, :], temperature=temperature, generator=generator)
    _unstack_kvs(requests, new_kvs, pad_lengths)

    # Sliding window eviction — trim KV caches after unstacking
    if sliding_window is not None:
        for req in requests:
            _evict_kv_cache(req, sliding_window)

    for i, req in enumerate(requests):
        _record_token(req, next_tokens[i:i + 1])
        _update_peak_kv(req)

    return len(requests)


# ── Main engine ──────────────────────────────────────────────────────

@torch.no_grad()
def run_sliding_window_engine(
    model, workload, *, policy="fcfs", device, max_batch_size=4,
    token_budget=16, prefill_chunk_size=8, max_kv_tokens=64,
    sliding_window=None, temperature=1.0, seed=1337, max_steps=10000,
):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    pending = [SWRequestState(spec=s) for s in sorted(workload, key=lambda r: (r.arrival_step, r.id))]
    all_states = list(pending)
    waiting, prefilling, active, completed = [], [], [], []
    step_metrics = []

    _sync_if_cuda(device)
    run_start = time.perf_counter()
    step = 0

    while pending or waiting or prefilling or active:
        if step > max_steps:
            raise RuntimeError(f"Exceeded max_steps={max_steps}")

        prefill_tokens = decode_tokens = 0
        step_preemptions = 0
        tokens_used = 0

        # Arrivals
        while pending and pending[0].spec.arrival_step <= step:
            req = pending.pop(0)
            req.arrived_at_s = time.perf_counter()
            heapq.heappush(waiting, (_sort_key(policy, req), req))

        step_preemptions += _admit_waiting(
            waiting, prefilling, active,
            policy=policy, max_batch_size=max_batch_size,
            max_kv_tokens=max_kv_tokens, sliding_window=sliding_window,
        )

        # Decode
        decode_batch = active[:min(max_batch_size, token_budget)]
        if decode_batch:
            decode_tokens = _decode_batch(model, decode_batch, device, temperature, generator, sliding_window)
            tokens_used += decode_tokens
            for req in list(active):
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    active.remove(req)
                    completed.append(req)

        # Prefill
        if prefilling and tokens_used < token_budget:
            req = prefilling[0]
            chunk = min(prefill_chunk_size, token_budget - tokens_used)
            if chunk <= 0:
                chunk = 1
            p_tokens, emitted = _prefill_chunk(model, req, chunk, device, temperature, generator)

            # Evict after prefill chunk too if window is set
            if sliding_window is not None and req.past_kvs is not None:
                _evict_kv_cache(req, sliding_window)

            _update_peak_kv(req)
            prefill_tokens += p_tokens
            decode_tokens += emitted
            if req.is_prefill_done:
                prefilling.remove(req)
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    completed.append(req)
                else:
                    active.append(req)

        # Deadlock check
        if not decode_batch and not prefilling and waiting and not active:
            _, blocked = waiting[0]
            raise RuntimeError(
                f"Deadlocked: req {blocked.spec.id} prompt_len={len(blocked.spec.prompt_tokens)}, "
                f"max_kv_tokens={max_kv_tokens}"
            )

        total_kv = sum(r.cache_tokens for r in active)
        step_metrics.append(SWStepMetrics(
            step=step, waiting=len(waiting) + len(pending),
            prefilling=len(prefilling), active=len(active),
            prefill_tokens=prefill_tokens, decode_tokens=decode_tokens,
            decode_batch_size=len(decode_batch), preemptions=step_preemptions,
            total_kv_tokens=total_kv,
        ))
        step += 1

    _sync_if_cuda(device)
    run_end = time.perf_counter()

    latencies = [
        r.completed_at_s - r.arrived_at_s for r in all_states
        if r.completed_at_s is not None and r.arrived_at_s is not None
    ]
    ttfts = [
        r.first_token_at_s - r.arrived_at_s for r in all_states
        if r.first_token_at_s is not None and r.arrived_at_s is not None
    ]
    token_seqs = {r.spec.id: list(r.generated_tokens) for r in all_states}
    peaks = [r.peak_kv_tokens for r in all_states]

    return SWRunMetrics(
        name=f"window={sliding_window}" if sliding_window else "full_cache",
        sliding_window=sliding_window,
        total_requests=len(all_states),
        completed_requests=len(completed),
        total_generated_tokens=sum(len(r.generated_tokens) for r in all_states),
        total_seconds=run_end - run_start,
        total_preemptions=sum(r.preemptions for r in all_states),
        avg_peak_kv_per_request=mean(peaks) if peaks else 0.0,
        max_peak_kv_per_request=max(peaks) if peaks else 0,
        request_latencies_s=latencies,
        ttft_s=ttfts,
        step_metrics=step_metrics,
        generated_token_sequences=token_seqs,
    )


# ── Workload generators ─────────────────────────────────────────────

def make_long_generation_workload(*, vocab_size, seed=1337):
    """3 requests with moderate prompts + long generation — shows memory growth."""
    rng = torch.Generator(); rng.manual_seed(seed)
    toks = lambda n: torch.randint(0, vocab_size, (n,), generator=rng).tolist()
    return [
        SWRequestSpec(0, toks(8), 30, arrival_step=0),
        SWRequestSpec(1, toks(6), 30, arrival_step=0),
        SWRequestSpec(2, toks(8), 30, arrival_step=1),
    ]


def make_tight_memory_workload(*, vocab_size, seed=1337):
    """5 requests that will overflow a tight max_kv_tokens without window."""
    rng = torch.Generator(); rng.manual_seed(seed)
    toks = lambda n: torch.randint(0, vocab_size, (n,), generator=rng).tolist()
    return [
        SWRequestSpec(0, toks(8), 20, arrival_step=0, priority=5),
        SWRequestSpec(1, toks(8), 20, arrival_step=0, priority=5),
        SWRequestSpec(2, toks(6), 15, arrival_step=1, priority=0),
        SWRequestSpec(3, toks(6), 15, arrival_step=2, priority=0),
        SWRequestSpec(4, toks(6), 10, arrival_step=3, priority=3),
    ]


def make_single_request_workload(*, vocab_size, prompt_len=10, max_new=40, seed=1337):
    """Single request — used for quality sweep (no batching noise)."""
    rng = torch.Generator(); rng.manual_seed(seed)
    toks = torch.randint(0, vocab_size, (prompt_len,), generator=rng).tolist()
    return [SWRequestSpec(0, toks, max_new, arrival_step=0)]


def make_batch_capacity_workload(*, vocab_size, num_requests=8, seed=1337):
    """Many small requests — shows how window increases concurrent batch size."""
    rng = torch.Generator(); rng.manual_seed(seed)
    toks = lambda n: torch.randint(0, vocab_size, (n,), generator=rng).tolist()
    return [
        SWRequestSpec(i, toks(6), 20, arrival_step=i // 2)
        for i in range(num_requests)
    ]


# ── Comparison printers ──────────────────────────────────────────────

def _pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[round((len(s) - 1) * p)]


def print_comparison_table(rows: list[SWRunMetrics]):
    headers = [
        "config", "reqs", "done", "gen_tok", "wall_s", "tok/s",
        "preempt", "avg_peak_kv", "max_peak_kv", "max_total_kv",
        "avg_batch", "max_batch",
    ]
    rendered = []
    for r in rows:
        rendered.append([
            r.name,
            str(r.total_requests),
            str(r.completed_requests),
            str(r.total_generated_tokens),
            f"{r.total_seconds:.4f}",
            f"{r.tokens_per_second:.1f}",
            str(r.total_preemptions),
            f"{r.avg_peak_kv_per_request:.1f}",
            str(r.max_peak_kv_per_request),
            str(r.max_total_kv_tokens),
            f"{r.avg_decode_batch_size:.2f}",
            str(r.max_decode_batch_size),
        ])
    widths = [max(len(headers[i]), *(len(r[i]) for r in rendered)) for i in range(len(headers))]

    def fmt(vals):
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rendered:
        print(fmt(row))


def print_quality_sweep_results(results: dict[int | None, SWRunMetrics]):
    """Compare generated token sequences against full-cache baseline."""
    baseline_key = None  # None = no window
    baseline_seqs = results[baseline_key].generated_token_sequences

    print(f"\n{'window':>8s} | {'agreement':>10s} | {'gen_tokens':>10s}")
    print("-" * 38)

    for ws in sorted((k for k in results if k is not None)):
        seqs = results[ws].generated_token_sequences
        total, matches = 0, 0
        for req_id, baseline_toks in baseline_seqs.items():
            ws_toks = seqs.get(req_id, [])
            for a, b in zip(baseline_toks, ws_toks):
                total += 1
                if a == b:
                    matches += 1
        agreement = matches / total * 100 if total > 0 else 0.0
        gen = results[ws].total_generated_tokens
        print(f"{ws:>8d} | {agreement:>9.1f}% | {gen:>10d}")

    gen = results[baseline_key].total_generated_tokens
    print(f"{'full':>8s} | {'100.0':>9s}% | {gen:>10d}  (baseline)")


# ── Top-level benchmark runners ──────────────────────────────────────

def run_window_vs_full_benchmark(
    model, *, vocab_size, device, workload_name="long_generation",
    sliding_window=32, max_batch_size=4, token_budget=16,
    prefill_chunk_size=8, max_kv_tokens=128, seed=1337,
):
    """Run same workload with and without sliding window. Print comparison."""
    if workload_name == "long_generation":
        workload = make_long_generation_workload(vocab_size=vocab_size, seed=seed)
    elif workload_name == "tight_memory":
        workload = make_tight_memory_workload(vocab_size=vocab_size, seed=seed)
    elif workload_name == "batch_capacity":
        workload = make_batch_capacity_workload(vocab_size=vocab_size, seed=seed)
    else:
        raise ValueError(f"Unknown workload: {workload_name}")

    common = dict(
        policy="fcfs", device=device, max_batch_size=max_batch_size,
        token_budget=token_budget, prefill_chunk_size=prefill_chunk_size,
        max_kv_tokens=max_kv_tokens, seed=seed,
    )

    full = run_sliding_window_engine(model, workload, sliding_window=None, **common)

    # Rebuild workload with same seed so token IDs match
    if workload_name == "long_generation":
        workload = make_long_generation_workload(vocab_size=vocab_size, seed=seed)
    elif workload_name == "tight_memory":
        workload = make_tight_memory_workload(vocab_size=vocab_size, seed=seed)
    elif workload_name == "batch_capacity":
        workload = make_batch_capacity_workload(vocab_size=vocab_size, seed=seed)

    windowed = run_sliding_window_engine(model, workload, sliding_window=sliding_window, **common)

    print_comparison_table([full, windowed])

    if full.max_peak_kv_per_request > 0:
        reduction = (1 - windowed.max_peak_kv_per_request / full.max_peak_kv_per_request) * 100
        print(f"\nPeak KV reduction: {reduction:.1f}%")
    if full.total_preemptions != windowed.total_preemptions:
        print(f"Preemption change: {full.total_preemptions} → {windowed.total_preemptions}")

    return {"full_cache": full, f"window={sliding_window}": windowed}


def run_quality_sweep(
    model, *, vocab_size, device, window_sizes=(8, 16, 32, 64),
    prompt_len=10, max_new=40, max_kv_tokens=128, seed=1337,
):
    """Generate with different window sizes and compare token agreement."""
    results = {}

    # Full cache baseline
    workload = make_single_request_workload(
        vocab_size=vocab_size, prompt_len=prompt_len, max_new=max_new, seed=seed
    )
    results[None] = run_sliding_window_engine(
        model, workload, policy="fcfs", device=device,
        max_batch_size=1, token_budget=64, prefill_chunk_size=64,
        max_kv_tokens=max_kv_tokens, sliding_window=None, seed=seed,
    )

    for ws in window_sizes:
        workload = make_single_request_workload(
            vocab_size=vocab_size, prompt_len=prompt_len, max_new=max_new, seed=seed
        )
        results[ws] = run_sliding_window_engine(
            model, workload, policy="fcfs", device=device,
            max_batch_size=1, token_budget=64, prefill_chunk_size=64,
            max_kv_tokens=max_kv_tokens, sliding_window=ws, seed=seed,
        )

    print_quality_sweep_results(results)
    return results
