"""
Single-request vs continuous-batching benchmark for a nanoGPT-style model.

This is a benchmark scaffold, not a full serving engine. It compares:

1. Sequential single-request serving:
   - serve each request to completion before starting the next
   - uses KV-cache decode

2. Continuous batching:
   - requests arrive over scheduler steps
   - active requests decode one token per engine step
   - finished requests leave the batch independently
   - newly arrived requests can join while others are still decoding

Assumptions:
- Your model API is:
    logits, loss, new_kvs = model(idx, targets=None, pos=None, past_kvs=None)
- Batch cached forward works when every active request has the same cache length.

Important simplification:
- This benchmark keeps continuous batching simple by using uniform prompt lengths
  and uniform output lengths by default. That makes cache lengths stay aligned.
  Once this works, extend it to mixed lengths with padding masks.

Suggested use:
1. Paste/import this after your model is defined and trained/loaded.
2. Build a workload with `make_uniform_workload(...)`.
3. Call `run_single_vs_continuous_batching_benchmark(...)`.
"""

from dataclasses import dataclass, field
from statistics import mean
import time
import torch
import torch.nn.functional as F


@dataclass
class RequestSpec:
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_step: int = 0
    priority: int = 0


@dataclass
class RequestState:
    spec: RequestSpec
    generated_tokens: list[int] = field(default_factory=list)
    past_kvs: object = None
    last_token: torch.Tensor | None = None
    arrived_at_s: float | None = None
    admitted_at_s: float | None = None
    first_token_at_s: float | None = None
    completed_at_s: float | None = None

    @property
    def is_done(self) -> bool:
        return len(self.generated_tokens) >= self.spec.max_new_tokens


@dataclass
class StepMetrics:
    step: int
    batch_size: int
    forward_seconds: float
    step_seconds: float
    emitted_tokens: int


@dataclass
class RunMetrics:
    name: str
    total_requests: int
    total_generated_tokens: int
    total_seconds: float
    request_latencies_s: list[float]
    ttft_s: list[float]
    step_metrics: list[StepMetrics]

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
    def avg_batch_size(self) -> float:
        non_empty = [s.batch_size for s in self.step_metrics if s.batch_size > 0]
        return mean(non_empty) if non_empty else 0.0

    @property
    def max_batch_size(self) -> int:
        return max((s.batch_size for s in self.step_metrics), default=0)

    @property
    def forward_seconds(self) -> float:
        return sum(s.forward_seconds for s in self.step_metrics)


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sample_next_token(logits, temperature=1.0, generator=None):
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _stack_kvs(per_request_kvs):
    """
    Stack per-request KV caches into a batched KV cache.

    Expected per-request structure:
        past_kvs[layer][head] = (k, v)
        k/v shape: (1, T_cache, head_size)

    Returned structure:
        batched[layer][head] = (k, v)
        k/v shape: (B, T_cache, head_size)
    """
    n_layer = len(per_request_kvs[0])
    batched = []

    for layer_idx in range(n_layer):
        layer_kv = []
        n_head = len(per_request_kvs[0][layer_idx])

        for head_idx in range(n_head):
            keys = []
            values = []

            for req_kvs in per_request_kvs:
                k, v = req_kvs[layer_idx][head_idx]
                keys.append(k)
                values.append(v)

            layer_kv.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        batched.append(layer_kv)

    return batched


def _unstack_kvs(batched_kvs):
    """
    Split a batched KV cache back into per-request KV caches.
    """
    first_k, _ = batched_kvs[0][0]
    batch_size = first_k.shape[0]
    per_request = []

    for batch_idx in range(batch_size):
        req_kvs = []

        for layer_kv in batched_kvs:
            req_layer = []

            for k, v in layer_kv:
                req_layer.append((
                    k[batch_idx:batch_idx + 1].contiguous(),
                    v[batch_idx:batch_idx + 1].contiguous(),
                ))

            req_kvs.append(req_layer)

        per_request.append(req_kvs)

    return per_request


@torch.no_grad()
def _prefill_one(model, request, device, temperature, generator):
    prompt = torch.tensor(
        [request.spec.prompt_tokens],
        dtype=torch.long,
        device=device,
    )
    prompt_len = prompt.shape[1]
    positions = torch.arange(prompt_len, device=device).unsqueeze(0)

    logits, _, past_kvs = model(prompt, pos=positions)
    next_token = _sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        generator=generator,
    )

    request.past_kvs = past_kvs
    request.last_token = next_token
    request.generated_tokens.append(int(next_token.item()))


@torch.no_grad()
def _decode_one(model, request, device, temperature, generator):
    cache_len = request.past_kvs[0][0][0].shape[1]
    pos = torch.tensor([[cache_len]], device=device)

    logits, _, new_kvs = model(
        request.last_token,
        pos=pos,
        past_kvs=request.past_kvs,
    )
    next_token = _sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        generator=generator,
    )

    request.past_kvs = new_kvs
    request.last_token = next_token
    request.generated_tokens.append(int(next_token.item()))


@torch.no_grad()
def run_sequential_single_request(
    model,
    workload,
    *,
    device,
    temperature=1.0,
    seed=1337,
):
    """
    Serve each request to completion before serving the next request.
    """
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    states = [RequestState(spec=req) for req in workload]
    step_metrics = []

    _sync_if_cuda()
    run_start = time.perf_counter()

    emitted_total = 0
    step = 0

    for request in states:
        request.arrived_at_s = time.perf_counter()
        request.admitted_at_s = request.arrived_at_s

        while not request.is_done:
            step_start = time.perf_counter()
            forward_start = time.perf_counter()

            if request.past_kvs is None:
                _prefill_one(model, request, device, temperature, generator)
            else:
                _decode_one(model, request, device, temperature, generator)

            _sync_if_cuda()
            forward_seconds = time.perf_counter() - forward_start

            if request.first_token_at_s is None:
                request.first_token_at_s = time.perf_counter()

            emitted_total += 1
            step_metrics.append(StepMetrics(
                step=step,
                batch_size=1,
                forward_seconds=forward_seconds,
                step_seconds=time.perf_counter() - step_start,
                emitted_tokens=1,
            ))
            step += 1

        request.completed_at_s = time.perf_counter()

    _sync_if_cuda()
    total_seconds = time.perf_counter() - run_start

    latencies = [
        req.completed_at_s - req.arrived_at_s
        for req in states
    ]
    ttfts = [
        req.first_token_at_s - req.arrived_at_s
        for req in states
    ]

    return RunMetrics(
        name="single_request_sequential",
        total_requests=len(states),
        total_generated_tokens=emitted_total,
        total_seconds=total_seconds,
        request_latencies_s=latencies,
        ttft_s=ttfts,
        step_metrics=step_metrics,
    )


@torch.no_grad()
def run_continuous_batching(
    model,
    workload,
    *,
    device,
    max_batch_size=8,
    temperature=1.0,
    seed=1337,
):
    """
    Continuous batching benchmark with simple per-step arrivals.

    This version admits newly arrived requests, prefills them individually, then
    batches active decode requests together. That keeps the benchmark easy to
    understand while still measuring the core win: multiple active requests share
    one decode forward pass.
    """
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    pending = [RequestState(spec=req) for req in sorted(workload, key=lambda r: r.arrival_step)]
    active = []
    completed = []
    step_metrics = []

    _sync_if_cuda()
    run_start = time.perf_counter()

    step = 0
    emitted_total = 0

    while pending or active:
        step_start = time.perf_counter()

        # Admit arrivals for this scheduler step.
        newly_arrived = []
        while pending and pending[0].spec.arrival_step <= step:
            req = pending.pop(0)
            now = time.perf_counter()
            req.arrived_at_s = now
            req.admitted_at_s = now
            newly_arrived.append(req)

        # Prefill newly admitted requests until the active batch is full.
        # This intentionally keeps prefill simple for the first batching benchmark.
        prefill_forward_seconds = 0.0
        for req in newly_arrived:
            if len(active) >= max_batch_size:
                pending.insert(0, req)
                break

            forward_start = time.perf_counter()
            _prefill_one(model, req, device, temperature, generator)
            _sync_if_cuda()
            prefill_forward_seconds += time.perf_counter() - forward_start

            req.first_token_at_s = time.perf_counter()
            emitted_total += 1

            if req.is_done:
                req.completed_at_s = time.perf_counter()
                completed.append(req)
            else:
                active.append(req)

        decode_forward_seconds = 0.0
        decode_emitted = 0

        if active:
            batch = active[:max_batch_size]

            # This simple benchmark assumes aligned cache lengths. If this
            # assertion trips, use a uniform workload first or add padding masks.
            cache_lengths = [req.past_kvs[0][0][0].shape[1] for req in batch]
            assert len(set(cache_lengths)) == 1, (
                "Active requests have different cache lengths. Use uniform "
                "prompt/output settings for this first benchmark, or extend "
                "the benchmark with padded batched KV caches."
            )

            input_tokens = torch.cat([req.last_token for req in batch], dim=0)
            cache_len = cache_lengths[0]
            positions = torch.full(
                (len(batch), 1),
                cache_len,
                dtype=torch.long,
                device=device,
            )
            batched_kvs = _stack_kvs([req.past_kvs for req in batch])

            forward_start = time.perf_counter()
            logits, _, new_batched_kvs = model(
                input_tokens,
                pos=positions,
                past_kvs=batched_kvs,
            )
            next_tokens = _sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                generator=generator,
            )
            _sync_if_cuda()
            decode_forward_seconds = time.perf_counter() - forward_start

            split_kvs = _unstack_kvs(new_batched_kvs)

            still_active = []
            batch_ids = {id(req) for req in batch}

            for i, req in enumerate(batch):
                token = next_tokens[i:i + 1]
                req.past_kvs = split_kvs[i]
                req.last_token = token
                req.generated_tokens.append(int(token.item()))
                decode_emitted += 1
                emitted_total += 1

                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    completed.append(req)
                else:
                    still_active.append(req)

            # Preserve any active requests that were beyond max_batch_size.
            overflow = [req for req in active if id(req) not in batch_ids]
            active = still_active + overflow

        step_metrics.append(StepMetrics(
            step=step,
            batch_size=len(active),
            forward_seconds=prefill_forward_seconds + decode_forward_seconds,
            step_seconds=time.perf_counter() - step_start,
            emitted_tokens=len(newly_arrived) + decode_emitted,
        ))

        step += 1

    _sync_if_cuda()
    total_seconds = time.perf_counter() - run_start

    latencies = [
        req.completed_at_s - req.arrived_at_s
        for req in completed
    ]
    ttfts = [
        req.first_token_at_s - req.arrived_at_s
        for req in completed
    ]

    return RunMetrics(
        name="continuous_batching",
        total_requests=len(completed),
        total_generated_tokens=emitted_total,
        total_seconds=total_seconds,
        request_latencies_s=latencies,
        ttft_s=ttfts,
        step_metrics=step_metrics,
    )


def make_uniform_workload(
    *,
    num_requests,
    prompt_len,
    max_new_tokens,
    vocab_size,
    arrival_gap=0,
    seed=1337,
):
    """
    Workload for the first batching benchmark.

    Uniform prompt/output lengths make this much easier to reason about because
    active requests keep aligned cache lengths.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    workload = []
    for i in range(num_requests):
        prompt = torch.randint(
            low=0,
            high=vocab_size,
            size=(prompt_len,),
            generator=rng,
        ).tolist()

        workload.append(RequestSpec(
            id=i,
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
            arrival_step=i * arrival_gap,
            priority=0,
        ))

    return workload


def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def print_comparison_table(rows):
    headers = [
        "method",
        "reqs",
        "tokens",
        "wall_s",
        "tok/s",
        "avg_ttft_ms",
        "p95_ttft_ms",
        "avg_lat_ms",
        "p95_lat_ms",
        "avg_batch",
        "max_batch",
        "forward_s",
    ]

    rendered = []
    for row in rows:
        rendered.append([
            row.name,
            str(row.total_requests),
            str(row.total_generated_tokens),
            f"{row.total_seconds:.4f}",
            f"{row.tokens_per_second:.2f}",
            f"{row.avg_ttft_s * 1000:.2f}",
            f"{_percentile(row.ttft_s, 0.95) * 1000:.2f}",
            f"{row.avg_latency_s * 1000:.2f}",
            f"{_percentile(row.request_latencies_s, 0.95) * 1000:.2f}",
            f"{row.avg_batch_size:.2f}",
            str(row.max_batch_size),
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
        baseline, batched = rows
        speedup = batched.tokens_per_second / baseline.tokens_per_second
        latency_ratio = batched.avg_latency_s / baseline.avg_latency_s
        ttft_ratio = batched.avg_ttft_s / baseline.avg_ttft_s
        print()
        print(f"Continuous batching throughput speedup: {speedup:.2f}x")
        print(f"Average latency ratio: {latency_ratio:.2f}x")
        print(f"Average TTFT ratio: {ttft_ratio:.2f}x")


def run_single_vs_continuous_batching_benchmark(
    model,
    *,
    vocab_size,
    num_requests=16,
    prompt_len=8,
    max_new_tokens=32,
    max_batch_size=8,
    arrival_gap=0,
    device=None,
    seed=1337,
    temperature=1.0,
):
    """
    Run the second benchmark:
    - sequential single-request serving
    - continuous batching
    - comparison table

    Start with arrival_gap=0 to stress batching. Try arrival_gap=1 later to make
    arrivals more streaming-like.
    """
    if device is None:
        device = next(model.parameters()).device

    workload = make_uniform_workload(
        num_requests=num_requests,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        vocab_size=vocab_size,
        arrival_gap=arrival_gap,
        seed=seed,
    )

    sequential = run_sequential_single_request(
        model,
        workload,
        device=device,
        temperature=temperature,
        seed=seed,
    )

    batched = run_continuous_batching(
        model,
        workload,
        device=device,
        max_batch_size=max_batch_size,
        temperature=temperature,
        seed=seed,
    )

    print_comparison_table([sequential, batched])

    return {
        "single_request_sequential": sequential,
        "continuous_batching": batched,
    }


# Example call inside your nanoGPT script:
#
# run_single_vs_continuous_batching_benchmark(
#     m,
#     vocab_size=vocab_size,
#     num_requests=16,
#     prompt_len=8,
#     max_new_tokens=24,
#     max_batch_size=8,
#     arrival_gap=0,
#     device=device,
# )
