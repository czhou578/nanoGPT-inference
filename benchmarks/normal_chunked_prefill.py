"""
Normal prefill vs chunked prefill benchmark for a nanoGPT-style model.

This compares two serving policies:

1. Normal prefill:
   - when a request is admitted, run its whole prompt in one forward pass
   - only then continue decoding active requests
   - long prompts can create decode stalls

2. Chunked prefill:
   - active decode gets served first each engine step
   - remaining token budget is used to prefill prompt chunks
   - long prompts are split across steps

Assumptions:
- Your model API is:
    logits, loss, new_kvs = model(idx, targets=None, pos=None, past_kvs=None)
- Your model supports cached forward with explicit `pos`.
- This benchmark intentionally decodes active requests one by one, not as a
  batched decode. That keeps the experiment focused on prefill policy rather
  than continuous batching mechanics.

Suggested use:
1. Paste/import this after your model is defined and trained/loaded.
2. Call `run_normal_vs_chunked_prefill_benchmark(...)`.
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
    group: str = "default"


@dataclass
class RequestState:
    spec: RequestSpec
    generated_tokens: list[int] = field(default_factory=list)
    past_kvs: object = None
    last_token: torch.Tensor | None = None
    prefill_cursor: int = 0
    arrived_at_s: float | None = None
    admitted_at_s: float | None = None
    first_token_at_s: float | None = None
    completed_at_s: float | None = None
    decode_token_times_s: list[float] = field(default_factory=list)

    @property
    def is_prefill_done(self) -> bool:
        return self.prefill_cursor >= len(self.spec.prompt_tokens)

    @property
    def is_done(self) -> bool:
        return len(self.generated_tokens) >= self.spec.max_new_tokens


@dataclass
class StepMetrics:
    step: int
    waiting: int
    prefilling: int
    active: int
    prefill_tokens: int
    decode_tokens: int
    forward_seconds: float
    step_seconds: float


@dataclass
class RunMetrics:
    name: str
    total_requests: int
    total_prompt_tokens: int
    total_generated_tokens: int
    total_seconds: float
    request_latencies_s: list[float]
    ttft_s: list[float]
    inter_token_gaps_s: list[float]
    step_metrics: list[StepMetrics]

    @property
    def generated_tokens_per_second(self) -> float:
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_generated_tokens / self.total_seconds

    @property
    def prompt_tokens_per_second(self) -> float:
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_prompt_tokens / self.total_seconds

    @property
    def avg_latency_s(self) -> float:
        return mean(self.request_latencies_s) if self.request_latencies_s else 0.0

    @property
    def avg_ttft_s(self) -> float:
        return mean(self.ttft_s) if self.ttft_s else 0.0

    @property
    def avg_inter_token_gap_s(self) -> float:
        return mean(self.inter_token_gaps_s) if self.inter_token_gaps_s else 0.0

    @property
    def max_inter_token_gap_s(self) -> float:
        return max(self.inter_token_gaps_s, default=0.0)

    @property
    def forward_seconds(self) -> float:
        return sum(s.forward_seconds for s in self.step_metrics)


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sample_next_token(logits, temperature=1.0, generator=None):
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _record_generated_token(req, token_tensor):
    now = time.perf_counter()
    req.last_token = token_tensor
    req.generated_tokens.append(int(token_tensor.item()))
    req.decode_token_times_s.append(now)

    if req.first_token_at_s is None:
        req.first_token_at_s = now


@torch.no_grad()
def _prefill_full(model, req, device, temperature, generator):
    prompt = torch.tensor(
        [req.spec.prompt_tokens],
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

    req.prefill_cursor = prompt_len
    req.past_kvs = past_kvs
    _record_generated_token(req, next_token)

    return prompt_len, 1


@torch.no_grad()
def _prefill_chunk(model, req, chunk_size, device, temperature, generator):
    start = req.prefill_cursor
    end = min(start + chunk_size, len(req.spec.prompt_tokens))
    chunk_tokens = req.spec.prompt_tokens[start:end]

    idx = torch.tensor([chunk_tokens], dtype=torch.long, device=device)
    positions = torch.arange(start, end, device=device).unsqueeze(0)

    logits, _, new_kvs = model(
        idx,
        pos=positions,
        past_kvs=req.past_kvs,
    )

    req.prefill_cursor = end
    req.past_kvs = new_kvs

    emitted = 0
    if req.is_prefill_done:
        next_token = _sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            generator=generator,
        )
        _record_generated_token(req, next_token)
        emitted = 1

    return len(chunk_tokens), emitted


@torch.no_grad()
def _decode_one(model, req, device, temperature, generator):
    cache_len = req.past_kvs[0][0][0].shape[1]
    pos = torch.tensor([[cache_len]], device=device)

    logits, _, new_kvs = model(
        req.last_token,
        pos=pos,
        past_kvs=req.past_kvs,
    )
    next_token = _sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        generator=generator,
    )

    req.past_kvs = new_kvs
    _record_generated_token(req, next_token)
    return 1


def _arrive_requests(pending, step):
    arrived = []
    while pending and pending[0].spec.arrival_step <= step:
        req = pending.pop(0)
        now = time.perf_counter()
        req.arrived_at_s = now
        req.admitted_at_s = now
        arrived.append(req)
    return arrived


def _finalize_metrics(name, states, step_metrics, run_start, run_end):
    total_prompt_tokens = sum(len(s.spec.prompt_tokens) for s in states)
    total_generated_tokens = sum(len(s.generated_tokens) for s in states)

    latencies = [
        s.completed_at_s - s.arrived_at_s
        for s in states
        if s.completed_at_s is not None and s.arrived_at_s is not None
    ]
    ttfts = [
        s.first_token_at_s - s.arrived_at_s
        for s in states
        if s.first_token_at_s is not None and s.arrived_at_s is not None
    ]

    inter_token_gaps = []
    for s in states:
        times = s.decode_token_times_s
        for i in range(1, len(times)):
            inter_token_gaps.append(times[i] - times[i - 1])

    return RunMetrics(
        name=name,
        total_requests=len(states),
        total_prompt_tokens=total_prompt_tokens,
        total_generated_tokens=total_generated_tokens,
        total_seconds=run_end - run_start,
        request_latencies_s=latencies,
        ttft_s=ttfts,
        inter_token_gaps_s=inter_token_gaps,
        step_metrics=step_metrics,
    )


@torch.no_grad()
def run_normal_prefill_policy(
    model,
    workload,
    *,
    device,
    temperature=1.0,
    seed=1337,
):
    """
    Full-prompt prefill policy.

    New arrivals are fully prefilled before active decode gets another turn.
    This is intentionally harsh so decode starvation is visible.
    """
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    pending = [RequestState(spec=req) for req in sorted(workload, key=lambda r: r.arrival_step)]
    all_states = list(pending)
    active = []
    completed = []
    step_metrics = []

    _sync_if_cuda()
    run_start = time.perf_counter()
    step = 0

    while pending or active:
        step_start = time.perf_counter()
        prefill_tokens = 0
        decode_tokens = 0
        forward_seconds = 0.0

        arrivals = _arrive_requests(pending, step)

        for req in arrivals:
            fwd_start = time.perf_counter()
            p_tokens, emitted = _prefill_full(model, req, device, temperature, generator)
            _sync_if_cuda()
            forward_seconds += time.perf_counter() - fwd_start

            prefill_tokens += p_tokens
            decode_tokens += emitted

            if req.is_done:
                req.completed_at_s = time.perf_counter()
                completed.append(req)
            else:
                active.append(req)

        still_active = []
        for req in active:
            if req.is_done:
                req.completed_at_s = time.perf_counter()
                completed.append(req)
                continue

            fwd_start = time.perf_counter()
            decode_tokens += _decode_one(model, req, device, temperature, generator)
            _sync_if_cuda()
            forward_seconds += time.perf_counter() - fwd_start

            if req.is_done:
                req.completed_at_s = time.perf_counter()
                completed.append(req)
            else:
                still_active.append(req)

        active = still_active

        step_metrics.append(StepMetrics(
            step=step,
            waiting=len(pending),
            prefilling=0,
            active=len(active),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            forward_seconds=forward_seconds,
            step_seconds=time.perf_counter() - step_start,
        ))

        step += 1

    _sync_if_cuda()
    run_end = time.perf_counter()
    return _finalize_metrics("normal_prefill", all_states, step_metrics, run_start, run_end)


@torch.no_grad()
def run_chunked_prefill_policy(
    model,
    workload,
    *,
    device,
    token_budget=16,
    chunk_size=8,
    temperature=1.0,
    seed=1337,
):
    """
    Chunked prefill policy.

    Each step:
    1. admit arrivals into a prefilling queue
    2. decode active requests first
    3. spend remaining token budget on prompt chunks
    """
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    pending = [RequestState(spec=req) for req in sorted(workload, key=lambda r: r.arrival_step)]
    all_states = list(pending)
    prefilling = []
    active = []
    completed = []
    step_metrics = []

    _sync_if_cuda()
    run_start = time.perf_counter()
    step = 0

    while pending or prefilling or active:
        step_start = time.perf_counter()
        prefill_tokens = 0
        decode_tokens = 0
        forward_seconds = 0.0
        tokens_used = 0

        prefilling.extend(_arrive_requests(pending, step))

        still_active = []
        for req in active:
            if tokens_used >= token_budget:
                still_active.append(req)
                continue

            fwd_start = time.perf_counter()
            decode_tokens += _decode_one(model, req, device, temperature, generator)
            _sync_if_cuda()
            forward_seconds += time.perf_counter() - fwd_start
            tokens_used += 1

            if req.is_done:
                req.completed_at_s = time.perf_counter()
                completed.append(req)
            else:
                still_active.append(req)

        active = still_active

        # Spend remaining budget on prefill chunks. This loop may prefill more
        # than one request in a step if the budget allows it.
        new_prefilling = []
        while prefilling and tokens_used < token_budget:
            req = prefilling.pop(0)
            remaining_budget = token_budget - tokens_used
            this_chunk = min(chunk_size, remaining_budget)

            fwd_start = time.perf_counter()
            p_tokens, emitted = _prefill_chunk(
                model,
                req,
                this_chunk,
                device,
                temperature,
                generator,
            )
            _sync_if_cuda()
            forward_seconds += time.perf_counter() - fwd_start

            prefill_tokens += p_tokens
            decode_tokens += emitted
            tokens_used += p_tokens

            if req.is_prefill_done:
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    completed.append(req)
                else:
                    active.append(req)
            else:
                new_prefilling.append(req)

        prefilling = new_prefilling + prefilling

        step_metrics.append(StepMetrics(
            step=step,
            waiting=len(pending),
            prefilling=len(prefilling),
            active=len(active),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            forward_seconds=forward_seconds,
            step_seconds=time.perf_counter() - step_start,
        ))

        step += 1

    _sync_if_cuda()
    run_end = time.perf_counter()
    return _finalize_metrics("chunked_prefill", all_states, step_metrics, run_start, run_end)


def make_prefill_stress_workload(
    *,
    vocab_size,
    num_short_requests=4,
    short_prompt_len=8,
    short_max_new_tokens=48,
    num_long_requests=4,
    long_prompt_len=32,
    long_max_new_tokens=8,
    long_arrival_step=2,
    seed=1337,
):
    """
    Workload designed to reveal decode starvation.

    Short chat-like requests arrive first and start decoding. Then long-prompt
    requests arrive while those short requests are active.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    requests = []
    next_id = 0

    for _ in range(num_short_requests):
        prompt = torch.randint(0, vocab_size, (short_prompt_len,), generator=rng).tolist()
        requests.append(RequestSpec(
            id=next_id,
            prompt_tokens=prompt,
            max_new_tokens=short_max_new_tokens,
            arrival_step=0,
            group="short_decode_heavy",
        ))
        next_id += 1

    for _ in range(num_long_requests):
        prompt = torch.randint(0, vocab_size, (long_prompt_len,), generator=rng).tolist()
        requests.append(RequestSpec(
            id=next_id,
            prompt_tokens=prompt,
            max_new_tokens=long_max_new_tokens,
            arrival_step=long_arrival_step,
            group="long_prefill_heavy",
        ))
        next_id += 1

    return sorted(requests, key=lambda r: (r.arrival_step, r.id))


def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def print_prefill_comparison_table(rows):
    headers = [
        "method",
        "reqs",
        "prompt_tok",
        "gen_tok",
        "wall_s",
        "gen_tok/s",
        "prompt_tok/s",
        "avg_ttft_ms",
        "p95_ttft_ms",
        "avg_lat_ms",
        "p95_lat_ms",
        "avg_gap_ms",
        "max_gap_ms",
        "forward_s",
    ]

    rendered = []
    for row in rows:
        rendered.append([
            row.name,
            str(row.total_requests),
            str(row.total_prompt_tokens),
            str(row.total_generated_tokens),
            f"{row.total_seconds:.4f}",
            f"{row.generated_tokens_per_second:.2f}",
            f"{row.prompt_tokens_per_second:.2f}",
            f"{row.avg_ttft_s * 1000:.2f}",
            f"{_percentile(row.ttft_s, 0.95) * 1000:.2f}",
            f"{row.avg_latency_s * 1000:.2f}",
            f"{_percentile(row.request_latencies_s, 0.95) * 1000:.2f}",
            f"{row.avg_inter_token_gap_s * 1000:.2f}",
            f"{row.max_inter_token_gap_s * 1000:.2f}",
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
        normal, chunked = rows
        throughput_ratio = chunked.generated_tokens_per_second / normal.generated_tokens_per_second
        ttft_ratio = chunked.avg_ttft_s / normal.avg_ttft_s
        max_gap_ratio = chunked.max_inter_token_gap_s / normal.max_inter_token_gap_s
        print()
        print(f"Chunked prefill generated-token throughput ratio: {throughput_ratio:.2f}x")
        print(f"Average TTFT ratio: {ttft_ratio:.2f}x")
        print(f"Max inter-token gap ratio: {max_gap_ratio:.2f}x")


def run_normal_vs_chunked_prefill_benchmark(
    model,
    *,
    vocab_size,
    num_short_requests=4,
    short_prompt_len=8,
    short_max_new_tokens=48,
    num_long_requests=4,
    long_prompt_len=32,
    long_max_new_tokens=8,
    long_arrival_step=2,
    token_budget=16,
    chunk_size=8,
    device=None,
    seed=1337,
    temperature=1.0,
):
    """
    Run the third benchmark:
    - normal full-prompt prefill
    - chunked prefill with decode-first token budgeting
    - comparison table

    Keep `long_prompt_len` within your model's valid position range unless you
    have already extended/fixed long-context position handling.
    """
    if device is None:
        device = next(model.parameters()).device

    workload = make_prefill_stress_workload(
        vocab_size=vocab_size,
        num_short_requests=num_short_requests,
        short_prompt_len=short_prompt_len,
        short_max_new_tokens=short_max_new_tokens,
        num_long_requests=num_long_requests,
        long_prompt_len=long_prompt_len,
        long_max_new_tokens=long_max_new_tokens,
        long_arrival_step=long_arrival_step,
        seed=seed,
    )

    normal = run_normal_prefill_policy(
        model,
        workload,
        device=device,
        temperature=temperature,
        seed=seed,
    )

    chunked = run_chunked_prefill_policy(
        model,
        workload,
        device=device,
        token_budget=token_budget,
        chunk_size=chunk_size,
        temperature=temperature,
        seed=seed,
    )

    print_prefill_comparison_table([normal, chunked])

    return {
        "normal_prefill": normal,
        "chunked_prefill": chunked,
    }


# Example call inside your nanoGPT script:
#
# run_normal_vs_chunked_prefill_benchmark(
#     m,
#     vocab_size=vocab_size,
#     num_short_requests=4,
#     short_prompt_len=8,
#     short_max_new_tokens=48,
#     num_long_requests=4,
#     long_prompt_len=32,
#     long_max_new_tokens=8,
#     long_arrival_step=2,
#     token_budget=16,
#     chunk_size=8,
#     device=device,
# )


# run_single_vs_continuous_batching_benchmark(
#     model,
#     vocab_size=vocab_size,
#     num_requests=32,
#     prompt_len=8,
#     max_new_tokens=24,
#     max_batch_size=8,
#     arrival_gap=0,
#     device=device,
# )

# --------------------------+------+--------+--------+---------+-------------+-------------+------------+------------+-----------+-----------+----------
# single_request_sequential | 32   | 768    | 1.7119 | 448.63  | 2.80        | 3.22        | 53.50      | 59.48      | 1.00      | 1         | 1.7101        
# continuous_batching       | 9    | 216    | 0.1452 | 1487.26 | 10.87       | 20.99       | 86.44      | 90.43      | 4.50      | 8         | 0.1238        

# Continuous batching throughput speedup: 3.32x
# Average latency ratio: 1.62x
# Average TTFT ratio: 3.88x
