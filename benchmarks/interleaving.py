"""
Decode/prefill interleaving benchmark for nanogpt-interleaving.py.

This benchmark is standalone so importing it does not train the NanoGPT script.
It assumes the model API used by nanogpt-interleaving.py:

    logits, loss, new_kvs = model(idx, targets=None, pos=None,
                                  past_kvs=None, attn_mask=None)

The benchmark compares:
- separate_calls: decode and prefill are scheduled in the same step, but run
  as separate model forwards when both kinds of work are present
- interleaved_fused: decode rows and one prefill chunk are packed into one
  mixed model forward when possible

The fused path uses right-padding for mixed rows and keeps only each row's real
new KV entries after the forward pass. This avoids requiring an input_mask
argument in the NanoGPT model.
"""

from dataclasses import dataclass, field
from statistics import mean
import time
import torch
import torch.nn.functional as F


@dataclass
class InterleaveRequestSpec:
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_step: int = 0
    group: str = "default"


@dataclass
class InterleaveRequestState:
    spec: InterleaveRequestSpec
    generated_tokens: list[int] = field(default_factory=list)
    past_kvs: object = None
    last_token: torch.Tensor | None = None
    prefill_cursor: int = 0
    arrived_at_s: float | None = None
    first_token_at_s: float | None = None
    completed_at_s: float | None = None
    token_times_s: list[float] = field(default_factory=list)

    @property
    def is_prefill_done(self) -> bool:
        return self.prefill_cursor >= len(self.spec.prompt_tokens)

    @property
    def is_done(self) -> bool:
        return len(self.generated_tokens) >= self.spec.max_new_tokens

    @property
    def cache_len(self) -> int:
        if self.past_kvs is None:
            return 0
        return self.past_kvs[0][0][0].shape[1]


@dataclass
class InterleaveStepMetrics:
    step: int
    waiting: int
    prefilling: int
    active: int
    prefill_tokens: int
    decode_tokens: int
    mixed_rows: int
    forward_calls: int
    forward_seconds: float
    step_seconds: float


@dataclass
class InterleaveRunMetrics:
    name: str
    total_requests: int
    completed_requests: int
    total_prompt_tokens: int
    total_generated_tokens: int
    total_seconds: float
    request_latencies_s: list[float]
    ttft_s: list[float]
    inter_token_gaps_s: list[float]
    step_metrics: list[InterleaveStepMetrics]

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
        return max(self.inter_token_gaps_s) if self.inter_token_gaps_s else 0.0

    @property
    def total_forward_calls(self) -> int:
        return sum(s.forward_calls for s in self.step_metrics)

    @property
    def mixed_forward_steps(self) -> int:
        return sum(1 for s in self.step_metrics if s.mixed_rows > 1)

    @property
    def avg_decode_batch_size(self) -> float:
        values = [s.decode_tokens for s in self.step_metrics if s.decode_tokens > 0]
        return mean(values) if values else 0.0

    @property
    def forward_seconds(self) -> float:
        return sum(s.forward_seconds for s in self.step_metrics)


def _make_generator(device, seed):
    if str(device).startswith("cuda"):
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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


def _arrive_requests(pending, step):
    arrivals = []
    while pending and pending[0].spec.arrival_step <= step:
        req = pending.pop(0)
        req.arrived_at_s = time.perf_counter()
        arrivals.append(req)
    return arrivals


def _stack_request_kvs(requests, device):
    lengths = [req.cache_len for req in requests]
    max_len = max(lengths) if lengths else 0
    pad_lengths = [max_len - length for length in lengths]

    if max_len == 0:
        return None, None, pad_lengths

    template = next(req.past_kvs for req in requests if req.past_kvs is not None)
    n_layer = len(template)
    batched = []

    attn_mask = torch.zeros(
        (len(requests), 1, max_len),
        dtype=torch.bool,
        device=device,
    )
    for i, pad in enumerate(pad_lengths):
        attn_mask[i, :, pad:] = True

    for layer_idx in range(n_layer):
        layer = []
        n_head = len(template[layer_idx])
        for head_idx in range(n_head):
            keys = []
            values = []
            for req, pad in zip(requests, pad_lengths):
                if req.past_kvs is None:
                    template_k, template_v = template[layer_idx][head_idx]
                    hs = template_k.shape[-1]
                    dtype = template_k.dtype
                    k = torch.empty((1, 0, hs), dtype=dtype, device=device)
                    v = torch.empty((1, 0, hs), dtype=template_v.dtype, device=device)
                else:
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


def _unstack_request_kvs(requests, batched_kvs, pad_lengths, real_new_tokens):
    for req_idx, req in enumerate(requests):
        keep_start = pad_lengths[req_idx]
        keep_end = keep_start + req.cache_len + real_new_tokens[req_idx]

        req_kvs = []
        for layer in batched_kvs:
            req_layer = []
            for k, v in layer:
                req_layer.append((
                    k[req_idx:req_idx + 1, keep_start:keep_end, :].contiguous(),
                    v[req_idx:req_idx + 1, keep_start:keep_end, :].contiguous(),
                ))
            req_kvs.append(req_layer)
        req.past_kvs = req_kvs


@torch.no_grad()
def _prefill_chunk(model, req, chunk_size, device, temperature, generator):
    start = req.prefill_cursor
    end = min(start + chunk_size, len(req.spec.prompt_tokens))
    tokens = req.spec.prompt_tokens[start:end]

    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    pos = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0)

    logits, _, req.past_kvs = model(idx, pos=pos, past_kvs=req.past_kvs)
    req.prefill_cursor = end

    emitted = 0
    if req.is_prefill_done and not req.is_done:
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

    idx = torch.cat([req.last_token for req in requests], dim=0)
    pos = torch.tensor(
        [[len(req.spec.prompt_tokens) + len(req.generated_tokens) - 1] for req in requests],
        dtype=torch.long,
        device=device,
    )
    past_kvs, attn_mask, pad_lengths = _stack_request_kvs(requests, device)

    logits, _, new_kvs = model(
        idx,
        pos=pos,
        past_kvs=past_kvs,
        attn_mask=attn_mask,
    )

    _unstack_request_kvs(requests, new_kvs, pad_lengths, [1] * len(requests))

    next_tokens = _sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        generator=generator,
    )

    for i, req in enumerate(requests):
        _record_generated_token(req, next_tokens[i:i + 1])

    return len(requests)


def _build_fused_inputs(decode_reqs, prefill_req, prefill_chunk_size, device):
    requests = list(decode_reqs)
    real_new_tokens = [1] * len(decode_reqs)
    row_roles = ["decode"] * len(decode_reqs)

    prefill_start = None
    prefill_end = None
    if prefill_req is not None and prefill_chunk_size > 0:
        prefill_start = prefill_req.prefill_cursor
        prefill_end = min(
            prefill_start + prefill_chunk_size,
            len(prefill_req.spec.prompt_tokens),
        )
        if prefill_end > prefill_start:
            requests.append(prefill_req)
            real_new_tokens.append(prefill_end - prefill_start)
            row_roles.append("prefill")

    if not requests:
        return None

    t_max = max(real_new_tokens)
    batch_tokens = []
    batch_positions = []

    for req, n_new, role in zip(requests, real_new_tokens, row_roles):
        if role == "decode":
            tokens = [int(req.last_token.item())]
            positions = [len(req.spec.prompt_tokens) + len(req.generated_tokens) - 1]
        else:
            tokens = req.spec.prompt_tokens[prefill_start:prefill_end]
            positions = list(range(prefill_start, prefill_end))

        pad = t_max - n_new
        batch_tokens.append(tokens + [0] * pad)
        batch_positions.append(positions + [0] * pad)

    idx = torch.tensor(batch_tokens, dtype=torch.long, device=device)
    pos = torch.tensor(batch_positions, dtype=torch.long, device=device)
    past_kvs, attn_mask, pad_lengths = _stack_request_kvs(requests, device)

    return {
        "requests": requests,
        "row_roles": row_roles,
        "real_new_tokens": real_new_tokens,
        "idx": idx,
        "pos": pos,
        "past_kvs": past_kvs,
        "attn_mask": attn_mask,
        "pad_lengths": pad_lengths,
    }


@torch.no_grad()
def _run_fused_step(
    model,
    decode_reqs,
    prefill_req,
    prefill_chunk_size,
    device,
    temperature,
    generator,
):
    batch = _build_fused_inputs(decode_reqs, prefill_req, prefill_chunk_size, device)
    if batch is None:
        return 0, 0, 0

    logits, _, new_kvs = model(
        batch["idx"],
        pos=batch["pos"],
        past_kvs=batch["past_kvs"],
        attn_mask=batch["attn_mask"],
    )

    _unstack_request_kvs(
        batch["requests"],
        new_kvs,
        batch["pad_lengths"],
        batch["real_new_tokens"],
    )

    prefill_tokens = 0
    decode_tokens = 0

    for row_idx, (req, role, n_new) in enumerate(zip(
        batch["requests"],
        batch["row_roles"],
        batch["real_new_tokens"],
    )):
        last_real_idx = n_new - 1
        next_token = _sample_next_token(
            logits[row_idx:row_idx + 1, last_real_idx, :],
            temperature=temperature,
            generator=generator,
        )

        if role == "decode":
            _record_generated_token(req, next_token)
            decode_tokens += 1
        else:
            req.prefill_cursor += n_new
            prefill_tokens += n_new
            if req.is_prefill_done and not req.is_done:
                _record_generated_token(req, next_token)
                decode_tokens += 1

    return prefill_tokens, decode_tokens, len(batch["requests"])


def _finalize_metrics(name, all_states, completed, step_metrics, run_start, run_end):
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
    gaps = []
    for req in all_states:
        gaps.extend(
            later - earlier
            for earlier, later in zip(req.token_times_s, req.token_times_s[1:])
        )

    return InterleaveRunMetrics(
        name=name,
        total_requests=len(all_states),
        completed_requests=len(completed),
        total_prompt_tokens=sum(len(req.spec.prompt_tokens) for req in all_states),
        total_generated_tokens=sum(len(req.generated_tokens) for req in all_states),
        total_seconds=run_end - run_start,
        request_latencies_s=latencies,
        ttft_s=ttfts,
        inter_token_gaps_s=gaps,
        step_metrics=step_metrics,
    )


@torch.no_grad()
def run_separate_calls_policy(
    model,
    workload,
    *,
    device,
    max_batch_size=4,
    token_budget=16,
    chunk_size=8,
    temperature=1.0,
    seed=1337,
    max_steps=10000,
):
    """
    Decode-first chunked prefill where decode and prefill use separate forwards.
    """
    model.eval()
    generator = _make_generator(device, seed)

    pending = [
        InterleaveRequestState(spec=req)
        for req in sorted(workload, key=lambda r: (r.arrival_step, r.id))
    ]
    all_states = list(pending)
    prefilling = []
    active = []
    completed = []
    step_metrics = []

    _sync_if_cuda(device)
    run_start = time.perf_counter()
    step = 0

    while pending or prefilling or active:
        if step > max_steps:
            raise RuntimeError(f"Interleaving benchmark exceeded max_steps={max_steps}")

        step_start = time.perf_counter()
        prefill_tokens = 0
        decode_tokens = 0
        forward_calls = 0
        forward_seconds = 0.0
        tokens_used = 0

        prefilling.extend(_arrive_requests(pending, step))

        decode_batch = active[:min(max_batch_size, token_budget)]
        if decode_batch:
            fwd_start = time.perf_counter()
            decoded = _decode_batch(model, decode_batch, device, temperature, generator)
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start
            forward_calls += 1
            decode_tokens += decoded
            tokens_used += decoded

            for req in list(active):
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    active.remove(req)
                    completed.append(req)

        if prefilling and tokens_used < token_budget:
            req = prefilling[0]
            this_chunk = min(chunk_size, token_budget - tokens_used)

            fwd_start = time.perf_counter()
            p_tokens, emitted = _prefill_chunk(
                model,
                req,
                this_chunk,
                device,
                temperature,
                generator,
            )
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start
            forward_calls += 1
            prefill_tokens += p_tokens
            decode_tokens += emitted
            tokens_used += p_tokens

            if req.is_prefill_done:
                prefilling.pop(0)
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    completed.append(req)
                else:
                    active.append(req)

        step_metrics.append(InterleaveStepMetrics(
            step=step,
            waiting=len(pending),
            prefilling=len(prefilling),
            active=len(active),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            mixed_rows=len(decode_batch) + (1 if prefill_tokens else 0),
            forward_calls=forward_calls,
            forward_seconds=forward_seconds,
            step_seconds=time.perf_counter() - step_start,
        ))

        step += 1

    _sync_if_cuda(device)
    run_end = time.perf_counter()
    return _finalize_metrics("separate_calls", all_states, completed, step_metrics, run_start, run_end)


@torch.no_grad()
def run_interleaved_fused_policy(
    model,
    workload,
    *,
    device,
    max_batch_size=4,
    token_budget=16,
    chunk_size=8,
    temperature=1.0,
    seed=1337,
    max_steps=10000,
):
    """
    Decode-first chunked prefill where decode and prefill share one forward.
    """
    model.eval()
    generator = _make_generator(device, seed)

    pending = [
        InterleaveRequestState(spec=req)
        for req in sorted(workload, key=lambda r: (r.arrival_step, r.id))
    ]
    all_states = list(pending)
    prefilling = []
    active = []
    completed = []
    step_metrics = []

    _sync_if_cuda(device)
    run_start = time.perf_counter()
    step = 0

    while pending or prefilling or active:
        if step > max_steps:
            raise RuntimeError(f"Interleaving benchmark exceeded max_steps={max_steps}")

        step_start = time.perf_counter()
        prefill_tokens = 0
        decode_tokens = 0
        forward_calls = 0
        forward_seconds = 0.0

        prefilling.extend(_arrive_requests(pending, step))

        decode_batch = active[:min(max_batch_size, token_budget)]
        remaining_budget = token_budget - len(decode_batch)
        prefill_req = prefilling[0] if prefilling and remaining_budget > 0 else None
        prefill_chunk = min(chunk_size, remaining_budget) if prefill_req is not None else 0

        if decode_batch or prefill_req is not None:
            fwd_start = time.perf_counter()
            p_tokens, d_tokens, mixed_rows = _run_fused_step(
                model,
                decode_batch,
                prefill_req,
                prefill_chunk,
                device,
                temperature,
                generator,
            )
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start
            forward_calls += 1
            prefill_tokens += p_tokens
            decode_tokens += d_tokens
        else:
            mixed_rows = 0

        for req in list(active):
            if req.is_done:
                req.completed_at_s = time.perf_counter()
                active.remove(req)
                completed.append(req)

        if prefill_req is not None and prefill_req.is_prefill_done:
            prefilling.pop(0)
            if prefill_req.is_done:
                prefill_req.completed_at_s = time.perf_counter()
                completed.append(prefill_req)
            else:
                active.append(prefill_req)

        step_metrics.append(InterleaveStepMetrics(
            step=step,
            waiting=len(pending),
            prefilling=len(prefilling),
            active=len(active),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            mixed_rows=mixed_rows,
            forward_calls=forward_calls,
            forward_seconds=forward_seconds,
            step_seconds=time.perf_counter() - step_start,
        ))

        step += 1

    _sync_if_cuda(device)
    run_end = time.perf_counter()
    return _finalize_metrics("interleaved_fused", all_states, completed, step_metrics, run_start, run_end)


def make_interleaving_workload(
    *,
    vocab_size,
    num_decode_heavy_requests=6,
    decode_prompt_len=8,
    decode_max_new_tokens=32,
    num_prefill_heavy_requests=3,
    prefill_prompt_len=32,
    prefill_max_new_tokens=8,
    prefill_arrival_step=2,
    stagger_prefill_arrivals=False,
    seed=1337,
):
    """
    Workload designed to create steps with both active decode and pending prefill.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    requests = []
    next_id = 0

    for _ in range(num_decode_heavy_requests):
        requests.append(InterleaveRequestSpec(
            id=next_id,
            prompt_tokens=torch.randint(0, vocab_size, (decode_prompt_len,), generator=rng).tolist(),
            max_new_tokens=decode_max_new_tokens,
            arrival_step=0,
            group="decode_heavy",
        ))
        next_id += 1

    for i in range(num_prefill_heavy_requests):
        arrival = prefill_arrival_step + i if stagger_prefill_arrivals else prefill_arrival_step
        requests.append(InterleaveRequestSpec(
            id=next_id,
            prompt_tokens=torch.randint(0, vocab_size, (prefill_prompt_len,), generator=rng).tolist(),
            max_new_tokens=prefill_max_new_tokens,
            arrival_step=arrival,
            group="prefill_heavy",
        ))
        next_id += 1

    return sorted(requests, key=lambda r: (r.arrival_step, r.id))


def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def print_interleaving_comparison_table(rows):
    headers = [
        "method",
        "reqs",
        "done",
        "prompt_tok",
        "gen_tok",
        "wall_s",
        "gen_tok/s",
        "prompt_tok/s",
        "fwd_calls",
        "mixed_steps",
        "avg_ttft_ms",
        "p95_ttft_ms",
        "avg_lat_ms",
        "p95_lat_ms",
        "avg_gap_ms",
        "max_gap_ms",
        "avg_decode_b",
        "forward_s",
    ]

    rendered = []
    for row in rows:
        rendered.append([
            row.name,
            str(row.total_requests),
            str(row.completed_requests),
            str(row.total_prompt_tokens),
            str(row.total_generated_tokens),
            f"{row.total_seconds:.4f}",
            f"{row.generated_tokens_per_second:.2f}",
            f"{row.prompt_tokens_per_second:.2f}",
            str(row.total_forward_calls),
            str(row.mixed_forward_steps),
            f"{row.avg_ttft_s * 1000:.2f}",
            f"{_percentile(row.ttft_s, 0.95) * 1000:.2f}",
            f"{row.avg_latency_s * 1000:.2f}",
            f"{_percentile(row.request_latencies_s, 0.95) * 1000:.2f}",
            f"{row.avg_inter_token_gap_s * 1000:.2f}",
            f"{row.max_inter_token_gap_s * 1000:.2f}",
            f"{row.avg_decode_batch_size:.2f}",
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
        separate, fused = rows
        print()
        print(
            "Interleaved generated-token throughput ratio: "
            f"{fused.generated_tokens_per_second / separate.generated_tokens_per_second:.2f}x"
        )
        print(
            "Forward-call ratio: "
            f"{fused.total_forward_calls / separate.total_forward_calls:.2f}x"
        )
        if separate.avg_ttft_s > 0:
            print(
                "Average TTFT ratio: "
                f"{fused.avg_ttft_s / separate.avg_ttft_s:.2f}x"
            )
        if separate.max_inter_token_gap_s > 0:
            print(
                "Max inter-token gap ratio: "
                f"{fused.max_inter_token_gap_s / separate.max_inter_token_gap_s:.2f}x"
            )


def run_separate_vs_interleaved_benchmark(
    model,
    *,
    vocab_size,
    num_decode_heavy_requests=6,
    decode_prompt_len=8,
    decode_max_new_tokens=32,
    num_prefill_heavy_requests=3,
    prefill_prompt_len=32,
    prefill_max_new_tokens=8,
    prefill_arrival_step=2,
    stagger_prefill_arrivals=False,
    max_batch_size=4,
    token_budget=16,
    chunk_size=8,
    device=None,
    seed=1337,
    temperature=1.0,
):
    if device is None:
        device = next(model.parameters()).device

    workload = make_interleaving_workload(
        vocab_size=vocab_size,
        num_decode_heavy_requests=num_decode_heavy_requests,
        decode_prompt_len=decode_prompt_len,
        decode_max_new_tokens=decode_max_new_tokens,
        num_prefill_heavy_requests=num_prefill_heavy_requests,
        prefill_prompt_len=prefill_prompt_len,
        prefill_max_new_tokens=prefill_max_new_tokens,
        prefill_arrival_step=prefill_arrival_step,
        stagger_prefill_arrivals=stagger_prefill_arrivals,
        seed=seed,
    )

    separate = run_separate_calls_policy(
        model,
        workload,
        device=device,
        max_batch_size=max_batch_size,
        token_budget=token_budget,
        chunk_size=chunk_size,
        temperature=temperature,
        seed=seed,
    )

    fused = run_interleaved_fused_policy(
        model,
        workload,
        device=device,
        max_batch_size=max_batch_size,
        token_budget=token_budget,
        chunk_size=chunk_size,
        temperature=temperature,
        seed=seed,
    )

    print_interleaving_comparison_table([separate, fused])

    return {
        "separate_calls": separate,
        "interleaved_fused": fused,
    }
