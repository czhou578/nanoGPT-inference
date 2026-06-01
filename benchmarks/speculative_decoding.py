"""
KV-cache vs speculative decoding benchmark for nanogpt-spec-decode.py.

This file is standalone so importing it does not train nanogpt-spec-decode.py.
It assumes the model API used by that script:

    logits, loss, new_kvs = model(idx, targets=None, pos=None,
                                  past_kvs=None, attn_mask=None,
                                  input_mask=None)

The benchmark compares:
- kv_decode: normal autoregressive decoding with a target-model KV cache
- spec_decode: a cheap bigram draft model proposes multiple tokens, then the
  target model verifies those tokens in one forward pass

The draft model here is intentionally simple and deterministic to construct.
It is useful for measuring speculative-decoding mechanics: acceptance rate,
target forward-call reduction, target tokens evaluated, and end-to-end latency.
"""

from dataclasses import dataclass, field
from statistics import mean
import time
import torch
import torch.nn.functional as F


@dataclass
class SpecDecodeRequestSpec:
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int


@dataclass
class SpecDecodeRequestState:
    spec: SpecDecodeRequestSpec
    generated_tokens: list[int] = field(default_factory=list)
    past_kvs: object = None
    last_token: int | None = None
    arrived_at_s: float | None = None
    first_token_at_s: float | None = None
    completed_at_s: float | None = None
    verify_iterations: int = 0

    @property
    def is_done(self) -> bool:
        return len(self.generated_tokens) >= self.spec.max_new_tokens

    @property
    def cache_len(self) -> int:
        if self.past_kvs is None:
            return 0
        return self.past_kvs[0][0][0].shape[1]


class BigramDraftModel:
    """
    Cheap draft model for speculative decoding.

    It estimates P(next_token | current_token) from a token corpus. The optional
    draft_noise parameter blends the learned transition table with a uniform
    distribution, which makes it easy to create lower-quality draft runs.
    """

    def __init__(self, token_ids, vocab_size, device, draft_noise=0.0):
        counts = torch.ones(vocab_size, vocab_size, dtype=torch.float32)

        if token_ids is not None:
            ids = torch.as_tensor(token_ids, dtype=torch.long).flatten().cpu()
            if ids.numel() > 1:
                for prev_tok, next_tok in zip(ids[:-1].tolist(), ids[1:].tolist()):
                    if 0 <= prev_tok < vocab_size and 0 <= next_tok < vocab_size:
                        counts[prev_tok, next_tok] += 1.0

        probs = counts / counts.sum(dim=1, keepdim=True)

        if draft_noise > 0:
            noise = max(0.0, min(float(draft_noise), 1.0))
            uniform = torch.full_like(probs, 1.0 / vocab_size)
            probs = (1.0 - noise) * probs + noise * uniform

        self.probs = probs.to(device)

    def get_probs(self, token_id, temperature=1.0):
        probs = self.probs[int(token_id)]
        if temperature == 1.0:
            return probs

        scaled = probs.clamp_min(1e-12).pow(1.0 / temperature)
        return scaled / scaled.sum()

    def sample(self, token_id, *, temperature=1.0, generator=None):
        probs = self.get_probs(token_id, temperature=temperature)
        token = torch.multinomial(probs, num_samples=1, generator=generator).item()
        return token, probs


@dataclass
class SpecDecodeRunMetrics:
    name: str
    total_requests: int
    total_prompt_tokens: int
    total_generated_tokens: int
    total_seconds: float
    request_latencies_s: list[float]
    ttft_s: list[float]
    forward_seconds: float
    target_forward_calls: int
    target_tokens_evaluated: int
    verify_iterations: int = 0
    draft_tokens_proposed: int = 0
    accepted_draft_tokens: int = 0
    bonus_tokens: int = 0
    resampled_tokens: int = 0

    @property
    def generated_tokens_per_second(self):
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_generated_tokens / self.total_seconds

    @property
    def avg_latency_s(self):
        return mean(self.request_latencies_s) if self.request_latencies_s else 0.0

    @property
    def avg_ttft_s(self):
        return mean(self.ttft_s) if self.ttft_s else 0.0

    @property
    def acceptance_rate(self):
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.accepted_draft_tokens / self.draft_tokens_proposed

    @property
    def target_tokens_per_generated_token(self):
        if self.total_generated_tokens == 0:
            return 0.0
        return self.target_tokens_evaluated / self.total_generated_tokens

    @property
    def avg_tokens_per_verify(self):
        if self.verify_iterations == 0:
            return 0.0
        return (
            self.accepted_draft_tokens + self.bonus_tokens + self.resampled_tokens
        ) / self.verify_iterations


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


def _sample_next_token(logits, *, temperature=1.0, generator=None):
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _target_probs_from_logits(logits, *, temperature=1.0):
    return [
        F.softmax(logits[0, i, :] / temperature, dim=-1)
        for i in range(logits.shape[1])
    ]


def _record_token(req, token_id):
    now = time.perf_counter()
    req.generated_tokens.append(int(token_id))
    req.last_token = int(token_id)
    if req.first_token_at_s is None:
        req.first_token_at_s = now


def _trim_kv_cache(new_kvs, cache_len_before_verify, keep_new_tokens):
    keep = cache_len_before_verify + keep_new_tokens
    trimmed = []
    for layer_kv in new_kvs:
        trimmed_layer = []
        for k, v in layer_kv:
            trimmed_layer.append((k[:, :keep, :].contiguous(), v[:, :keep, :].contiguous()))
        trimmed.append(trimmed_layer)
    return trimmed


def _draft_tokens(draft_model, current_token, k, *, temperature, generator):
    candidates = []
    draft_probs = []
    token = current_token

    for _ in range(k):
        next_token, probs = draft_model.sample(
            token,
            temperature=temperature,
            generator=generator,
        )
        candidates.append(next_token)
        draft_probs.append(probs)
        token = next_token

    return candidates, draft_probs


def _accept_reject(
    candidates,
    draft_probs,
    target_probs,
    *,
    allow_bonus,
    generator,
):
    output_tokens = []
    accepted_draft_tokens = 0
    bonus_tokens = 0
    resampled_tokens = 0

    for i, token in enumerate(candidates):
        q = draft_probs[i][token].clamp_min(1e-12)
        p = target_probs[i][token]
        accept_prob = (p / q).clamp(max=1.0)

        draw = torch.rand((), device=p.device, generator=generator)
        if draw.item() < accept_prob.item():
            output_tokens.append(token)
            accepted_draft_tokens += 1
            continue

        adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
        adjusted_sum = adjusted.sum()
        if adjusted_sum > 0:
            adjusted = adjusted / adjusted_sum
        else:
            adjusted = target_probs[i]

        resampled = torch.multinomial(adjusted, num_samples=1, generator=generator).item()
        output_tokens.append(resampled)
        resampled_tokens = 1
        return output_tokens, accepted_draft_tokens, bonus_tokens, resampled_tokens

    if allow_bonus:
        bonus = torch.multinomial(
            target_probs[len(candidates)],
            num_samples=1,
            generator=generator,
        ).item()
        output_tokens.append(bonus)
        bonus_tokens = 1

    return output_tokens, accepted_draft_tokens, bonus_tokens, resampled_tokens


@torch.no_grad()
def run_kv_decode_policy(
    model,
    workload,
    *,
    name,
    device,
    temperature=1.0,
    seed=1337,
):
    model.eval()
    generator = _make_generator(device, seed)
    states = [SpecDecodeRequestState(spec=req) for req in workload]

    _sync_if_cuda(device)
    run_start = time.perf_counter()
    forward_seconds = 0.0
    target_forward_calls = 0
    target_tokens_evaluated = 0

    for req in states:
        req.arrived_at_s = time.perf_counter()

        prompt = torch.tensor([req.spec.prompt_tokens], dtype=torch.long, device=device)
        pos = torch.arange(len(req.spec.prompt_tokens), dtype=torch.long, device=device).unsqueeze(0)

        fwd_start = time.perf_counter()
        logits, _, req.past_kvs = model(prompt, pos=pos)
        _sync_if_cuda(device)
        forward_seconds += time.perf_counter() - fwd_start
        target_forward_calls += 1
        target_tokens_evaluated += len(req.spec.prompt_tokens)

        next_token = _sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            generator=generator,
        ).item()
        _record_token(req, next_token)

        while not req.is_done:
            input_id = torch.tensor([[req.last_token]], dtype=torch.long, device=device)
            pos_value = len(req.spec.prompt_tokens) + len(req.generated_tokens) - 1
            pos = torch.tensor([[pos_value]], dtype=torch.long, device=device)

            fwd_start = time.perf_counter()
            logits, _, req.past_kvs = model(input_id, pos=pos, past_kvs=req.past_kvs)
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start
            target_forward_calls += 1
            target_tokens_evaluated += 1

            next_token = _sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                generator=generator,
            ).item()
            _record_token(req, next_token)

        req.completed_at_s = time.perf_counter()

    _sync_if_cuda(device)
    run_end = time.perf_counter()

    return SpecDecodeRunMetrics(
        name=name,
        total_requests=len(states),
        total_prompt_tokens=sum(len(req.spec.prompt_tokens) for req in states),
        total_generated_tokens=sum(len(req.generated_tokens) for req in states),
        total_seconds=run_end - run_start,
        request_latencies_s=[req.completed_at_s - req.arrived_at_s for req in states],
        ttft_s=[req.first_token_at_s - req.arrived_at_s for req in states],
        forward_seconds=forward_seconds,
        target_forward_calls=target_forward_calls,
        target_tokens_evaluated=target_tokens_evaluated,
    )


@torch.no_grad()
def run_spec_decode_policy(
    model,
    workload,
    *,
    name,
    draft_model,
    speculation_len,
    device,
    temperature=1.0,
    seed=1337,
):
    if speculation_len < 1:
        raise ValueError("speculation_len must be at least 1")

    model.eval()
    generator = _make_generator(device, seed)
    states = [SpecDecodeRequestState(spec=req) for req in workload]

    _sync_if_cuda(device)
    run_start = time.perf_counter()
    forward_seconds = 0.0
    target_forward_calls = 0
    target_tokens_evaluated = 0
    verify_iterations = 0
    draft_tokens_proposed = 0
    accepted_draft_tokens = 0
    bonus_tokens = 0
    resampled_tokens = 0

    for req in states:
        req.arrived_at_s = time.perf_counter()

        prompt = torch.tensor([req.spec.prompt_tokens], dtype=torch.long, device=device)
        pos = torch.arange(len(req.spec.prompt_tokens), dtype=torch.long, device=device).unsqueeze(0)

        fwd_start = time.perf_counter()
        logits, _, req.past_kvs = model(prompt, pos=pos)
        _sync_if_cuda(device)
        forward_seconds += time.perf_counter() - fwd_start
        target_forward_calls += 1
        target_tokens_evaluated += len(req.spec.prompt_tokens)

        next_token = _sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            generator=generator,
        ).item()
        _record_token(req, next_token)

        while not req.is_done:
            remaining = req.spec.max_new_tokens - len(req.generated_tokens)
            k = min(speculation_len, remaining)

            candidates, draft_probs = _draft_tokens(
                draft_model,
                req.last_token,
                k,
                temperature=temperature,
                generator=generator,
            )
            draft_tokens_proposed += len(candidates)

            verify_tokens = [req.last_token] + candidates
            input_ids = torch.tensor([verify_tokens], dtype=torch.long, device=device)
            cache_len = req.cache_len
            positions = torch.arange(
                cache_len,
                cache_len + len(verify_tokens),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            fwd_start = time.perf_counter()
            logits, _, new_kvs = model(input_ids, pos=positions, past_kvs=req.past_kvs)
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start
            target_forward_calls += 1
            target_tokens_evaluated += len(verify_tokens)
            verify_iterations += 1
            req.verify_iterations += 1

            target_probs = _target_probs_from_logits(logits, temperature=temperature)
            allow_bonus = remaining > len(candidates)
            (
                accepted,
                accepted_count,
                bonus_count,
                resampled_count,
            ) = _accept_reject(
                candidates,
                draft_probs,
                target_probs,
                allow_bonus=allow_bonus,
                generator=generator,
            )

            req.past_kvs = _trim_kv_cache(
                new_kvs,
                cache_len_before_verify=cache_len,
                keep_new_tokens=1 + accepted_count,
            )

            for token in accepted[:remaining]:
                _record_token(req, token)

            accepted_draft_tokens += accepted_count
            bonus_tokens += bonus_count
            resampled_tokens += resampled_count

        req.completed_at_s = time.perf_counter()

    _sync_if_cuda(device)
    run_end = time.perf_counter()

    return SpecDecodeRunMetrics(
        name=name,
        total_requests=len(states),
        total_prompt_tokens=sum(len(req.spec.prompt_tokens) for req in states),
        total_generated_tokens=sum(len(req.generated_tokens) for req in states),
        total_seconds=run_end - run_start,
        request_latencies_s=[req.completed_at_s - req.arrived_at_s for req in states],
        ttft_s=[req.first_token_at_s - req.arrived_at_s for req in states],
        forward_seconds=forward_seconds,
        target_forward_calls=target_forward_calls,
        target_tokens_evaluated=target_tokens_evaluated,
        verify_iterations=verify_iterations,
        draft_tokens_proposed=draft_tokens_proposed,
        accepted_draft_tokens=accepted_draft_tokens,
        bonus_tokens=bonus_tokens,
        resampled_tokens=resampled_tokens,
    )


def make_spec_decode_workload(
    *,
    vocab_size,
    num_requests=8,
    prompt_len=16,
    max_new_tokens=12,
    prompt_source_tokens=None,
    seed=1337,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    source = None
    if prompt_source_tokens is not None:
        source = torch.as_tensor(prompt_source_tokens, dtype=torch.long).flatten().cpu()
        if source.numel() < prompt_len + 1:
            source = None

    workload = []
    for req_id in range(num_requests):
        if source is None:
            prompt = torch.randint(
                0,
                vocab_size,
                (prompt_len,),
                generator=generator,
            ).tolist()
        else:
            start = torch.randint(
                0,
                source.numel() - prompt_len,
                (1,),
                generator=generator,
            ).item()
            prompt = source[start:start + prompt_len].tolist()

        workload.append(SpecDecodeRequestSpec(
            id=req_id,
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
        ))

    return workload


def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def print_spec_decode_comparison_table(rows):
    headers = [
        "method",
        "reqs",
        "prompt_tok",
        "gen_tok",
        "wall_s",
        "gen_tok/s",
        "target_calls",
        "target_tok",
        "tgt_tok/gen",
        "avg_verify",
        "accept",
        "draft_tok",
        "bonus",
        "resample",
        "avg_ttft_ms",
        "p95_ttft_ms",
        "avg_lat_ms",
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
            str(row.target_forward_calls),
            str(row.target_tokens_evaluated),
            f"{row.target_tokens_per_generated_token:.2f}",
            f"{row.avg_tokens_per_verify:.2f}",
            f"{row.acceptance_rate * 100:.1f}%",
            str(row.draft_tokens_proposed),
            str(row.bonus_tokens),
            str(row.resampled_tokens),
            f"{row.avg_ttft_s * 1000:.2f}",
            f"{_percentile(row.ttft_s, 0.95) * 1000:.2f}",
            f"{row.avg_latency_s * 1000:.2f}",
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
        baseline, spec = rows
        print()
        print(
            "Speculative generated-token throughput ratio: "
            f"{spec.generated_tokens_per_second / baseline.generated_tokens_per_second:.2f}x"
        )
        print(
            "Target forward-call ratio: "
            f"{spec.target_forward_calls / baseline.target_forward_calls:.2f}x"
        )
        print(
            "Target tokens evaluated ratio: "
            f"{spec.target_tokens_evaluated / baseline.target_tokens_evaluated:.2f}x"
        )
        print(f"Draft acceptance rate: {spec.acceptance_rate * 100:.1f}%")


def run_kv_vs_speculative_decoding_benchmark(
    model,
    *,
    vocab_size,
    training_token_ids=None,
    prompt_source_tokens=None,
    num_requests=8,
    prompt_len=16,
    max_new_tokens=12,
    speculation_len=4,
    draft_noise=0.0,
    device=None,
    seed=1337,
    temperature=1.0,
):
    if device is None:
        device = next(model.parameters()).device

    workload = make_spec_decode_workload(
        vocab_size=vocab_size,
        num_requests=num_requests,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        prompt_source_tokens=prompt_source_tokens,
        seed=seed,
    )

    draft_model = BigramDraftModel(
        training_token_ids,
        vocab_size=vocab_size,
        device=device,
        draft_noise=draft_noise,
    )

    baseline = run_kv_decode_policy(
        model,
        workload,
        name="kv_decode",
        device=device,
        temperature=temperature,
        seed=seed,
    )

    speculative = run_spec_decode_policy(
        model,
        workload,
        name=f"spec_decode_k{speculation_len}",
        draft_model=draft_model,
        speculation_len=speculation_len,
        device=device,
        temperature=temperature,
        seed=seed,
    )

    print_spec_decode_comparison_table([baseline, speculative])

    return {
        "kv_decode": baseline,
        "spec_decode": speculative,
    }
