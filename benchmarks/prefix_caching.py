"""
Prefix-caching benchmark for nanogpt-prefix-caching.py.

This benchmark is standalone so importing it does not train the NanoGPT script.
It assumes the model API used by nanogpt-prefix-caching.py:

    logits, loss, new_kvs = model(idx, targets=None, pos=None,
                                  past_kvs=None, attn_mask=None)

The benchmark compares:
- no_prefix_cache: every request prefills its full prompt
- prefix_cache: previously completed prompt blocks can be reused by later
  requests that share the same block-aligned prefix
"""

from dataclasses import dataclass, field
import hashlib
from statistics import mean
import time
import torch
import torch.nn.functional as F


NONE_HASH = b"\x00" * 16


def hash_block_tokens(parent_hash, token_ids):
    data = (parent_hash, tuple(token_ids))
    return hashlib.md5(str(data).encode()).digest()


@dataclass
class PrefixRequestSpec:
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_step: int = 0
    group: str = "default"


@dataclass
class PrefixRequestState:
    spec: PrefixRequestSpec
    generated_tokens: list[int] = field(default_factory=list)
    past_kvs: object = None
    last_token: torch.Tensor | None = None
    prefill_cursor: int = 0
    arrived_at_s: float | None = None
    first_token_at_s: float | None = None
    completed_at_s: float | None = None
    cached_prefix_tokens: int = 0
    actual_prefill_tokens: int = 0

    @property
    def is_done(self) -> bool:
        return len(self.generated_tokens) >= self.spec.max_new_tokens


@dataclass
class CachedPrefixBlock:
    block_hash: bytes
    token_ids: tuple[int, ...]
    kv_data: object
    last_access_step: int = 0


class PrefixBlockCache:
    def __init__(self, max_blocks=64):
        self.max_blocks = max_blocks
        self.cache = {}
        self.current_step = 0
        self.lookups = 0
        self.hits = 0
        self.inserts = 0
        self.evictions = 0

    def lookup(self, block_hash):
        self.lookups += 1
        block = self.cache.get(block_hash)
        if block is not None:
            self.hits += 1
            block.last_access_step = self.current_step
        return block

    def insert(self, block_hash, token_ids, kv_data):
        if block_hash in self.cache:
            self.cache[block_hash].last_access_step = self.current_step
            return

        if len(self.cache) >= self.max_blocks:
            oldest = min(self.cache.values(), key=lambda b: b.last_access_step)
            del self.cache[oldest.block_hash]
            self.evictions += 1

        self.cache[block_hash] = CachedPrefixBlock(
            block_hash=block_hash,
            token_ids=tuple(token_ids),
            kv_data=_clone_kvs(kv_data),
            last_access_step=self.current_step,
        )
        self.inserts += 1


@dataclass
class PrefixCacheRunMetrics:
    name: str
    total_requests: int
    total_prompt_tokens: int
    actual_prefill_tokens: int
    cached_prefix_tokens: int
    total_generated_tokens: int
    total_seconds: float
    request_latencies_s: list[float]
    ttft_s: list[float]
    cache_lookups: int
    cache_hits: int
    cache_inserts: int
    cache_evictions: int
    final_cache_blocks: int
    forward_seconds: float

    @property
    def generated_tokens_per_second(self):
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_generated_tokens / self.total_seconds

    @property
    def prompt_tokens_per_second(self):
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_prompt_tokens / self.total_seconds

    @property
    def actual_prefill_tokens_per_second(self):
        if self.total_seconds <= 0:
            return float("inf")
        return self.actual_prefill_tokens / self.total_seconds

    @property
    def avg_latency_s(self):
        return mean(self.request_latencies_s) if self.request_latencies_s else 0.0

    @property
    def avg_ttft_s(self):
        return mean(self.ttft_s) if self.ttft_s else 0.0

    @property
    def cache_hit_rate(self):
        if self.cache_lookups == 0:
            return 0.0
        return self.cache_hits / self.cache_lookups

    @property
    def prefill_token_reduction(self):
        if self.total_prompt_tokens == 0:
            return 0.0
        return 1.0 - (self.actual_prefill_tokens / self.total_prompt_tokens)


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
    if req.first_token_at_s is None:
        req.first_token_at_s = now


def _clone_kvs(kvs):
    return [
        [(k.clone(), v.clone()) for k, v in layer]
        for layer in kvs
    ]


def _slice_kvs(kvs, start, end):
    return [
        [(k[:, start:end, :].clone(), v[:, start:end, :].clone()) for k, v in layer]
        for layer in kvs
    ]


def _concat_kvs(left, right):
    if left is None:
        return _clone_kvs(right)

    out = []
    for left_layer, right_layer in zip(left, right):
        layer = []
        for (lk, lv), (rk, rv) in zip(left_layer, right_layer):
            layer.append((torch.cat([lk, rk], dim=1), torch.cat([lv, rv], dim=1)))
        out.append(layer)
    return out


def _load_cached_prefix(req, block_cache, prefix_block_size):
    parent_hash = NONE_HASH
    loaded_kvs = None
    cached_tokens = 0

    for block_idx in range(len(req.spec.prompt_tokens) // prefix_block_size):
        start = block_idx * prefix_block_size
        end = start + prefix_block_size
        chunk = req.spec.prompt_tokens[start:end]
        parent_hash = hash_block_tokens(parent_hash, chunk)

        cached = block_cache.lookup(parent_hash)
        if cached is None:
            break

        loaded_kvs = _concat_kvs(loaded_kvs, cached.kv_data)
        cached_tokens += prefix_block_size

    req.past_kvs = loaded_kvs
    req.prefill_cursor = cached_tokens
    req.cached_prefix_tokens = cached_tokens
    return cached_tokens


def _commit_completed_blocks(req, block_cache, prefix_block_size):
    if req.past_kvs is None:
        return

    parent_hash = NONE_HASH
    num_full_blocks = len(req.spec.prompt_tokens) // prefix_block_size

    for block_idx in range(num_full_blocks):
        start = block_idx * prefix_block_size
        end = start + prefix_block_size
        chunk = req.spec.prompt_tokens[start:end]
        parent_hash = hash_block_tokens(parent_hash, chunk)

        block_kvs = _slice_kvs(req.past_kvs, start, end)
        block_cache.insert(parent_hash, tuple(chunk), block_kvs)


@torch.no_grad()
def _prefill_remaining(model, req, device, temperature, generator):
    start = req.prefill_cursor
    end = len(req.spec.prompt_tokens)

    if start >= end:
        raise ValueError(
            "This benchmark expects every prompt to have an uncached suffix. "
            "Use unique_suffix_len > 0 so first-token logits come from a real forward."
        )

    idx = torch.tensor([req.spec.prompt_tokens[start:end]], dtype=torch.long, device=device)
    pos = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0)

    logits, _, new_kvs = model(idx, pos=pos, past_kvs=req.past_kvs)
    req.past_kvs = new_kvs
    req.prefill_cursor = end
    req.actual_prefill_tokens += end - start

    next_token = _sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        generator=generator,
    )
    _record_generated_token(req, next_token)


@torch.no_grad()
def _decode_one(model, req, device, temperature, generator):
    pos_value = len(req.spec.prompt_tokens) + len(req.generated_tokens) - 1
    pos = torch.tensor([[pos_value]], dtype=torch.long, device=device)

    logits, _, new_kvs = model(
        req.last_token,
        pos=pos,
        past_kvs=req.past_kvs,
    )
    req.past_kvs = new_kvs
    next_token = _sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        generator=generator,
    )
    _record_generated_token(req, next_token)


@torch.no_grad()
def run_prefix_cache_policy(
    model,
    workload,
    *,
    name,
    use_prefix_cache,
    device,
    prefix_block_size=4,
    max_cache_blocks=64,
    temperature=1.0,
    seed=1337,
):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    block_cache = PrefixBlockCache(max_blocks=max_cache_blocks)
    states = [
        PrefixRequestState(spec=req)
        for req in sorted(workload, key=lambda r: (r.arrival_step, r.id))
    ]

    _sync_if_cuda(device)
    run_start = time.perf_counter()
    forward_seconds = 0.0

    for step, req in enumerate(states):
        block_cache.current_step = step
        req.arrived_at_s = time.perf_counter()

        if use_prefix_cache:
            _load_cached_prefix(req, block_cache, prefix_block_size)

        fwd_start = time.perf_counter()
        _prefill_remaining(model, req, device, temperature, generator)
        _sync_if_cuda(device)
        forward_seconds += time.perf_counter() - fwd_start

        if use_prefix_cache:
            _commit_completed_blocks(req, block_cache, prefix_block_size)

        while not req.is_done:
            fwd_start = time.perf_counter()
            _decode_one(model, req, device, temperature, generator)
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start

        req.completed_at_s = time.perf_counter()

    _sync_if_cuda(device)
    run_end = time.perf_counter()

    latencies = [req.completed_at_s - req.arrived_at_s for req in states]
    ttfts = [req.first_token_at_s - req.arrived_at_s for req in states]

    return PrefixCacheRunMetrics(
        name=name,
        total_requests=len(states),
        total_prompt_tokens=sum(len(req.spec.prompt_tokens) for req in states),
        actual_prefill_tokens=sum(req.actual_prefill_tokens for req in states),
        cached_prefix_tokens=sum(req.cached_prefix_tokens for req in states),
        total_generated_tokens=sum(len(req.generated_tokens) for req in states),
        total_seconds=run_end - run_start,
        request_latencies_s=latencies,
        ttft_s=ttfts,
        cache_lookups=block_cache.lookups,
        cache_hits=block_cache.hits,
        cache_inserts=block_cache.inserts,
        cache_evictions=block_cache.evictions,
        final_cache_blocks=len(block_cache.cache),
        forward_seconds=forward_seconds,
    )


def make_shared_prefix_workload(
    *,
    vocab_size,
    num_requests=8,
    shared_prefix_len=16,
    unique_suffix_len=4,
    max_new_tokens=4,
    num_groups=1,
    seed=1337,
):
    rng = torch.Generator()
    rng.manual_seed(seed)

    prefixes = [
        torch.randint(0, vocab_size, (shared_prefix_len,), generator=rng).tolist()
        for _ in range(num_groups)
    ]

    workload = []
    for i in range(num_requests):
        group_id = i % num_groups
        suffix = torch.randint(0, vocab_size, (unique_suffix_len,), generator=rng).tolist()
        workload.append(PrefixRequestSpec(
            id=i,
            prompt_tokens=prefixes[group_id] + suffix,
            max_new_tokens=max_new_tokens,
            arrival_step=i,
            group=f"shared_prefix_{group_id}",
        ))

    return workload


def make_low_reuse_workload(
    *,
    vocab_size,
    num_requests=8,
    prompt_len=20,
    max_new_tokens=4,
    seed=1337,
):
    rng = torch.Generator()
    rng.manual_seed(seed)
    return [
        PrefixRequestSpec(
            id=i,
            prompt_tokens=torch.randint(0, vocab_size, (prompt_len,), generator=rng).tolist(),
            max_new_tokens=max_new_tokens,
            arrival_step=i,
            group="unique_prefix",
        )
        for i in range(num_requests)
    ]


def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def print_prefix_cache_comparison_table(rows):
    headers = [
        "method",
        "reqs",
        "prompt_tok",
        "actual_prefill",
        "cached_tok",
        "gen_tok",
        "wall_s",
        "gen_tok/s",
        "prefill_tok/s",
        "avg_ttft_ms",
        "p95_ttft_ms",
        "avg_lat_ms",
        "hit_rate",
        "blocks",
        "evict",
        "forward_s",
    ]

    rendered = []
    for row in rows:
        rendered.append([
            row.name,
            str(row.total_requests),
            str(row.total_prompt_tokens),
            str(row.actual_prefill_tokens),
            str(row.cached_prefix_tokens),
            str(row.total_generated_tokens),
            f"{row.total_seconds:.4f}",
            f"{row.generated_tokens_per_second:.2f}",
            f"{row.actual_prefill_tokens_per_second:.2f}",
            f"{row.avg_ttft_s * 1000:.2f}",
            f"{_percentile(row.ttft_s, 0.95) * 1000:.2f}",
            f"{row.avg_latency_s * 1000:.2f}",
            f"{row.cache_hit_rate * 100:.1f}%",
            str(row.final_cache_blocks),
            str(row.cache_evictions),
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
        baseline, cached = rows
        print()
        print(
            "Prefix-cache generated-token throughput ratio: "
            f"{cached.generated_tokens_per_second / baseline.generated_tokens_per_second:.2f}x"
        )
        print(
            "Actual prefill-token reduction: "
            f"{cached.prefill_token_reduction * 100:.1f}%"
        )
        if baseline.avg_ttft_s > 0:
            print(
                "Average TTFT ratio: "
                f"{cached.avg_ttft_s / baseline.avg_ttft_s:.2f}x"
            )


def run_no_prefix_vs_prefix_cache_benchmark(
    model,
    *,
    vocab_size,
    workload_name="shared_prefix",
    prefix_block_size=4,
    max_cache_blocks=64,
    device=None,
    seed=1337,
    temperature=1.0,
    **workload_kwargs,
):
    if device is None:
        device = next(model.parameters()).device

    if workload_name == "shared_prefix":
        workload = make_shared_prefix_workload(
            vocab_size=vocab_size,
            seed=seed,
            **workload_kwargs,
        )
    elif workload_name == "low_reuse":
        workload = make_low_reuse_workload(
            vocab_size=vocab_size,
            seed=seed,
            **workload_kwargs,
        )
    else:
        raise ValueError(f"Unknown prefix caching workload: {workload_name}")

    no_cache = run_prefix_cache_policy(
        model,
        workload,
        name="no_prefix_cache",
        use_prefix_cache=False,
        device=device,
        prefix_block_size=prefix_block_size,
        max_cache_blocks=max_cache_blocks,
        temperature=temperature,
        seed=seed,
    )

    prefix_cache = run_prefix_cache_policy(
        model,
        workload,
        name="prefix_cache",
        use_prefix_cache=True,
        device=device,
        prefix_block_size=prefix_block_size,
        max_cache_blocks=max_cache_blocks,
        temperature=temperature,
        seed=seed,
    )

    print_prefix_cache_comparison_table([no_cache, prefix_cache])

    return {
        "no_prefix_cache": no_cache,
        "prefix_cache": prefix_cache,
    }
