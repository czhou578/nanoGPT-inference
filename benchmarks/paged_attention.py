"""
Contiguous KV vs paged KV benchmark for nanogpt-paged-attention.py.

This file is standalone so importing it does not train nanogpt-paged-attention.py.
It assumes the model API used by that script:

    logits, loss, new_kvs = model(idx, targets=None, pos=None,
                                  past_kvs=None, attn_mask=None)

The benchmark compares:
- contiguous_kv: each request owns a normal contiguous per-layer KV cache
- paged_kv: each request owns a block table into a fixed-size physical KV pool

This educational implementation still gathers paged KV into contiguous tensors
before each model forward because the NanoGPT model does not implement a true
paged-attention kernel. The benchmark therefore measures the serving-engine
bookkeeping and memory behavior, not a production fused PagedAttention kernel.
"""

from dataclasses import dataclass, field
from statistics import mean
import time
import torch
import torch.nn.functional as F


@dataclass
class PagedAttentionRequestSpec:
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_step: int = 0
    group: str = "default"


@dataclass
class ContiguousRequestState:
    spec: PagedAttentionRequestSpec
    generated_tokens: list[int] = field(default_factory=list)
    past_kvs: object = None
    last_token: torch.Tensor | None = None
    arrived_at_s: float | None = None
    first_token_at_s: float | None = None
    completed_at_s: float | None = None

    @property
    def is_done(self) -> bool:
        return len(self.generated_tokens) >= self.spec.max_new_tokens

    @property
    def cache_len(self) -> int:
        if self.past_kvs is None:
            return 0
        return self.past_kvs[0][0][0].shape[1]


@dataclass
class PagedRequestState:
    spec: PagedAttentionRequestSpec
    generated_tokens: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    num_filled_slots: int = 0
    last_token: torch.Tensor | None = None
    arrived_at_s: float | None = None
    first_token_at_s: float | None = None
    completed_at_s: float | None = None

    @property
    def is_done(self) -> bool:
        return len(self.generated_tokens) >= self.spec.max_new_tokens

    @property
    def allocated_slots(self) -> int:
        return len(self.block_table)


class BlockAllocator:
    def __init__(self, num_blocks):
        self.free_blocks = list(range(num_blocks))
        self.num_blocks = num_blocks
        self.allocations = 0
        self.frees = 0
        self.peak_used = 0

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def allocate_one(self):
        if not self.free_blocks:
            raise MemoryError("No free KV blocks available")
        block = self.free_blocks.pop()
        self.allocations += 1
        self.peak_used = max(self.peak_used, self.used_blocks)
        return block

    def allocate_n(self, n):
        if len(self.free_blocks) < n:
            raise MemoryError("No free KV blocks available")
        return [self.allocate_one() for _ in range(n)]

    def free(self, blocks):
        self.free_blocks.extend(blocks)
        self.frees += len(blocks)


class KVBlockPool:
    def __init__(self, num_blocks, page_block_size, n_layer, n_head, head_size, device, dtype):
        self.num_blocks = num_blocks
        self.page_block_size = page_block_size
        self.k_pool = {}
        self.v_pool = {}

        for layer in range(n_layer):
            for head in range(n_head):
                self.k_pool[(layer, head)] = torch.zeros(
                    num_blocks,
                    page_block_size,
                    head_size,
                    device=device,
                    dtype=dtype,
                )
                self.v_pool[(layer, head)] = torch.zeros(
                    num_blocks,
                    page_block_size,
                    head_size,
                    device=device,
                    dtype=dtype,
                )


@dataclass
class PagedStepMetrics:
    step: int
    waiting: int
    active: int
    decoded_tokens: int
    prefilled_requests: int
    decode_batch_size: int
    used_blocks: int
    waste_slots: int
    forward_seconds: float
    step_seconds: float


@dataclass
class PagedAttentionRunMetrics:
    name: str
    total_requests: int
    completed_requests: int
    total_prompt_tokens: int
    total_generated_tokens: int
    total_seconds: float
    request_latencies_s: list[float]
    ttft_s: list[float]
    step_metrics: list[PagedStepMetrics]
    forward_seconds: float
    allocation_count: int = 0
    free_count: int = 0
    peak_blocks_used: int = 0
    total_physical_blocks: int = 0
    page_block_size: int = 0

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
    def avg_decode_batch_size(self) -> float:
        values = [s.decode_batch_size for s in self.step_metrics if s.decode_batch_size > 0]
        return mean(values) if values else 0.0

    @property
    def max_decode_batch_size(self) -> int:
        return max((s.decode_batch_size for s in self.step_metrics), default=0)

    @property
    def peak_waste_slots(self) -> int:
        return max((s.waste_slots for s in self.step_metrics), default=0)

    @property
    def avg_waste_slots(self) -> float:
        values = [s.waste_slots for s in self.step_metrics]
        return mean(values) if values else 0.0


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


def _stack_contiguous_kvs(requests, device):
    lengths = [req.cache_len for req in requests]
    max_len = max(lengths)
    pad_lengths = [max_len - length for length in lengths]

    attn_mask = torch.zeros((len(requests), 1, max_len), dtype=torch.bool, device=device)
    for i, pad in enumerate(pad_lengths):
        attn_mask[i, :, pad:] = True

    n_layer = len(requests[0].past_kvs)
    batched = []
    for layer_idx in range(n_layer):
        layer = []
        n_head = len(requests[0].past_kvs[layer_idx])
        for head_idx in range(n_head):
            keys = []
            values = []
            for req, pad in zip(requests, pad_lengths):
                k, v = req.past_kvs[layer_idx][head_idx]
                if pad > 0:
                    hs = k.shape[-1]
                    keys.append(torch.cat([torch.zeros(1, pad, hs, dtype=k.dtype, device=device), k], dim=1))
                    values.append(torch.cat([torch.zeros(1, pad, hs, dtype=v.dtype, device=device), v], dim=1))
                else:
                    keys.append(k)
                    values.append(v)
            layer.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        batched.append(layer)
    return batched, attn_mask, pad_lengths


def _unstack_contiguous_kvs(requests, batched_kvs, pad_lengths):
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


def _infer_cache_shape(model, vocab_size, device):
    probe = torch.zeros((1, 1), dtype=torch.long, device=device)
    pos = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        _, _, kvs = model(probe % vocab_size, pos=pos)
    n_layer = len(kvs)
    n_head = len(kvs[0])
    head_size = kvs[0][0][0].shape[-1]
    dtype = kvs[0][0][0].dtype
    return n_layer, n_head, head_size, dtype


@torch.no_grad()
def _prefill_contiguous(model, req, device, temperature, generator):
    prompt = torch.tensor([req.spec.prompt_tokens], dtype=torch.long, device=device)
    positions = torch.arange(len(req.spec.prompt_tokens), dtype=torch.long, device=device).unsqueeze(0)
    logits, _, kvs = model(prompt, pos=positions)
    req.past_kvs = kvs
    next_token = _sample_next_token(logits[:, -1, :], temperature=temperature, generator=generator)
    _record_generated_token(req, next_token)


@torch.no_grad()
def _decode_contiguous_batch(model, requests, device, temperature, generator):
    tokens = torch.cat([req.last_token for req in requests], dim=0)
    positions = torch.tensor(
        [[len(req.spec.prompt_tokens) + len(req.generated_tokens) - 1] for req in requests],
        dtype=torch.long,
        device=device,
    )
    past_kvs, attn_mask, pad_lengths = _stack_contiguous_kvs(requests, device)
    logits, _, new_kvs = model(tokens, pos=positions, past_kvs=past_kvs, attn_mask=attn_mask)
    next_tokens = _sample_next_token(logits[:, -1, :], temperature=temperature, generator=generator)
    _unstack_contiguous_kvs(requests, new_kvs, pad_lengths)
    for i, req in enumerate(requests):
        _record_generated_token(req, next_tokens[i:i + 1])


def _write_kvs_to_pool(pool, block_table, page_block_size, start_pos, kvs):
    for layer_idx, layer in enumerate(kvs):
        for head_idx, (k, v) in enumerate(layer):
            for t in range(k.shape[1]):
                logical_pos = start_pos + t
                block_idx = logical_pos // page_block_size
                slot_idx = logical_pos % page_block_size
                phys_block = block_table[block_idx]
                pool.k_pool[(layer_idx, head_idx)][phys_block, slot_idx, :] = k[0, t, :]
                pool.v_pool[(layer_idx, head_idx)][phys_block, slot_idx, :] = v[0, t, :]


def _gather_paged_kv(pool, req, page_block_size, layer_idx, head_idx):
    if req.num_filled_slots == 0:
        hs = pool.k_pool[(layer_idx, head_idx)].shape[-1]
        device = pool.k_pool[(layer_idx, head_idx)].device
        dtype = pool.k_pool[(layer_idx, head_idx)].dtype
        return (
            torch.empty(1, 0, hs, device=device, dtype=dtype),
            torch.empty(1, 0, hs, device=device, dtype=dtype),
        )

    full_blocks = req.num_filled_slots // page_block_size
    trailing = req.num_filled_slots % page_block_size
    k_parts = []
    v_parts = []

    for i in range(full_blocks):
        block = req.block_table[i]
        k_parts.append(pool.k_pool[(layer_idx, head_idx)][block, :, :])
        v_parts.append(pool.v_pool[(layer_idx, head_idx)][block, :, :])

    if trailing > 0:
        block = req.block_table[full_blocks]
        k_parts.append(pool.k_pool[(layer_idx, head_idx)][block, :trailing, :])
        v_parts.append(pool.v_pool[(layer_idx, head_idx)][block, :trailing, :])

    return torch.cat(k_parts, dim=0).unsqueeze(0), torch.cat(v_parts, dim=0).unsqueeze(0)


def _assemble_paged_kvs(requests, pool, page_block_size, device):
    lengths = [req.num_filled_slots for req in requests]
    max_len = max(lengths)
    pad_lengths = [max_len - length for length in lengths]
    attn_mask = torch.zeros((len(requests), 1, max_len), dtype=torch.bool, device=device)
    for i, pad in enumerate(pad_lengths):
        attn_mask[i, :, pad:] = True

    layer_head_keys = sorted(pool.k_pool.keys())
    n_layer = max(layer for layer, _ in layer_head_keys) + 1
    n_head = max(head for _, head in layer_head_keys) + 1

    batched = []
    for layer_idx in range(n_layer):
        layer = []
        for head_idx in range(n_head):
            keys = []
            values = []
            for req, pad in zip(requests, pad_lengths):
                k, v = _gather_paged_kv(pool, req, page_block_size, layer_idx, head_idx)
                if pad > 0:
                    hs = k.shape[-1]
                    k = torch.cat([torch.zeros(1, pad, hs, dtype=k.dtype, device=device), k], dim=1)
                    v = torch.cat([torch.zeros(1, pad, hs, dtype=v.dtype, device=device), v], dim=1)
                keys.append(k)
                values.append(v)
            layer.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        batched.append(layer)

    return batched, attn_mask, pad_lengths


@torch.no_grad()
def _prefill_paged(model, req, pool, allocator, page_block_size, device, temperature, generator):
    prompt_len = len(req.spec.prompt_tokens)
    blocks_needed = (prompt_len + page_block_size - 1) // page_block_size
    req.block_table = allocator.allocate_n(blocks_needed)

    prompt = torch.tensor([req.spec.prompt_tokens], dtype=torch.long, device=device)
    positions = torch.arange(prompt_len, dtype=torch.long, device=device).unsqueeze(0)
    logits, _, kvs = model(prompt, pos=positions)
    _write_kvs_to_pool(pool, req.block_table, page_block_size, 0, kvs)
    req.num_filled_slots = prompt_len
    next_token = _sample_next_token(logits[:, -1, :], temperature=temperature, generator=generator)
    _record_generated_token(req, next_token)


@torch.no_grad()
def _decode_paged_batch(model, requests, pool, allocator, page_block_size, device, temperature, generator):
    for req in requests:
        if req.num_filled_slots % page_block_size == 0:
            req.block_table.append(allocator.allocate_one())

    tokens = torch.cat([req.last_token for req in requests], dim=0)
    positions = torch.tensor(
        [[len(req.spec.prompt_tokens) + len(req.generated_tokens) - 1] for req in requests],
        dtype=torch.long,
        device=device,
    )
    past_kvs, attn_mask, _ = _assemble_paged_kvs(requests, pool, page_block_size, device)
    logits, _, new_kvs = model(tokens, pos=positions, past_kvs=past_kvs, attn_mask=attn_mask)
    next_tokens = _sample_next_token(logits[:, -1, :], temperature=temperature, generator=generator)

    for req_idx, req in enumerate(requests):
        for layer_idx, layer in enumerate(new_kvs):
            for head_idx, (k, v) in enumerate(layer):
                phys_block = req.block_table[req.num_filled_slots // page_block_size]
                slot = req.num_filled_slots % page_block_size
                pool.k_pool[(layer_idx, head_idx)][phys_block, slot, :] = k[req_idx, -1, :]
                pool.v_pool[(layer_idx, head_idx)][phys_block, slot, :] = v[req_idx, -1, :]

    for i, req in enumerate(requests):
        req.num_filled_slots += 1
        _record_generated_token(req, next_tokens[i:i + 1])


def _active_waste_slots(active, page_block_size):
    return sum(len(req.block_table) * page_block_size - req.num_filled_slots for req in active)


def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


@torch.no_grad()
def run_contiguous_kv_policy(
    model,
    workload,
    *,
    device,
    max_batch_size=4,
    temperature=1.0,
    seed=1337,
):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    pending = [
        ContiguousRequestState(spec=req)
        for req in sorted(workload, key=lambda r: (r.arrival_step, r.id))
    ]
    all_states = list(pending)
    active = []
    completed = []
    step_metrics = []
    forward_seconds = 0.0

    _sync_if_cuda(device)
    start = time.perf_counter()
    step = 0

    while pending or active:
        step_start = time.perf_counter()
        prefilled = 0

        while pending and pending[0].spec.arrival_step <= step and len(active) < max_batch_size:
            req = pending.pop(0)
            req.arrived_at_s = time.perf_counter()
            fwd_start = time.perf_counter()
            _prefill_contiguous(model, req, device, temperature, generator)
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start
            prefilled += 1
            active.append(req)

        decode_batch = active[:max_batch_size]
        if decode_batch:
            fwd_start = time.perf_counter()
            _decode_contiguous_batch(model, decode_batch, device, temperature, generator)
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start

            for req in list(active):
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    active.remove(req)
                    completed.append(req)

        step_metrics.append(PagedStepMetrics(
            step=step,
            waiting=len(pending),
            active=len(active),
            decoded_tokens=len(decode_batch),
            prefilled_requests=prefilled,
            decode_batch_size=len(decode_batch),
            used_blocks=0,
            waste_slots=0,
            forward_seconds=forward_seconds,
            step_seconds=time.perf_counter() - step_start,
        ))
        step += 1

    _sync_if_cuda(device)
    total_seconds = time.perf_counter() - start
    return PagedAttentionRunMetrics(
        name="contiguous_kv",
        total_requests=len(all_states),
        completed_requests=len(completed),
        total_prompt_tokens=sum(len(req.spec.prompt_tokens) for req in all_states),
        total_generated_tokens=sum(len(req.generated_tokens) for req in all_states),
        total_seconds=total_seconds,
        request_latencies_s=[req.completed_at_s - req.arrived_at_s for req in all_states],
        ttft_s=[req.first_token_at_s - req.arrived_at_s for req in all_states],
        step_metrics=step_metrics,
        forward_seconds=forward_seconds,
    )


@torch.no_grad()
def run_paged_kv_policy(
    model,
    workload,
    *,
    vocab_size,
    device,
    page_block_size=4,
    num_physical_blocks=64,
    max_batch_size=4,
    temperature=1.0,
    seed=1337,
):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    n_layer, n_head, head_size, dtype = _infer_cache_shape(model, vocab_size, device)
    pool = KVBlockPool(num_physical_blocks, page_block_size, n_layer, n_head, head_size, device, dtype)
    allocator = BlockAllocator(num_physical_blocks)

    pending = [
        PagedRequestState(spec=req)
        for req in sorted(workload, key=lambda r: (r.arrival_step, r.id))
    ]
    all_states = list(pending)
    active = []
    completed = []
    step_metrics = []
    forward_seconds = 0.0

    _sync_if_cuda(device)
    start = time.perf_counter()
    step = 0

    while pending or active:
        step_start = time.perf_counter()
        prefilled = 0

        admitted_any = True
        while admitted_any and pending and pending[0].spec.arrival_step <= step and len(active) < max_batch_size:
            admitted_any = False
            req = pending[0]
            blocks_needed = (len(req.spec.prompt_tokens) + page_block_size - 1) // page_block_size
            if allocator.num_free >= blocks_needed:
                pending.pop(0)
                req.arrived_at_s = time.perf_counter()
                fwd_start = time.perf_counter()
                _prefill_paged(model, req, pool, allocator, page_block_size, device, temperature, generator)
                _sync_if_cuda(device)
                forward_seconds += time.perf_counter() - fwd_start
                active.append(req)
                prefilled += 1
                admitted_any = True

        decode_batch = active[:max_batch_size]
        runnable = []
        for req in decode_batch:
            needs_new_block = req.num_filled_slots % page_block_size == 0
            if not needs_new_block or allocator.num_free > 0:
                runnable.append(req)

        if runnable:
            fwd_start = time.perf_counter()
            _decode_paged_batch(model, runnable, pool, allocator, page_block_size, device, temperature, generator)
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start

            for req in list(active):
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    active.remove(req)
                    allocator.free(req.block_table)
                    completed.append(req)

        if not runnable and active:
            raise RuntimeError(
                "Paged KV benchmark is blocked: active requests need new blocks "
                "but the physical block pool is exhausted."
            )

        step_metrics.append(PagedStepMetrics(
            step=step,
            waiting=len(pending),
            active=len(active),
            decoded_tokens=len(runnable),
            prefilled_requests=prefilled,
            decode_batch_size=len(runnable),
            used_blocks=allocator.used_blocks,
            waste_slots=_active_waste_slots(active, page_block_size),
            forward_seconds=forward_seconds,
            step_seconds=time.perf_counter() - step_start,
        ))
        step += 1

    _sync_if_cuda(device)
    total_seconds = time.perf_counter() - start
    return PagedAttentionRunMetrics(
        name="paged_kv",
        total_requests=len(all_states),
        completed_requests=len(completed),
        total_prompt_tokens=sum(len(req.spec.prompt_tokens) for req in all_states),
        total_generated_tokens=sum(len(req.generated_tokens) for req in all_states),
        total_seconds=total_seconds,
        request_latencies_s=[req.completed_at_s - req.arrived_at_s for req in all_states],
        ttft_s=[req.first_token_at_s - req.arrived_at_s for req in all_states],
        step_metrics=step_metrics,
        forward_seconds=forward_seconds,
        allocation_count=allocator.allocations,
        free_count=allocator.frees,
        peak_blocks_used=allocator.peak_used,
        total_physical_blocks=num_physical_blocks,
        page_block_size=page_block_size,
    )


def make_uniform_workload(
    *,
    vocab_size,
    num_requests=8,
    prompt_len=8,
    max_new_tokens=12,
    arrival_gap=0,
    seed=1337,
):
    rng = torch.Generator()
    rng.manual_seed(seed)
    return [
        PagedAttentionRequestSpec(
            id=i,
            prompt_tokens=torch.randint(0, vocab_size, (prompt_len,), generator=rng).tolist(),
            max_new_tokens=max_new_tokens,
            arrival_step=i * arrival_gap,
            group="uniform",
        )
        for i in range(num_requests)
    ]


def make_mixed_length_workload(
    *,
    vocab_size,
    num_requests=12,
    prompt_lens=(4, 7, 8, 11),
    output_lens=(4, 8, 12),
    arrival_gap=0,
    seed=1337,
):
    rng = torch.Generator()
    rng.manual_seed(seed)
    requests = []
    for i in range(num_requests):
        prompt_len = prompt_lens[i % len(prompt_lens)]
        output_len = output_lens[i % len(output_lens)]
        requests.append(PagedAttentionRequestSpec(
            id=i,
            prompt_tokens=torch.randint(0, vocab_size, (prompt_len,), generator=rng).tolist(),
            max_new_tokens=output_len,
            arrival_step=i * arrival_gap,
            group="mixed",
        ))
    return requests


def make_block_boundary_workload(
    *,
    vocab_size,
    num_requests=10,
    max_new_tokens=8,
    seed=1337,
):
    rng = torch.Generator()
    rng.manual_seed(seed)
    prompt_lens = [3, 4, 5, 7, 8, 9, 11, 12, 13, 15]
    return [
        PagedAttentionRequestSpec(
            id=i,
            prompt_tokens=torch.randint(0, vocab_size, (prompt_lens[i % len(prompt_lens)],), generator=rng).tolist(),
            max_new_tokens=max_new_tokens,
            arrival_step=0,
            group="block_boundary",
        )
        for i in range(num_requests)
    ]


def print_paged_attention_comparison_table(rows):
    headers = [
        "method",
        "reqs",
        "done",
        "prompt_tok",
        "gen_tok",
        "wall_s",
        "tok/s",
        "avg_ttft_ms",
        "p95_ttft_ms",
        "avg_lat_ms",
        "p95_lat_ms",
        "avg_batch",
        "max_batch",
        "peak_blocks",
        "allocs",
        "frees",
        "avg_waste",
        "peak_waste",
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
            f"{row.avg_ttft_s * 1000:.2f}",
            f"{_percentile(row.ttft_s, 0.95) * 1000:.2f}",
            f"{row.avg_latency_s * 1000:.2f}",
            f"{_percentile(row.request_latencies_s, 0.95) * 1000:.2f}",
            f"{row.avg_decode_batch_size:.2f}",
            str(row.max_decode_batch_size),
            str(row.peak_blocks_used),
            str(row.allocation_count),
            str(row.free_count),
            f"{row.avg_waste_slots:.2f}",
            str(row.peak_waste_slots),
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
        contiguous, paged = rows
        print()
        print(
            "Paged KV throughput ratio: "
            f"{paged.generated_tokens_per_second / contiguous.generated_tokens_per_second:.2f}x"
        )
        if contiguous.avg_latency_s > 0:
            print(
                "Paged KV average latency ratio: "
                f"{paged.avg_latency_s / contiguous.avg_latency_s:.2f}x"
            )
        print(
            "Paged KV peak block utilization: "
            f"{paged.peak_blocks_used}/{paged.total_physical_blocks}"
        )


def run_contiguous_vs_paged_attention_benchmark(
    model,
    *,
    vocab_size,
    workload_name="uniform",
    page_block_size=4,
    num_physical_blocks=64,
    max_batch_size=4,
    device=None,
    seed=1337,
    temperature=1.0,
    **workload_kwargs,
):
    if device is None:
        device = next(model.parameters()).device

    if workload_name == "uniform":
        workload = make_uniform_workload(vocab_size=vocab_size, seed=seed, **workload_kwargs)
    elif workload_name == "mixed_lengths":
        workload = make_mixed_length_workload(vocab_size=vocab_size, seed=seed, **workload_kwargs)
    elif workload_name == "block_boundary":
        workload = make_block_boundary_workload(vocab_size=vocab_size, seed=seed, **workload_kwargs)
    else:
        raise ValueError(f"Unknown paged-attention workload: {workload_name}")

    contiguous = run_contiguous_kv_policy(
        model,
        workload,
        device=device,
        max_batch_size=max_batch_size,
        temperature=temperature,
        seed=seed,
    )

    paged = run_paged_kv_policy(
        model,
        workload,
        vocab_size=vocab_size,
        device=device,
        page_block_size=page_block_size,
        num_physical_blocks=num_physical_blocks,
        max_batch_size=max_batch_size,
        temperature=temperature,
        seed=seed,
    )

    print_paged_attention_comparison_table([contiguous, paged])

    return {
        "contiguous_kv": contiguous,
        "paged_kv": paged,
    }
