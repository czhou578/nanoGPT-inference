"""
Radix-tree prefix-caching benchmark for nanogpt-radix-tree-.py.

Standalone benchmark — importing it does NOT train the model.
It assumes the model API from the nanogpt files:

    logits, loss, new_kvs = model(idx, targets=None, pos=None,
                                  past_kvs=None, attn_mask=None)

Compares three strategies:
- no_cache:      every request prefills its full prompt from scratch
- flat_cache:    block-aligned hash-chained prefix caching (BlockCache)
- radix_cache:   radix-tree prefix caching (RadixTree)
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import hashlib
from statistics import mean
import time
import torch
import torch.nn.functional as F


# ── Workload specs ────────────────────────────────────────────────────────────

@dataclass
class BenchRequestSpec:
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_step: int = 0
    group: str = "default"


@dataclass
class BenchRequestState:
    spec: BenchRequestSpec
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


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
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
    final_cache_entries: int
    forward_seconds: float
    tree_node_count: int = 0

    @property
    def generated_tokens_per_second(self):
        return self.total_generated_tokens / self.total_seconds if self.total_seconds > 0 else float("inf")

    @property
    def avg_latency_s(self):
        return mean(self.request_latencies_s) if self.request_latencies_s else 0.0

    @property
    def avg_ttft_s(self):
        return mean(self.ttft_s) if self.ttft_s else 0.0

    @property
    def cache_hit_rate(self):
        return self.cache_hits / self.cache_lookups if self.cache_lookups > 0 else 0.0

    @property
    def prefill_token_reduction(self):
        return 1.0 - (self.actual_prefill_tokens / self.total_prompt_tokens) if self.total_prompt_tokens > 0 else 0.0


# ── KV helpers ────────────────────────────────────────────────────────────────

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


# ── Flat hash-based prefix cache ─────────────────────────────────────────────

NONE_HASH = b"\x00" * 16


def _hash_block(parent_hash, token_ids):
    data = (parent_hash, tuple(token_ids))
    return hashlib.md5(str(data).encode()).digest()


@dataclass
class _FlatCachedBlock:
    block_hash: bytes
    token_ids: tuple
    kv_data: object
    last_access_step: int = 0


class _FlatBlockCache:
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
        self.cache[block_hash] = _FlatCachedBlock(
            block_hash=block_hash,
            token_ids=tuple(token_ids),
            kv_data=_clone_kvs(kv_data),
            last_access_step=self.current_step,
        )
        self.inserts += 1


def _flat_load_prefix(req, cache, block_size):
    parent_hash = NONE_HASH
    loaded_kvs = None
    cached_tokens = 0
    for block_idx in range(len(req.spec.prompt_tokens) // block_size):
        start = block_idx * block_size
        end = start + block_size
        chunk = req.spec.prompt_tokens[start:end]
        parent_hash = _hash_block(parent_hash, chunk)
        cached = cache.lookup(parent_hash)
        if cached is None:
            break
        loaded_kvs = _concat_kvs(loaded_kvs, cached.kv_data)
        cached_tokens += block_size
    req.past_kvs = loaded_kvs
    req.prefill_cursor = cached_tokens
    req.cached_prefix_tokens = cached_tokens
    return cached_tokens


def _flat_commit_blocks(req, cache, block_size):
    if req.past_kvs is None:
        return
    parent_hash = NONE_HASH
    for block_idx in range(len(req.spec.prompt_tokens) // block_size):
        start = block_idx * block_size
        end = start + block_size
        chunk = req.spec.prompt_tokens[start:end]
        parent_hash = _hash_block(parent_hash, chunk)
        block_kvs = _slice_kvs(req.past_kvs, start, end)
        cache.insert(parent_hash, tuple(chunk), block_kvs)


# ── Radix tree ────────────────────────────────────────────────────────────────

class _RadixNode:
    def __init__(self):
        self.children: Dict[int, '_RadixNode'] = {}
        self.parent: Optional['_RadixNode'] = None
        self.token_ids: Tuple[int, ...] = ()
        self.kv_data: Optional[object] = None  # list[list[(k,v)]]
        self.lock_ref: int = 0
        self.last_access_time: int = 0


class _RadixTree:
    def __init__(self):
        self.root = _RadixNode()
        self.step = 0
        self.lookups = 0
        self.hits = 0
        self.inserts = 0
        self.evictions = 0

    def match_prefix(self, token_ids):
        self.lookups += 1
        node = self.root
        matched = 0
        while matched < len(token_ids):
            next_token = token_ids[matched]
            child = node.children.get(next_token)
            if child is None:
                break
            edge_tokens = child.token_ids
            edge_match_len = 0
            while (edge_match_len < len(edge_tokens) and
                   matched + edge_match_len < len(token_ids) and
                   edge_tokens[edge_match_len] == token_ids[matched + edge_match_len]):
                edge_match_len += 1
            if edge_match_len < len(edge_tokens):
                child = self._split_node(child, edge_match_len)
                matched += edge_match_len
                node = child
                break
            matched += len(edge_tokens)
            node = child
        if matched > 0:
            self.hits += 1
        return node, matched

    def _split_node(self, child, split_len):
        mid = _RadixNode()
        mid.token_ids = child.token_ids[:split_len]
        mid.parent = child.parent
        mid.last_access_time = child.last_access_time
        mid.lock_ref = child.lock_ref

        if child.kv_data is not None:
            mid.kv_data = _slice_kvs(child.kv_data, 0, split_len)
            child.kv_data = _slice_kvs(child.kv_data, split_len, split_len + len(child.token_ids) - split_len)

        suffix = child.token_ids[split_len:]
        child.token_ids = suffix
        child.parent = mid
        mid.children[suffix[0]] = child
        mid.parent.children[mid.token_ids[0]] = mid
        return mid

    def insert(self, token_ids, kv_data_full, block_size):
        node, matched = self.match_prefix(token_ids)
        if matched == len(token_ids):
            node.last_access_time = self.step
            return
        remaining = token_ids[matched:]
        new_node = _RadixNode()
        new_node.token_ids = tuple(remaining)
        new_node.parent = node
        new_node.last_access_time = self.step
        new_node.kv_data = _slice_kvs(kv_data_full, matched, matched + len(remaining))
        node.children[remaining[0]] = new_node
        self.inserts += 1

    def evict_lru(self):
        leaves = []
        self._find_leaves(self.root, leaves)
        candidates = [n for n in leaves if n.lock_ref == 0 and n != self.root]
        if not candidates:
            return False
        victim = min(candidates, key=lambda n: n.last_access_time)
        parent = victim.parent
        del parent.children[victim.token_ids[0]]
        victim.kv_data = None
        victim.parent = None
        self.evictions += 1
        return True

    def _find_leaves(self, node, result):
        if not node.children:
            result.append(node)
        for child in node.children.values():
            self._find_leaves(child, result)

    def count_nodes(self):
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children.values())
        return count


def _radix_load_prefix(req, tree, block_size):
    node, matched = tree.match_prefix(req.spec.prompt_tokens)
    if matched == 0:
        return 0

    path = []
    curr = node
    while curr != tree.root:
        path.append(curr)
        curr = curr.parent
    path.reverse()

    # Collect KV along path
    loaded_kvs = None
    for pnode in path:
        if pnode.kv_data is not None:
            loaded_kvs = _concat_kvs(loaded_kvs, pnode.kv_data)

    # Pin nodes
    for pnode in path:
        pnode.lock_ref += 1

    num_cached = (matched // block_size) * block_size
    req.past_kvs = loaded_kvs
    req.prefill_cursor = num_cached
    req.cached_prefix_tokens = num_cached
    req._radix_path = path
    return num_cached


def _radix_commit(req, tree, block_size):
    if req.past_kvs is None:
        return
    tree.insert(req.spec.prompt_tokens, req.past_kvs, block_size)


def _radix_unlock(req):
    path = getattr(req, '_radix_path', None)
    if path is None:
        return
    for node in path:
        node.lock_ref -= 1
    req._radix_path = None


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def _prefill_remaining(model, req, device, temperature, generator):
    start = req.prefill_cursor
    end = len(req.spec.prompt_tokens)
    if start >= end:
        raise ValueError("Every prompt must have an uncached suffix (use unique_suffix_len > 0).")

    idx = torch.tensor([req.spec.prompt_tokens[start:end]], dtype=torch.long, device=device)
    pos = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0)

    logits, _, new_kvs = model(idx, pos=pos, past_kvs=req.past_kvs)
    req.past_kvs = new_kvs
    req.prefill_cursor = end
    req.actual_prefill_tokens += end - start

    next_token = _sample_next_token(logits[:, -1, :], temperature=temperature, generator=generator)
    _record_generated_token(req, next_token)


@torch.no_grad()
def _decode_one(model, req, device, temperature, generator):
    pos_value = len(req.spec.prompt_tokens) + len(req.generated_tokens) - 1
    pos = torch.tensor([[pos_value]], dtype=torch.long, device=device)

    logits, _, new_kvs = model(req.last_token, pos=pos, past_kvs=req.past_kvs)
    req.past_kvs = new_kvs
    next_token = _sample_next_token(logits[:, -1, :], temperature=temperature, generator=generator)
    _record_generated_token(req, next_token)


# ── Run a single policy ──────────────────────────────────────────────────────

@torch.no_grad()
def _run_policy(
    model,
    workload,
    *,
    name,
    policy,  # "none", "flat", "radix"
    device,
    block_size=4,
    max_cache_blocks=64,
    temperature=1.0,
    seed=1337,
):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    flat_cache = _FlatBlockCache(max_blocks=max_cache_blocks) if policy == "flat" else None
    radix_tree = _RadixTree() if policy == "radix" else None

    states = [
        BenchRequestState(spec=req)
        for req in sorted(workload, key=lambda r: (r.arrival_step, r.id))
    ]

    _sync_if_cuda(device)
    run_start = time.perf_counter()
    forward_seconds = 0.0

    for step, req in enumerate(states):
        req.arrived_at_s = time.perf_counter()

        # Load cached prefix
        if policy == "flat":
            flat_cache.current_step = step
            _flat_load_prefix(req, flat_cache, block_size)
        elif policy == "radix":
            radix_tree.step = step
            _radix_load_prefix(req, radix_tree, block_size)

        # Prefill
        fwd_start = time.perf_counter()
        _prefill_remaining(model, req, device, temperature, generator)
        _sync_if_cuda(device)
        forward_seconds += time.perf_counter() - fwd_start

        # Commit blocks
        if policy == "flat":
            _flat_commit_blocks(req, flat_cache, block_size)
        elif policy == "radix":
            _radix_commit(req, radix_tree, block_size)
            _radix_unlock(req)

        # Decode
        while not req.is_done:
            fwd_start = time.perf_counter()
            _decode_one(model, req, device, temperature, generator)
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start

        req.completed_at_s = time.perf_counter()

        if policy == "radix":
            _radix_unlock(req)  # unlock again on completion (safe if already unlocked)

    _sync_if_cuda(device)
    run_end = time.perf_counter()

    latencies = [req.completed_at_s - req.arrived_at_s for req in states]
    ttfts = [req.first_token_at_s - req.arrived_at_s for req in states]

    # Gather cache stats
    if policy == "flat":
        lookups, hits, inserts, evictions = flat_cache.lookups, flat_cache.hits, flat_cache.inserts, flat_cache.evictions
        final_entries = len(flat_cache.cache)
        tree_nodes = 0
    elif policy == "radix":
        lookups, hits, inserts, evictions = radix_tree.lookups, radix_tree.hits, radix_tree.inserts, radix_tree.evictions
        final_entries = radix_tree.count_nodes() - 1  # exclude root
        tree_nodes = radix_tree.count_nodes()
    else:
        lookups = hits = inserts = evictions = final_entries = tree_nodes = 0

    return RunMetrics(
        name=name,
        total_requests=len(states),
        total_prompt_tokens=sum(len(req.spec.prompt_tokens) for req in states),
        actual_prefill_tokens=sum(req.actual_prefill_tokens for req in states),
        cached_prefix_tokens=sum(req.cached_prefix_tokens for req in states),
        total_generated_tokens=sum(len(req.generated_tokens) for req in states),
        total_seconds=run_end - run_start,
        request_latencies_s=latencies,
        ttft_s=ttfts,
        cache_lookups=lookups,
        cache_hits=hits,
        cache_inserts=inserts,
        cache_evictions=evictions,
        final_cache_entries=final_entries,
        forward_seconds=forward_seconds,
        tree_node_count=tree_nodes,
    )


# ── Workload generators ──────────────────────────────────────────────────────

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
        workload.append(BenchRequestSpec(
            id=i,
            prompt_tokens=prefixes[group_id] + suffix,
            max_new_tokens=max_new_tokens,
            arrival_step=i,
            group=f"group_{group_id}",
        ))
    return workload


def make_branching_workload(
    *,
    vocab_size,
    trunk_len=16,
    branch_suffix_len=4,
    num_branches=4,
    max_new_tokens=4,
    seed=1337,
):
    """Multi-turn chat simulation: shared trunk + diverging branches."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    trunk = torch.randint(0, vocab_size, (trunk_len,), generator=rng).tolist()
    workload = []
    for i in range(num_branches):
        suffix = torch.randint(0, vocab_size, (branch_suffix_len,), generator=rng).tolist()
        workload.append(BenchRequestSpec(
            id=i,
            prompt_tokens=trunk + suffix,
            max_new_tokens=max_new_tokens,
            arrival_step=i,
            group=f"branch_{i}",
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
        BenchRequestSpec(
            id=i,
            prompt_tokens=torch.randint(0, vocab_size, (prompt_len,), generator=rng).tolist(),
            max_new_tokens=max_new_tokens,
            arrival_step=i,
        )
        for i in range(num_requests)
    ]


# ── Comparison runner ─────────────────────────────────────────────────────────

def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def print_comparison_table(rows):
    headers = [
        "method", "reqs", "prompt_tok", "prefill_tok", "cached_tok",
        "gen_tok", "wall_s", "gen_tok/s", "avg_ttft_ms", "p95_ttft_ms",
        "avg_lat_ms", "hit_rate", "entries", "evict", "tree_nodes",
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
            f"{row.avg_ttft_s * 1000:.2f}",
            f"{_percentile(row.ttft_s, 0.95) * 1000:.2f}",
            f"{row.avg_latency_s * 1000:.2f}",
            f"{row.cache_hit_rate * 100:.1f}%",
            str(row.final_cache_entries),
            str(row.cache_evictions),
            str(row.tree_node_count),
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

    # Summary comparisons
    baseline = rows[0]
    for row in rows[1:]:
        if baseline.generated_tokens_per_second > 0:
            ratio = row.generated_tokens_per_second / baseline.generated_tokens_per_second
            print(f"\n{row.name} throughput vs {baseline.name}: {ratio:.2f}x")
        print(f"{row.name} prefill token reduction: {row.prefill_token_reduction * 100:.1f}%")


def run_radix_vs_flat_benchmark(
    model,
    *,
    vocab_size,
    workload_name="shared_prefix",
    block_size=4,
    max_cache_blocks=64,
    device=None,
    seed=1337,
    temperature=1.0,
    **workload_kwargs,
):
    if device is None:
        device = next(model.parameters()).device

    if workload_name == "shared_prefix":
        workload = make_shared_prefix_workload(vocab_size=vocab_size, seed=seed, **workload_kwargs)
    elif workload_name == "branching":
        workload = make_branching_workload(vocab_size=vocab_size, seed=seed, **workload_kwargs)
    elif workload_name == "low_reuse":
        workload = make_low_reuse_workload(vocab_size=vocab_size, seed=seed, **workload_kwargs)
    else:
        raise ValueError(f"Unknown workload: {workload_name}")

    no_cache = _run_policy(
        model, workload, name="no_cache", policy="none",
        device=device, block_size=block_size, max_cache_blocks=max_cache_blocks,
        temperature=temperature, seed=seed,
    )

    flat = _run_policy(
        model, workload, name="flat_cache", policy="flat",
        device=device, block_size=block_size, max_cache_blocks=max_cache_blocks,
        temperature=temperature, seed=seed,
    )

    radix = _run_policy(
        model, workload, name="radix_tree", policy="radix",
        device=device, block_size=block_size, max_cache_blocks=max_cache_blocks,
        temperature=temperature, seed=seed,
    )

    print_comparison_table([no_cache, flat, radix])

    return {"no_cache": no_cache, "flat_cache": flat, "radix_tree": radix}
