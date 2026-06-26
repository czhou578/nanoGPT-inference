"""
Profiled Inference Engine - Instrumented copy of nanogpt-interleaving.py.

This file duplicates the model architecture and generate logic from
nanogpt-interleaving.py with profiling spans inserted at key points.
The duplication is intentional - same pattern as benchmarks/eval_runs.py -
to avoid importing from files with top-level training code.

The profiled functions are:
  - scheduled_generate_profiled:  non-interleaved scheduling with per-step tracing
  - interleaved_generate_profiled: interleaved prefill+decode with per-step tracing

Both record spans for: scheduling, memory (cache assembly/disassembly),
prefill forward passes, decode forward passes, sampling, and request
lifecycle events.
"""

import hashlib
import heapq
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from profiler import trace
from profiler.instrument import trace_span, trace_event


# ══════════════════════════════════════════════════════════════════════════════
#  Model Architecture (identical to nanogpt-interleaving.py)
# ══════════════════════════════════════════════════════════════════════════════

# These are set by configure() before use.
VOCAB_SIZE = None
BLOCK_SIZE = None
N_EMBD = None
N_HEAD = None
N_LAYER = None
DROPOUT = None
DEVICE = None


def configure(*, vocab_size, block_size, n_embd, n_head, n_layer, dropout, device):
    """Set module-level config. Must be called before building the model."""
    global VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, DROPOUT, DEVICE
    VOCAB_SIZE = vocab_size
    BLOCK_SIZE = block_size
    N_EMBD = n_embd
    N_HEAD = n_head
    N_LAYER = n_layer
    DROPOUT = dropout
    DEVICE = device


class Head(nn.Module):
    """One head of self-attention - stateless (no internal cache)."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, past_k=None, past_v=None, attn_mask=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if not self.training:
            if past_k is not None:
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)

            T_full = k.shape[1]
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

            causal_mask = torch.ones(T, T_full, device=x.device, dtype=torch.bool)
            if T > 1:
                new_token_mask = self.tril[:T, :T]
                causal_mask[:, -T:] = new_token_mask

            causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)

            if attn_mask is not None:
                new_valid = torch.ones(B, 1, T, device=x.device, dtype=torch.bool)
                full_pad_mask = torch.cat([attn_mask, new_valid], dim=-1)
                causal_mask = causal_mask & full_pad_mask

            wei = wei.masked_fill(~causal_mask, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out, k, v

        else:
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out, None, None


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, past_kv=None, attn_mask=None):
        if past_kv is None:
            past_kv = [(None, None)] * len(self.heads)

        outputs, new_kvs = [], []
        for i, h in enumerate(self.heads):
            pk, pv = past_kv[i]
            out, nk, nv = h(x, pk, pv, attn_mask=attn_mask)
            outputs.append(out)
            new_kvs.append((nk, nv))

        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, new_kvs


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, past_kv=None, attn_mask=None):
        sa_out, new_kv = self.sa(self.ln1(x), past_kv, attn_mask=attn_mask)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_kv


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, pos=None, past_kvs=None, attn_mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)

        if pos is None:
            pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        else:
            pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb

        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            x, block_kv = block(x, past_kvs[i], attn_mask=attn_mask)
            new_kvs.append(block_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_kvs


# ══════════════════════════════════════════════════════════════════════════════
#  Prefix Caching (identical to nanogpt-interleaving.py)
# ══════════════════════════════════════════════════════════════════════════════

NONE_HASH = b'\x00' * 16


def hash_block_tokens(parent_hash, token_ids):
    data = (parent_hash, tuple(token_ids))
    return hashlib.md5(str(data).encode()).digest()


@dataclass
class CachedBlock:
    block_hash: bytes
    token_ids: tuple
    kv_data: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]
    last_access_step: int = 0


class BlockCache:
    def __init__(self, max_blocks=64):
        self.max_blocks = max_blocks
        self.cache: Dict[bytes, CachedBlock] = {}
        self.current_step = 0

    def lookup(self, block_hash):
        block = self.cache.get(block_hash)
        if block is not None:
            block.last_access_step = self.current_step
        return block

    def insert(self, block_hash, token_ids, kv_data):
        if len(self.cache) >= self.max_blocks:
            self._evict_lru()
        self.cache[block_hash] = CachedBlock(
            block_hash=block_hash,
            token_ids=token_ids,
            kv_data=kv_data,
        )

    def _evict_lru(self):
        oldest = min(self.cache.values(), key=lambda b: b.last_access_step)
        del self.cache[oldest.block_hash]


def find_cached_prefix(block_cache, prompt_tokens, block_size):
    num_cached = 0
    parent_hash = NONE_HASH
    for start in range(0, len(prompt_tokens), block_size):
        end = start + block_size
        if end > len(prompt_tokens):
            break
        chunk = prompt_tokens[start:end]
        chunk_hash = hash_block_tokens(parent_hash, chunk)
        cached_block = block_cache.lookup(chunk_hash)
        if cached_block is None:
            break
        num_cached += block_size
        parent_hash = chunk_hash
    return num_cached


def load_cached_blocks(request, block_cache, prompt_tokens, block_size):
    parent_hash = NONE_HASH
    num_cached = 0
    for start in range(0, len(prompt_tokens), block_size):
        end = start + block_size
        if end > len(prompt_tokens):
            break
        chunk = prompt_tokens[start:end]
        chunk_hash = hash_block_tokens(parent_hash, chunk)
        cached = block_cache.lookup(chunk_hash)
        if cached is None:
            break
        for (layer, head), (k, v) in cached.kv_data.items():
            if (layer, head) in request.kv_cache:
                existing_k, existing_v = request.kv_cache[(layer, head)]
                request.kv_cache[(layer, head)] = (
                    torch.cat([existing_k, k.clone()], dim=1),
                    torch.cat([existing_v, v.clone()], dim=1),
                )
            else:
                request.kv_cache[(layer, head)] = (k.clone(), v.clone())
        num_cached += block_size
        parent_hash = chunk_hash
    request.prefill_cursor = num_cached
    return num_cached


def commit_completed_blocks(request, block_cache, block_size):
    num_full_blocks = request.prefill_cursor // block_size
    parent_hash = NONE_HASH
    for block_idx in range(num_full_blocks):
        start = block_idx * block_size
        end = start + block_size
        chunk = request.prompt_tokens[start:end]
        block_hash = hash_block_tokens(parent_hash, chunk)
        if block_idx >= request._committed_blocks:
            kv_data = {}
            for (layer, head), (k, v) in request.kv_cache.items():
                kv_data[(layer, head)] = (
                    k[:, start:end, :].clone(),
                    v[:, start:end, :].clone(),
                )
            block_cache.insert(block_hash, tuple(chunk), kv_data)
        parent_hash = block_hash
    request._committed_blocks = num_full_blocks


# ══════════════════════════════════════════════════════════════════════════════
#  Request + Scheduler (identical to nanogpt-interleaving.py)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Request:
    id: int
    prompt_tokens: List[int]
    max_new_tokens: int
    generated_tokens: List[int] = field(default_factory=list)
    status: str = "waiting"
    prefill_cursor: int = 0
    _committed_blocks: int = 0
    priority: int = 0
    arrival_time: int = 0
    kv_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    @property
    def tokens_so_far(self) -> List[int]:
        return self.prompt_tokens + self.generated_tokens

    @property
    def num_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def is_done(self) -> bool:
        return self.num_generated >= self.max_new_tokens

    @property
    def is_fully_prefilled(self) -> bool:
        return self.prefill_cursor == len(self.prompt_tokens)

    def clear_cache(self):
        self.kv_cache.clear()


class Scheduler:
    def __init__(self, policy="fcfs", max_batch_size=4, token_budget=16,
                 max_kv_tokens=22, block_size=4):
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.token_budget = token_budget
        self.max_kv_tokens = max_kv_tokens
        self.block_size = block_size
        self.block_cache = BlockCache()

        self.waiting = []
        self.prefilling = []
        self.active = []
        self.preempted = []

    def promote(self, req):
        self.prefilling.remove(req)
        req.status = "active"
        self.active.append(req)

    def complete(self, req):
        self.active.remove(req)
        req.status = "done"

    def _sort_key(self, req):
        if self.policy == "fcfs":
            return (0, req.arrival_time)
        elif self.policy == "priority":
            return (req.priority, req.arrival_time)

    def add_request(self, req):
        key = self._sort_key(req)
        heapq.heappush(self.waiting, (*key, req.id, req))

    def is_done(self):
        return not (self.waiting or self.prefilling or self.active)

    def _maybe_admit(self, step):
        if self.prefilling:
            return
        if not self.waiting:
            return
        kv_used = sum(
            len(req.prompt_tokens) + req.num_generated
            for req in self.active + self.prefilling
        )
        _, _, _, candidate = self.waiting[0]
        num_cached = find_cached_prefix(
            self.block_cache, candidate.prompt_tokens, self.block_size
        )
        actual_kv_cost = len(candidate.prompt_tokens) - num_cached
        if kv_used + actual_kv_cost > self.max_kv_tokens:
            return
        if len(self.active) + len(self.prefilling) >= self.max_batch_size:
            return

        heapq.heappop(self.waiting)
        load_cached_blocks(
            candidate, self.block_cache, candidate.prompt_tokens, self.block_size
        )
        candidate.arrival_time = step
        candidate.status = "prefilling"
        self.prefilling.append(candidate)

        # ── Profiling: record admission event ──
        trace_event(
            "request_admitted", "lifecycle",
            request_id=f"req_{candidate.id}",
            prompt_len=len(candidate.prompt_tokens),
            cached_tokens=num_cached,
        )

    def _maybe_preempt(self):
        kv_used = sum(
            len(req.prompt_tokens) + req.num_generated
            for req in self.active + self.prefilling
        )
        while self.active and kv_used > self.max_kv_tokens:
            victim = max(self.active, key=lambda r: (r.priority, -r.arrival_time))
            self.active.remove(victim)
            victim.clear_cache()
            victim.prefill_cursor = 0
            victim.generated_tokens = []
            victim.status = "waiting"
            self.preempted.append(victim)

            # ── Profiling: record preemption event ──
            trace_event(
                "request_preempted", "lifecycle",
                request_id=f"req_{victim.id}",
            )

            key = self._sort_key(victim)
            heapq.heappush(self.waiting, (*key, victim.id, victim))
            kv_used = sum(
                len(req.prompt_tokens) + req.num_generated
                for req in self.active + self.prefilling
            )

    def schedule(self, step: int):
        self.block_cache.current_step = step
        self._maybe_admit(step)
        self._maybe_preempt()
        prefill_req = self.prefilling[0] if self.prefilling else None
        decode_reqs = list(self.active)
        return prefill_req, decode_reqs


# ══════════════════════════════════════════════════════════════════════════════
#  Batch Assembly / Disassembly (identical + instrumented)
# ══════════════════════════════════════════════════════════════════════════════

def assemble_batch_cache(requests):
    """Gather per-request KV caches into batched tensors with left-padding."""
    B = len(requests)
    lengths = [req.kv_cache[(0, 0)][0].shape[1] for req in requests]
    max_t = max(lengths)
    pad_lengths = [max_t - t for t in lengths]

    attn_mask = torch.zeros(B, 1, max_t, device=DEVICE, dtype=torch.bool)
    for i, pad in enumerate(pad_lengths):
        attn_mask[i, 0, pad:] = True

    past_kvs = []
    for layer_idx in range(N_LAYER):
        block_kv = []
        for head_idx in range(N_HEAD):
            keys, values = [], []
            for i, req in enumerate(requests):
                k, v = req.kv_cache[(layer_idx, head_idx)]
                if pad_lengths[i] > 0:
                    hs = k.shape[2]
                    pad = torch.zeros(1, pad_lengths[i], hs, device=DEVICE)
                    k = torch.cat([pad, k], dim=1)
                    v = torch.cat([pad, v], dim=1)
                keys.append(k)
                values.append(v)
            block_kv.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        past_kvs.append(block_kv)

    return past_kvs, attn_mask, pad_lengths


def assemble_fused_batch(decode_reqs, prefill_req, chunk_size):
    """Build a single (B, T_max) input tensor + batched cache for fused pass."""
    num_new_tokens = []
    all_reqs = []

    for req in decode_reqs:
        all_reqs.append(req)
        num_new_tokens.append(1)

    if prefill_req:
        all_reqs.append(prefill_req)
        num_new_tokens.append(chunk_size)

    B = len(all_reqs)
    T_max = max(num_new_tokens)

    batch_tokens = []
    batch_positions = []

    for req in decode_reqs:
        pos_val = len(req.tokens_so_far) - 1
        row = [0] * (T_max - 1) + [pos_val]
        batch_positions.append(row)
        token_row = [0] * (T_max - 1) + [req.tokens_so_far[-1]]
        batch_tokens.append(token_row)

    if prefill_req:
        cursor = prefill_req.prefill_cursor
        chunk_positions = list(range(cursor, cursor + chunk_size))
        padding = [0] * (T_max - chunk_size)
        batch_positions.append(padding + chunk_positions)
        chunk = prefill_req.prompt_tokens[cursor:cursor + chunk_size]
        pad = [0] * (T_max - chunk_size)
        batch_tokens.append(pad + chunk)

    batch_positions = torch.tensor(batch_positions, device=DEVICE)
    batch_tokens = torch.tensor(batch_tokens, dtype=torch.long, device=DEVICE)

    if prefill_req and not prefill_req.kv_cache:
        head_size = N_EMBD // N_HEAD
        for li in range(N_LAYER):
            for hi in range(N_HEAD):
                prefill_req.kv_cache[(li, hi)] = (
                    torch.empty(1, 0, head_size, device=DEVICE),
                    torch.empty(1, 0, head_size, device=DEVICE),
                )

    past_kvs, attn_mask, pad_lengths = assemble_batch_cache(all_reqs)
    return batch_tokens, batch_positions, past_kvs, attn_mask, pad_lengths


def disassemble_batch_cache(requests, new_kvs, pad_lengths):
    """Scatter batched KV cache back to per-request storage."""
    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
            for i, req in enumerate(requests):
                pad = pad_lengths[i]
                req.kv_cache[(layer_idx, head_idx)] = (
                    batched_k[i:i + 1, pad:, :],
                    batched_v[i:i + 1, pad:, :],
                )


def disassemble_fused_cache(requests, new_kvs, num_new_tokens_per_req):
    """Scatter fused batch KV back to per-request storage."""
    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
            for i, req in enumerate(requests):
                t_new = num_new_tokens_per_req[i]
                k_new_valid = batched_k[i:i + 1, -t_new:, :]
                v_new_valid = batched_v[i:i + 1, -t_new:, :]

                if (layer_idx, head_idx) in req.kv_cache:
                    k_old, v_old = req.kv_cache[(layer_idx, head_idx)]
                    req.kv_cache[(layer_idx, head_idx)] = (
                        torch.cat([k_old, k_new_valid], dim=1),
                        torch.cat([v_old, v_new_valid], dim=1),
                    )
                else:
                    req.kv_cache[(layer_idx, head_idx)] = (k_new_valid, v_new_valid)


def build_tok_pos_kv(decode_reqs):
    """Build decode-only batch inputs from active requests."""
    batch_tokens = torch.cat([req._last_token for req in decode_reqs])
    batch_positions = torch.tensor(
        [[len(req.tokens_so_far) - 1] for req in decode_reqs], device=DEVICE
    )
    past_kvs, attn_mask, pad_lengths = assemble_batch_cache(decode_reqs)
    return batch_tokens, batch_positions, past_kvs, attn_mask, pad_lengths


# ══════════════════════════════════════════════════════════════════════════════
#  Profiled Generate Functions
# ══════════════════════════════════════════════════════════════════════════════

def interleaved_generate_profiled(model, requests, policy="fcfs",
                                  token_budget=16, max_kv_tokens=256):
    """
    Interleaved prefill+decode with full profiling instrumentation.

    This is the most interesting engine to profile because a single
    forward pass contains both prefill chunks and decode tokens.
    The trace shows exactly how they're packed together.
    """
    scheduler = Scheduler(policy, token_budget=token_budget,
                          max_kv_tokens=max_kv_tokens)
    step = 0

    for req in requests:
        req.arrival_time = step
        scheduler.add_request(req)

    model.eval()

    with torch.no_grad():
        while not scheduler.is_done():

            # ── Scheduling ──
            with trace_span("scheduler.schedule", "scheduling",
                            step=step,
                            num_waiting=len(scheduler.waiting),
                            num_active=len(scheduler.active),
                            num_prefilling=len(scheduler.prefilling)):
                prefill_req, decode_reqs = scheduler.schedule(step)

            chunk_size = 0
            remaining_budget = token_budget - len(decode_reqs)

            if remaining_budget > 0 and prefill_req is not None:
                tokens_left = len(prefill_req.prompt_tokens) - prefill_req.prefill_cursor
                chunk_size = min(remaining_budget, tokens_left)

            if chunk_size == 0 and not decode_reqs:
                step += 1
                continue

            # ── Batch Assembly ──
            with trace_span("assemble_fused_batch", "memory",
                            num_decode=len(decode_reqs),
                            chunk_size=chunk_size,
                            has_prefill=chunk_size > 0):
                batch_tokens, batch_positions, past_kvs, attn_mask, pad_lengths = \
                    assemble_fused_batch(
                        decode_reqs,
                        prefill_req if chunk_size > 0 else None,
                        chunk_size,
                    )

            # ── Forward Pass ──
            batch_size = batch_tokens.shape[0]
            seq_len = batch_tokens.shape[1]
            with trace_span("model_forward", "compute",
                            step=step,
                            batch_size=batch_size,
                            seq_len=seq_len,
                            num_decode=len(decode_reqs),
                            prefill_chunk=chunk_size):
                logits, _, new_kvs = model(
                    batch_tokens,
                    pos=batch_positions,
                    past_kvs=past_kvs,
                    attn_mask=attn_mask,
                )

            # ── Cache Disassembly ──
            all_reqs = decode_reqs[:]
            num_new_tokens_per_req = [1] * len(decode_reqs)

            if chunk_size > 0:
                all_reqs.append(prefill_req)
                num_new_tokens_per_req.append(chunk_size)

            with trace_span("disassemble_fused_cache", "memory",
                            num_requests=len(all_reqs)):
                disassemble_fused_cache(all_reqs, new_kvs, num_new_tokens_per_req)

            # ── Decode Sampling ──
            if len(decode_reqs) > 0:
                with trace_span("decode_sampling", "sampling",
                                num_decode=len(decode_reqs)):
                    logits_decode = logits[:len(decode_reqs), -1, :]
                    probs = F.softmax(logits_decode, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                for i, req in enumerate(decode_reqs):
                    req.generated_tokens.append(idx_next[i].item())
                    req._last_token = idx_next[i:i + 1]
                    if req.is_done:
                        trace_event(
                            "request_completed", "lifecycle",
                            request_id=f"req_{req.id}",
                            total_tokens=len(req.tokens_so_far),
                            generated=req.num_generated,
                        )
                        scheduler.complete(req)

            # ── Prefill Post-processing ──
            if chunk_size > 0:
                req_id = f"req_{prefill_req.id}"

                trace_event(
                    "prefill_chunk", "prefill",
                    request_id=req_id,
                    chunk_start=prefill_req.prefill_cursor,
                    chunk_end=prefill_req.prefill_cursor + chunk_size,
                    chunk_tokens=chunk_size,
                )

                prefill_req.prefill_cursor += chunk_size

                if prefill_req.is_fully_prefilled:
                    with trace_span("prefill_first_token", "sampling",
                                    request_id=req_id):
                        prefill_logits = logits[-1:, -1, :]
                        probs = F.softmax(prefill_logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)

                    prefill_req.generated_tokens.append(idx_next.item())
                    prefill_req._last_token = idx_next

                    with trace_span("commit_blocks", "cache_management",
                                    request_id=req_id):
                        commit_completed_blocks(
                            prefill_req, scheduler.block_cache, scheduler.block_size
                        )

                    trace_event(
                        "prefill_complete", "lifecycle",
                        request_id=req_id,
                        prompt_len=len(prefill_req.prompt_tokens),
                    )
                    scheduler.promote(prefill_req)

            step += 1

    return scheduler
