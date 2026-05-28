import torch
import torch.nn as nn
from torch.nn import functional as F
import heapq
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss, _ = model(X, Y)  # unpack 3 return values now
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

NONE_HASH = b'\x00' * 16  # sentinel for the first block (no parent)

def hash_block_tokens(parent_hash, token_ids):
    """Compute a chained content hash for a KV block."""
    data = (parent_hash, tuple(token_ids))
    return hashlib.md5(str(data).encode()).digest()

@dataclass
class CachedBlock:
    """A cached KV block with its content hash."""

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
            self.evict_lru()

        self.cache[block_hash] = CachedBlock(
            block_hash=block_hash,
            token_ids=token_ids,
            kv_data=kv_data,
            last_access_step=self.current_step
        )
    
    def evict_lru(self):
        oldest = min(self.cache.values(), key=lambda b: b.last_access_step)
        del self.cache[oldest.block_hash]

def find_cached_prefix(block_cache: BlockCache, prompt_tokens: List[int], block_size: int):
    """
    Walk the prompt left-to-right in block-sized chunks.
    Return the number of tokens that are fully cached
    """
    num_cached = 0
    parent_hash = NONE_HASH

    for i in range(len(prompt_tokens) // block_size):
        chunk = prompt_tokens[i * block_size : (i + 1) * block_size]
        parent_hash = hash_block_tokens(parent_hash, chunk)

        if block_cache.lookup(parent_hash) is None: break
        num_cached += block_size
    
    return num_cached


def load_cached_blocks(request, block_cache, prompt_tokens, block_size):
    """ 
    Load cached KV blocks onto a request and return how many tokens were cached. 
    Sets request.prefill_cursor to skip past the cached potion
    """

    parent_hash = NONE_HASH
    num_cached = 0

    for i in range(len(prompt_tokens) // block_size):
        chunk = prompt_tokens[i * block_size : (i + 1) * block_size]
        parent_hash = hash_block_tokens(parent_hash, chunk)

        cached = block_cache.lookup(parent_hash)
        if cached is None: break

        for (layer, head), (k, v) in cached.kv_data.items():
            if (layer, head) in request.kv_cache:
                existing_k, existing_v = request.kv_cache[(layer, head)]

                request.kv_cache[(layer, head)] = (
                    torch.cat([existing_k, k.clone()], dim=1),
                    torch.cat([existing_v, v.clone()], dim=1)
                )
            else:
                request.kv_cache[(layer, head)] = (k.clone(), v.clone())
        
        num_cached += block_size
        parent_hash = parent_hash
    
    request.prefill_cursor = num_cached
    return num_cached

@dataclass
class Request:
    """Each in-flight generation carries its own state and KV cache."""
    id: int
    prompt_tokens: List[int]          # the original encoded prompt
    max_new_tokens: int               # how many tokens this request wants
    generated_tokens: List[int] = field(default_factory=list)
    status: str = "waiting"           # "waiting" -> "prefilling" -> "active" -> "done"
    prefill_cursor: int = 0
    _committed_blocks: int = 0

    # Hint 2: Per-request KV cache, keyed by (layer_idx, head_idx)
    # Each value is a (key_tensor, value_tensor) tuple of shape (1, T_i, head_size)
    # T_i grows by 1 each decode step — different requests have different T_i
    kv_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    @property
    def tokens_so_far(self) -> List[int]:
        """Full sequence: prompt + everything generated."""
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
    def __init__(self, policy="fcfs", max_batch_size=4, token_budget=16, max_kv_tokens=22, block_size=4):
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

        kv_used = sum(len(req.prompt_tokens) + req.num_generated for req in self.active + self.prefilling)

        _, _, _, candidate = self.waiting[0]

        num_cached = find_cached_prefix(self.block_cache, candidate.prompt_tokens, self.block_size)
        
        actual_kv_cost = len(candidate.prompt_tokens) - num_cached

        if kv_used + actual_kv_cost > self.max_kv_tokens:
            return
        
        if len(self.active) + len(self.prefilling) >= self.max_batch_size: return

        heapq.heappop(self.waiting)
        load_cached_blocks(candidate, self.block_cache, candidate.prompt_tokens, self.block_size)
        candidate.arrival_time = step
        candidate.status = "prefilling"
        self.prefilling.append(candidate)
    
    def _maybe_preempt(self):
        kv_used = sum(len(req.prompt_tokens) + req.num_generated for req in self.active + self.prefilling)

        while self.active and kv_used > self.max_kv_tokens:
            victim = max(self.active, key=lambda r: (r.priority, -r.arrival_time))
            self.active.remove(victim)
            victim.clear_cache()
            victim.prefill_cursor = 0
            victim.status = "waiting"
            self.preempted.append(victim)

            key = self._sort_key(victim)
            heapq.heappush(self.waiting, (*key, victim.id, victim))
            kv_used = sum(len(req.prompt_tokens) + req.num_generated for req in self.active + self.prefilling)

    def schedule(self, step: int):
        """
        Returns:
            prefill_req:  Request | None  — one request getting a prefill chunk (or None)
            decode_reqs:  List[Request]   — all requests currently being decoded (active)

        """
        self.block_cache.current_step = step

        self._maybe_admit(step)       # promote waiting → prefilling if memory allows
        self._maybe_preempt()         # evict if over memory budget

        prefill_req = self.prefilling[0] if self.prefilling else None
        decode_reqs = list(self.active)

        return prefill_req, decode_reqs


class Head(nn.Module):
    """One head of self-attention — now STATELESS (no internal cache)."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_k=None, past_v=None, attn_mask=None):
        """
        Args:
            x:      (B, T, C)       input embeddings
            past_k: (B, T_past, hs) cached keys, or None
            past_v: (B, T_past, hs) cached values, or None
        Returns:
            out:   (B, T, hs)           attention output
            new_k: (B, T_past+T, hs)    updated key cache   (None during training)
            new_v: (B, T_past+T, hs)    updated value cache  (None during training)
        """
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        if not self.training:
            if past_k is not None:
                T_past = past_k.shape[1]
                # ── Decode step: append new K/V onto cached past ──
                k = torch.cat([past_k, k], dim=1)  # (B, T_past + T, hs)
                v = torch.cat([past_v, v], dim=1)

                # Q attends over full cache — no causal mask needed (T=1)
                wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

            # ── Causal mask for multi-token prefill continuation ──
            # When T > 1, new tokens must not attend to future tokens
            # within the chunk. (When T == 1, this is a no-op — skip it.)                

                if T > 1:
                    past_part = torch.ones(T, T_past, device=wei.device, dtype=torch.bool)
                    new_part = torch.tril(torch.ones(T, T, device=wei.device, dtype=torch.bool))
                    causal_mask = torch.cat([past_part, new_part], dim=-1)
                    wei = wei.masked_fill(~causal_mask, float('-inf'))

                if attn_mask is not None:
                    new_valid = torch.ones(B, 1, T, device=wei.device, dtype=torch.bool)
                    full_mask = torch.cat([attn_mask, new_valid], dim=-1)
                    wei = wei.masked_fill(~full_mask, float('-inf'))

                wei = F.softmax(wei, dim=-1)
                wei = self.dropout(wei)
                out = wei @ v
            else:
                # ── Prefill step: full prompt, needs causal mask ──
                wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
                wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
                wei = F.softmax(wei, dim=-1)
                wei = self.dropout(wei)
                out = wei @ v

            return out, k, v   # return updated cache
        else:
            # ── Training path — unchanged, no cache ──
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out, None, None


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, attn_mask=None):
        """
        Args:
            x:       (B, T, C)
            past_kv: list of (past_k, past_v) per head, or None
        Returns:
            out:    (B, T, n_embd)
            new_kv: list of (new_k, new_v) per head
        """
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


class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, past_kv=None, attn_mask=None):
        """
        Returns:
            x:      (B, T, n_embd)
            new_kv: list of (new_k, new_v) per head in this block
        """
        sa_out, new_kv = self.sa(self.ln1(x), past_kv, attn_mask=attn_mask)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_kv


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # ModuleList instead of Sequential so we can pass per-block cache
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, pos=None, past_kvs=None, attn_mask=None):
        """
        Args:
            idx:      (B, T) token indices
            targets:  (B, T) target indices, or None
            pos:      (B, T) explicit position indices, or None (uses arange)
            past_kvs: list-of-lists cache structure, or None
                      past_kvs[layer][head] = (key_tensor, value_tensor)
        Returns:
            logits:   (B, T, vocab_size)
            loss:     scalar or None
            new_kvs:  updated cache with same structure as past_kvs
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)

        if pos is None:
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        else:
            pos_emb = self.position_embedding_table(pos)  # (B, T, C)

        x = tok_emb + pos_emb  # (B, T, C)

        # Thread cache through each block
        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            x, block_kv = block(x, past_kvs[i], attn_mask=attn_mask)
            new_kvs.append(block_kv)

        x = self.ln_f(x)          # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_kvs

    def generate(self, idx, max_new_tokens):
        """Original generate (no cache, full recompute) for reference."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss, _ = model(xb, yb)  # _ discards the cache during training
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Quick sanity check with the original no-cache generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

def assemble_batch_cache(requests):
    """
    Gather per-request KV caches into batched tensors.
    LEFT-pads shorter caches so new tokens always land at the right edge.

    Big problem: You have 3 active requests. Each owns its own KV cache. You need to feed them to the model as one
    batched tensor. But their caches have different lengths:

    Returns:
        past_kvs:    batched cache structure  [layer][head] = (B, T_max, hs)
        attn_mask:   (B, 1, T_max) bool — True = valid, False = padding
        pad_lengths: list of int — how many pad positions per request (for disassembly)
    """

    B = len(requests)
    lengths = [req.kv_cache[(0, 0)][0].shape[1] for req in requests]
    max_t = max(lengths)

    pad_lengths = [max_t - t for t in lengths] # pad lengths for every position in t

    attn_mask = torch.zeros(B, 1, max_t, device=device, dtype=torch.bool)

    for i, pad in enumerate(pad_lengths):
        attn_mask[i, 0, pad:] = True

    past_kvs = []

    for layer_idx in range(n_layer):
        block_kv = []

        for head_idx in range(n_head):
            keys, values = [], []

            for i, req in enumerate(requests):
                k, v = req.kv_cache[(layer_idx, head_idx)]
                if pad_lengths[i] > 0:
                    hs = k.shape[2]
                    pad = torch.zeros(1, pad_lengths[i], hs, device=device)
                    k = torch.cat([pad, k], dim=1)
                    v = torch.cat([pad, v], dim=1)

                keys.append(k)
                values.append(v)

            block_kv.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        past_kvs.append(block_kv)

    return past_kvs, attn_mask, pad_lengths

def disassemble_batch_cache(requests, new_kvs, pad_lengths):
    """
    Scatter batched KV cache back to per-request storage.
    After Head's torch.cat, each row is (T_max + 1) — strip the left-padding.
    """
    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
            for i, req in enumerate(requests):
                pad = pad_lengths[i]
                req.kv_cache[(layer_idx, head_idx)] = (
                    batched_k[i : i + 1, pad:, :],      # (1, T_i + 1, hs)
                    batched_v[i : i + 1, pad:, :],
                )

def commit_completed_blocks(request: Request, block_cache: BlockCache, block_size: int):
    """
    After a prefill step, check if any new full blocks were completed.
    If so, insert them into the global cache.
    """

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
                    v[:, start:end, :].clone()
                )
            
            block_cache.insert(block_hash, tuple(chunk), kv_data)
        
        parent_hash = block_hash
    
    request._committed_blocks = num_full_blocks

def scheduled_generate(model, requests, policy="fcfs", token_budget=16, max_kv_tokens=256):
    scheduler = Scheduler(policy, token_budget=token_budget, max_kv_tokens=max_kv_tokens)

    step = 0

    for req in requests:
        req.arrival_time = step
        scheduler.add_request(req)
    
    model.eval()

    with torch.no_grad():
        while not scheduler.is_done():

            prefill_req, decode_reqs = scheduler.schedule(step)

            if prefill_req:

                prefill_chunk_tokens = []

                remaining_budget = token_budget - len(scheduler.active)

                if remaining_budget > 0 and scheduler.prefilling:
                    p_req = scheduler.prefilling[0]

                    tokens_left = len(p_req.prompt_tokens) - p_req.prefill_cursor
                    chunk_size = min(remaining_budget, tokens_left)

                    chunk_start = p_req.prefill_cursor 

                    chunk_tokens = p_req.prompt_tokens[chunk_start: chunk_start + chunk_size]

                    prefill_chunk_tokens = torch.tensor([chunk_tokens], dtype=torch.long, device=device)

                    p_req.prefill_cursor += chunk_size

                if len(prefill_chunk_tokens) == 0 and not scheduler.active:
                    step += 1
                    continue

                if len(prefill_chunk_tokens) > 0:
                    pos = torch.arange(chunk_start, chunk_start + chunk_size, device=device).unsqueeze(0)

                    if p_req.kv_cache:
                        #  This format is wrong 
                        # logits, _, new_kvs = model(prefill_chunk_tokens, past_kvs=req.kv_cache)
                        # list[list[(k, v)]] is shape
                        
                        past_kvs = []
                        for layer_idx in range(n_layer):
                            block_kv = [(p_req.kv_cache[(layer_idx, hi)]) for hi in range(n_head)] 
                            past_kvs.append(block_kv)
                        
                        logits, _, new_kvs = model(prefill_chunk_tokens, pos=pos, past_kvs=past_kvs)

                    else:
                        logits, _, new_kvs = model(prefill_chunk_tokens, pos=pos)

                    for li, bkv in enumerate(new_kvs):
                        for hi, (k, v) in enumerate(bkv):
                            p_req.kv_cache[(li, hi)] = (k, v)
            
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                    if prefill_req.is_fully_prefilled:
                        prefill_req.generated_tokens.append(idx_next.item())
                        prefill_req._last_token = idx_next
                        commit_completed_blocks(prefill_req, scheduler.block_cache, BLOCK_SIZE)
                        # print(f"[step {step}] Committed {num_new_blocks} blocks from req {req.id} to cache "
                        #     f"(cache size: {len(block_cache.cache)}/{block_cache.max_blocks})")                        
                        scheduler.promote(prefill_req)
    
            if decode_reqs:

                batch_tokens = torch.cat([req._last_token for req in scheduler.active])

                batch_positions = torch.tensor([[len(req.tokens_so_far) - 1] for req in scheduler.active], device=device)

                past_kvs, attn_mask, pad_lengths = assemble_batch_cache(scheduler.active)

                logits, _, new_kvs = model(
                    batch_tokens,
                    pos=batch_positions,
                    past_kvs=past_kvs,
                    attn_mask=attn_mask
                )

                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                disassemble_batch_cache(scheduler.active, new_kvs, pad_lengths)

                for i, req in enumerate(decode_reqs):
                    req.generated_tokens.append(idx_next[i].item())
                    req._last_token = idx_next[i : i + 1]
                
                for req in decode_reqs:
                    if req.is_done:
                        scheduler.complete(req)
        
            step += 1
    
    return scheduler