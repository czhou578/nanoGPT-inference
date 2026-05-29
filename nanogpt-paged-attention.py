import enum
from numpy import dtype
import torch
import torch.nn as nn
import time
import heapq
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import hashlib

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
    token_ids: tuple                    # the tokens this block covers
    kv_data: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]
    # kv_data[(layer, head)] = (k, v), each (1, BLOCK_SIZE, head_size)
    last_access_step: int = 0          # for LRU eviction

class BlockCache:
    def __init__(self, max_blocks=64):
        self.max_blocks = max_blocks
        self.cache: Dict[bytes, CachedBlock] = {}  # hash → CachedBlock
        self.current_step = 0

    def lookup(self, block_hash) -> CachedBlock | None:
        """Look up a block by its content hash."""
        block = self.cache.get(block_hash)
        if block is not None:
            block.last_access_step = self.current_step  # touch for LRU
        return block

    def insert(self, block_hash, token_ids, kv_data):
        """Insert a completed block into the cache."""
        if len(self.cache) >= self.max_blocks:
            self._evict_lru()
            
        self.cache[block_hash] = CachedBlock(
            block_hash=block_hash,
            token_ids=token_ids,
            kv_data=kv_data,
        )

    def _evict_lru(self):
        """Evict the least-recently-used block."""
        oldest = min(self.cache.values(), key=lambda b: b.last_access_step)
        del self.cache[oldest.block_hash]

class KVBlockPool:
    """
    Pre-allocated GPU memory pool for KV cache blocks.
    
    Physical layout: one big tensor per (layer, head, k/v).
    Shape: (num_physical_blocks, block_size, head_size)
    
    Block i occupies pool[i, :, :] — a fixed-size (block_size, head_size) slab.
    """

    def __init__(self, num_blocks, block_size, n_layer, n_head, head_size, device):
       self.num_blocks = num_blocks
       self.block_size = block_size

       self.k_pool = {}
       self.v_pool = {}

       for layer in range(n_layer):
         for head in range(n_head):
           self.k_pool[(layer, head)] = torch.zeros(
             num_blocks,
             block_size,
             head_size,
             device=device
           )

           self.v_pool[(layer, head)] = torch.zeros(
             num_blocks,
             block_size,
             head_size,
             device=device
           )

def find_cached_prefix(block_cache: BlockCache, prompt_tokens, block_size):
    """
        Walk the prompt left-to-right in block-sized chunks.
        Return the number of tokens that are fully cached
    """

    num_cached = 0
    parent_hash = NONE_HASH

    for start in range(0, len(prompt_tokens), block_size):
        end = start + block_size
        if end > len(prompt_tokens): break

        chunk = prompt_tokens[start:end]
        chunk_hash = hash_block_tokens(parent_hash, chunk)

        cached_block = block_cache.lookup(chunk_hash)

        if cached_block is None: break

        num_cached += block_size
        parent_hash = chunk_hash
    
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

    block_table: List[int] = field(default_factory=list)
    num_filled_slots: int = 0
    priority: int = 0
    arrival_time: int = 0

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

    def clear_cache(self, block_allocator):
        block_allocator.free_blocks_for_request(self.block_table)
        self.block_table = []
        self.num_filled_slots = 0

def write_kv_to_pool(pool, block_table, block_size, start_pos, k_new, v_new, layer, head):
    """
    Write new KV data into the physical pool using the block table.
    
    Args:
        pool:        KVBlockPool
        block_table: list of physical block indices for this request
        block_size:  tokens per block
        start_pos:   logical position of the first new token
        k_new:       (1, T_new, head_size) — new key data
        v_new:       (1, T_new, head_size) — new value data
    """

    T_new = k_new.shape[1]

    for t in range(T_new):
        logical_pos = start_pos + t
        block_idx = logical_pos // block_size
        slot_idx = logical_pos % block_size

        phys_block = block_table[block_idx]

        pool.k_pool[(layer, head)][phys_block, slot_idx, :] = k_new[0, t, :]
        pool.v_pool[(layer, head)][phys_block, slot_idx, :] = v_new[0, t, :]

def maybe_allocate_block(request, block_allocator, block_size):
    """Allocate a new physical block if the current one is full."""

    if request.num_filled_slots % block_size == 0:
        new_block = block_allocator.allocate_one()
        request.block_table.append(new_block)

def gather_kv_from_pool(pool, block_table, block_size, num_filled, layer, head):
    """
    Gather a request's KV cache from the physical pool into a contiguous tensor.
    
    Returns:
        k: (1, num_filled, head_size)
        v: (1, num_filled, head_size)
    """

    if num_filled == 0:
        hs = pool.k_pool[(layer, head)].shape[-1]
        device = pool.k_pool[(layer, head)].device
        return (
            torch.empty(1, 0, hs, device=device),
            torch.empty(1, 0, hs, device=device),
        )

    num_full_blocks = num_filled // block_size
    trailing_slots = num_filled % block_size

    k_parts, v_parts = [], []

    for i in range(num_full_blocks):
        phys_block = block_table[i]

        k_parts.append(pool.k_pool[(layer, head)][phys_block, :, :])
        v_parts.append(pool.v_pool[(layer, head)][phys_block, :, :])
    
    if trailing_slots > 0:
        phys_block = block_table[num_full_blocks]

        k_parts.append(pool.k_pool[(layer, head)][phys_block, :trailing_slots, :])
        v_parts.append(pool.v_pool[(layer, head)][phys_block, :trailing_slots, :])

    k_cat = torch.cat(k_parts, dim=0).unsqueeze(0)
    v_cat = torch.cat(v_parts, dim=0).unsqueeze(0)

    return k_cat, v_cat


class BlockAllocator:
    def __init__(self, num_blocks, block_size=4):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))

    def allocate_one(self):
        if not self.free_blocks:
            raise MemoryError("No free blocks available")
        
        return self.free_blocks.pop()

    def allocate_n(self, n):
        if len(self.free_blocks) < n:
            raise MemoryError("No free blocks available")
        
        return [self.free_blocks.pop() for _ in range(n)]

    def free_blocks_for_request(self, block_table):
        self.free_blocks.extend(block_table)

    @property
    def num_free(self):
        return len(self.free_blocks)


class Scheduler:
    def __init__(self, policy="fcfs", max_batch_size=4, token_budget=16, max_kv_tokens=22, block_size=4):
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.token_budget = token_budget
        self.max_kv_tokens = max_kv_tokens
        self.block_size = block_size
        self.block_cache = BlockCache()
        self.block_allocator = None
        self.current_compute_tokens = 0

        self.waiting = []
        self.prefilling = []
        self.active = []
        self.preempted = []     

    def promote(self, req):
        self.prefilling.remove(req)
        req.status = "active"
        self.active.append(req)
    
    def complete(self, req):
        if req in self.active:
            
            self.active.remove(req)
        req.status = "done"
        self.block_allocator.free_blocks_for_request(req.block_table) 

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
        if not self.waiting: return

        _, _, _, candidate = self.waiting[0]
        prompt_len = len(candidate.prompt_tokens)

        blocks_needed = (prompt_len + self.block_size - 1) // block_size

        if self.block_allocator.num_free < blocks_needed:
            return

        needed_compute = min(prompt_len, self.token_budget)

        if self.current_compute_tokens + needed_compute > self.token_budget: return

        heapq.heappop(self.waiting)
        candidate.status = "prefilling"

        candidate.block_table = self.block_allocator.allocate_n(blocks_needed)
        candidate.num_filled_slots = 0

        self.prefilling.append(candidate)     


    def _maybe_preempt(self):
        kv_used = sum(len(req.prompt_tokens) + req.num_generated for req in self.active + self.prefilling)

        while self.active and kv_used > self.max_kv_tokens:
            victim = max(self.active, key=lambda r: (r.priority, -r.arrival_time))
            self.active.remove(victim)
            victim.clear_cache(self.block_allocator)
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
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)

            T_full = k.shape[1]

            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T_full)

            causal_mask = torch.ones(T, T_full, device=x.device, dtype=torch.bool)

            if T > 1:
                new_token_mask = self.tril[:T, :T]
                causal_mask[:, -T:] = new_token_mask

            causal_mask = causal_mask.unsqueeze(0).expand(B, -1,- 1)
        
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

def assemble_paged_cache(requests, pool, block_size):
    """
    Gather per-request KV from the paged pool into batched tensors.
    Same interface as assemble_batch_cache — returns left-padded batched cache.
    """


    B = len(requests)

    lengths = [req.num_filled_slots for req in requests]
    max_t = max(lengths) if lengths else 0
    pad_lengths = [max_t - t for t in lengths]

    attn_mask = torch.zeros(B, 1, max_t, device=device, dtype=torch.bool)

    for i, pad in enumerate(pad_lengths):
        attn_mask[i, :, pad:] = True
    
    past_kvs = []

    for layer_idx in range(n_layer):
        block_kv = []

        for head_idx in range(n_head):
            keys, values = [], []

            for i, req in enumerate(requests):
                k, v  = gather_kv_from_pool(pool, req.block_table, block_size, req.num_filled_slots, layer_idx, head_idx)
                
                if pad_lengths[i] > 0:
                    hs = k.shape[2]
                    pad_tensor = torch.zeros(1, pad_lengths[i], hs, device=device)
                    k = torch.cat([pad_tensor, k], dim=1)
                    v = torch.cat([pad_tensor, v], dim=1)
                
                keys.append(k)
                values.append(v)
            
            block_kv.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        
        past_kvs.append(block_kv)

    return past_kvs, attn_mask, pad_lengths

def disassemble_paged_cache(requests, new_kvs, pad_lengths, pool, block_size):
    """
    Scatter new KV data from model output back into the paged pool.
    Each request gets 1 new KV entry (decode token).
    """

    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
            for i, req in enumerate(requests):
                pad = pad_lengths[i]
                
                k_new = batched_k[i: i + 1, pad:, :]
                v_new = batched_v[i: i + 1, pad:, :]

                write_kv_to_pool(pool, req.block_table, block_size,
                req.num_filled_slots, k_new, v_new, layer_idx, head_idx) 

    for req in requests:
        req.num_filled_slots += 1

def disassemble_paged_fused(all_reqs, new_kvs, num_new_per_req, pool, block_size):
    """Like disassemble_paged_cache but handles variable new tokens per row."""
    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
            for i, req in enumerate(all_reqs):
                t_new = num_new_per_req[i]
                k_new = batched_k[i:i+1, -t_new:, :]
                v_new = batched_v[i:i+1, -t_new:, :]
                
                write_kv_to_pool(
                    pool, req.block_table, block_size,
                    req.num_filled_slots,
                    k_new, v_new, layer_idx, head_idx
                )
    
    for i, req in enumerate(all_reqs):
        req.num_filled_slots += num_new_per_req[i]        

def assemble_fused_batch(decode_reqs: List[Request], prefill_req, chunk_size, pool, block_size):
    """
    Build a single (B, T_max) input tensor + batched cache for the fused forward pass.

    Args:
        decode_reqs:  list of active Request objects (each contributes 1 token)
        prefill_req:  the request being prefilled (contributes chunk_size tokens), or None
        chunk_size:   number of prefill tokens this step

    Returns:
        batch_tokens:   (B, T_max) input tensor
        batch_positions: (B, T_max) position indices
        past_kvs:       batched cache [layer][head] = (B, T_max_cache, hs)
        attn_mask:      (B, 1, T_max_cache) bool mask for cached positions
        pad_info:       dict with per-row metadata for disassembly
    """

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

        chunk = prefill_req.prompt_tokens[cursor: cursor + chunk_size]
        pad = [0] * (T_max - chunk_size)

        batch_tokens.append(pad + chunk)

    batch_positions = torch.tensor(batch_positions, device=device)        
    batch_tokens = torch.tensor(batch_tokens, dtype=torch.long, device=device)  
    # Assemble KV cache 
        
    past_kvs, attn_mask, pad_lengths = assemble_paged_cache(all_reqs, pool, block_size)    
    return batch_tokens, batch_positions, past_kvs, attn_mask, pad_lengths

def commit_completed_blocks(request: Request, block_cache: BlockCache, block_size, pool):
    """
    After a prefill step, check if any new full blocks were completed.
    If so, insert them into the global cache.
    """

    total_tokens = len(request.prompt_tokens) + request.num_generated
    num_full_blocks = total_tokens // block_size    
    
    parent_hash = NONE_HASH
    
    for block_idx in range(num_full_blocks):
        start = block_idx * block_size
        end = start + block_size

        chunk = request.tokens_so_far[start:end]
        block_hash = hash_block_tokens(parent_hash, chunk)

        if block_idx >= request._committed_blocks:
            kv_data = {}
            phys_block = request.block_table[block_idx]

            for layer in range(n_layer):
                for head in range(n_head):
                    kv_data[(layer, head)] = (
                        pool.k_pool[(layer, head)][phys_block].unsqueeze(0).clone(),
                        pool.v_pool[(layer, head)][phys_block].unsqueeze(0).clone(),
                    )
            
            block_cache.insert(block_hash, tuple(chunk), kv_data)

        parent_hash = block_hash
    
    request._committed_blocks = num_full_blocks
           
def interleaved_generate(model, requests, policy="fcfs", token_budget=16, max_kv_tokens=256):
    scheduler = Scheduler(policy, token_budget=token_budget, max_kv_tokens=max_kv_tokens)
    head_size = n_embd // n_head
    num_blocks = max_kv_tokens // block_size
    pool = KVBlockPool(num_blocks, block_size, n_layer, n_head, head_size, device)
    scheduler.block_allocator = BlockAllocator(num_blocks)

    step = 0

    for req in requests:
        req.arrival_time = step
        scheduler.add_request(req)
    
    model.eval()

    with torch.no_grad():
        while not scheduler.is_done():
            prefill_req, decode_reqs = scheduler.schedule(step)

            chunk_size = 0
            remaining_budget = token_budget - len(decode_reqs)

            if remaining_budget > 0 and prefill_req is not None:
                tokens_left = len(prefill_req.prompt_tokens) - prefill_req.prefill_cursor

                chunk_size = min(remaining_budget, tokens_left)
            
            if chunk_size == 0 and not decode_reqs:
                step += 1
                continue
            
            for req in decode_reqs:
                maybe_allocate_block(req, scheduler.block_allocator, block_size)

            batch_tokens, batch_positions, past_kvs, attn_mask, pad_lengths = assemble_fused_batch(
                decode_reqs, 
                prefill_req if chunk_size > 0 else None, 
                chunk_size,
                pool,
                block_size
            )

            logits, _, new_kvs = model(
                batch_tokens,
                pos=batch_positions,
                past_kvs=past_kvs,
                attn_mask=attn_mask
            )     

            all_reqs = decode_reqs[:]
            num_new_tokens_per_req = [1] * len(decode_reqs)

            if chunk_size > 0:
                all_reqs.append(prefill_req)
                num_new_tokens_per_req.append(chunk_size)
                
            disassemble_paged_fused(all_reqs, new_kvs, num_new_tokens_per_req, pool, block_size)  # CHANGED

            if len(decode_reqs) > 0:
                logits_decode = logits[:len(decode_reqs), -1, :]
                probs = F.softmax(logits_decode, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                for i, req in enumerate(decode_reqs):
                    req.generated_tokens.append(idx_next[i].item())
                    req._last_token = idx_next[i : i + 1]
                    if req.is_done:
                        scheduler.complete(req)
            
            if chunk_size > 0:
                prefill_req.prefill_cursor += chunk_size
            
                if prefill_req.is_fully_prefilled:
                    prefill_logits = logits[-1:, -1, :]
                    probs = F.softmax(prefill_logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    
                    prefill_req.generated_tokens.append(idx_next.item())
                    prefill_req._last_token = idx_next
                    commit_completed_blocks(prefill_req, scheduler.block_cache, block_size, pool)
                    scheduler.promote(prefill_req)

            step += 1

    return scheduler   
