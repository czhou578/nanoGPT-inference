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
