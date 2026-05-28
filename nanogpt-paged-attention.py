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

    k_cat = torch.cat(k_parts, dim=1).unsqueeze(0)
    v_cat = torch.cat(v_parts, dim=1).unsqueeze(0)

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