# vLLM Block Manager vs NanoGPT Paged Attention — Side-by-Side Comparison

This document compares the block management code in [vLLM v1](../../vllm/vllm/v1/core/) with our educational implementation in [nanogpt-paged-attention.py](../../nanogpt-paged-attention.py).

The goal is to show how the same concepts map between a production system (vLLM — 80K+ lines across the block management subsystem) and a minimal educational one (~350 lines of block management code).

---

## Architecture Overview

### vLLM's Block Management Stack

```
┌─────────────────────────────────────────────┐
│  KVCacheManager (kv_cache_manager.py)       │ ← Top-level API
│    Coordinates across cache groups           │
├─────────────────────────────────────────────┤
│  SingleTypeKVCacheManager                    │ ← Per-group block management
│    (single_type_kv_cache_manager.py)         │    Logical → physical mapping
├─────────────────────────────────────────────┤
│  BlockPool (block_pool.py)                   │ ← Physical block allocation
│    Free list, eviction, ref counting         │
├─────────────────────────────────────────────┤
│  FreeKVCacheBlockQueue (kv_cache_utils.py)   │ ← Doubly-linked free list
│    O(1) alloc, free, and mid-list removal    │
├─────────────────────────────────────────────┤
│  KVCacheBlock (kv_cache_utils.py)            │ ← Block metadata
│    block_id, ref_cnt, block_hash, linked     │
│    list pointers                             │
└─────────────────────────────────────────────┘
```

### NanoGPT's Block Management Stack

```
┌─────────────────────────────────────────────┐
│  Scheduler (nanogpt-paged-attention.py)      │ ← Top-level API
│    Admission, scheduling, preemption         │
├─────────────────────────────────────────────┤
│  BlockAllocator                              │ ← Physical block allocation
│    Simple free list (Python list)             │
├─────────────────────────────────────────────┤
│  KVBlockPool                                 │ ← Physical GPU memory pool
│    Pre-allocated tensors per (layer, head)    │
├─────────────────────────────────────────────┤
│  BlockCache                                  │ ← Prefix caching
│    Hash → CachedBlock, LRU eviction          │
└─────────────────────────────────────────────┘
```

---

## Component-by-Component Comparison

### 1. Block Metadata

**vLLM: `KVCacheBlock`** ([kv_cache_utils.py:L116-L162](../../vllm/vllm/v1/core/kv_cache_utils.py))

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int # the actual KV tensors live in pre-allocated GPU memory indexed by this block ID
    ref_cnt: int = 0 # not implemented in NanoGPT
    _block_hash: BlockHashWithGroupId | None = None # not implemented in NanoGPT

    # Doubly-linked list pointers for O(1) free queue operations
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    is_null: bool = False
```

**NanoGPT: `CachedBlock`** ([nanogpt-paged-attention.py:L94-L101](../../nanogpt-paged-attention.py))

```python
@dataclass
class CachedBlock:
    block_hash: bytes
    token_ids: tuple
    kv_data: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]
    last_access_step: int = 0
```

| Feature | vLLM | NanoGPT |
|---------|------|---------|
| Reference counting | `ref_cnt` field, incremented/decremented by `touch()` / `free_blocks()` | Not implemented — blocks are simply in cache or not |
| Free list pointers | `prev_free_block` / `next_free_block` — intrusive doubly-linked list | No linked list — free blocks are a Python `list` |
| Block hash | Stored as `BlockHashWithGroupId` (bytes), set only when block is full | `block_hash` is `bytes` from MD5, set at insertion time |
| KV data location | Block ID is an index into pre-allocated GPU tensors (no KV on block object) | `kv_data` dict stored directly on the block object |
| Null block | `is_null` flag for placeholder blocks | Not implemented |
| Memory layout | `@dataclass(slots=True)` for memory efficiency | Standard `@dataclass` |

**Key Insight:** vLLM separates block *metadata* from block *data*. The `KVCacheBlock` is a lightweight metadata object (~64 bytes) — the actual KV tensors live in pre-allocated GPU memory indexed by `block_id`. NanoGPT's `CachedBlock` stores KV tensor references directly on the object, which is simpler but means the block metadata is tied to the KV data lifetime.

---

### 2. Free Block Queue

**vLLM: `FreeKVCacheBlockQueue`** ([kv_cache_utils.py:L165-L394](../../vllm/vllm/v1/core/kv_cache_utils.py))

A custom doubly-linked list with fake head/tail sentinels:

```python
class FreeKVCacheBlockQueue:
    def __init__(self, blocks):
        # Initialize doubly-linked list from blocks
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        # ... wire up all blocks between head and tail

    def popleft(self) -> KVCacheBlock: ...      # O(1) allocation
    def popleft_n(self, n) -> list: ...          # O(n) batch allocation
    def remove(self, block) -> None: ...         # O(1) mid-list removal
    def append(self, block) -> None: ...         # O(1) free (LRU end)
    def prepend_n(self, blocks) -> None: ...     # O(n) priority free
    def append_n(self, blocks) -> None: ...      # O(n) batch free
```

**NanoGPT: `BlockAllocator`** ([nanogpt-paged-attention.py:L336-L359](../../nanogpt-paged-attention.py))

A simple Python list used as a stack:

```python
class BlockAllocator:
    def __init__(self, num_blocks, block_size=4):
        self.free_blocks = list(range(num_blocks))

    def allocate_one(self):
        return self.free_blocks.pop()         # O(1)

    def allocate_n(self, n):
        return [self.free_blocks.pop() for _ in range(n)]  # O(n)

    def free_blocks_for_request(self, block_table):
        self.free_blocks.extend(block_table)  # O(k)
```

| Feature | vLLM | NanoGPT |
|---------|------|---------|
| Data structure | Intrusive doubly-linked list with sentinel nodes | Python `list` (stack) |
| Allocation | `popleft()` — O(1), takes from front (oldest) | `pop()` — O(1), takes from back (LIFO) |
| Free | `append()` — adds to back (most recent) | `extend()` — adds to back |
| Mid-list removal | `remove()` — O(1) via linked list pointers | Not supported |
| Eviction order | LRU by position in list; tail blocks evicted last | No eviction order — all free blocks are equal |
| Batch operations | `popleft_n()`, `append_n()`, `prepend_n()` | `allocate_n()` via loop |
| GC pressure | Zero — manipulates existing object pointers, no new objects | Low — `pop()` and `extend()` on a list |

**Key Insight:** vLLM's `FreeKVCacheBlockQueue` is a hand-written doubly-linked list specifically to support **O(1) mid-list removal**. When a cached block gets a cache hit (`touch()`), it needs to be removed from the free queue without scanning. Python's `deque` doesn't support O(1) arbitrary removal, so vLLM implements its own. NanoGPT doesn't need this because it doesn't put cached blocks back in the free list — they're either allocated or free, with no in-between state.

---

### 3. Block Pool (Physical Memory)

**vLLM: `BlockPool`** ([block_pool.py:L130-L529](../../vllm/vllm/v1/core/block_pool.py))

```python
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size, ...):
        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.null_block = self.free_block_queue.popleft()  # reserved

    def get_new_blocks(self, num_blocks) -> list[KVCacheBlock]: ...
    def touch(self, blocks) -> None: ...
    def free_blocks(self, ordered_blocks, prepend=False) -> None: ...
    def cache_full_blocks(self, request, blocks, ...) -> None: ...
    def evict_blocks(self, block_ids) -> None: ...
    def reset_prefix_cache(self) -> bool: ...
```

**NanoGPT: `KVBlockPool`** ([nanogpt-paged-attention.py:L132-L163](../../nanogpt-paged-attention.py))

```python
class KVBlockPool:
    def __init__(self, num_blocks, block_size, n_layer, n_head, head_size, device):
        self.k_pool = {}
        self.v_pool = {}
        for layer in range(n_layer):
            for head in range(n_head):
                self.k_pool[(layer, head)] = torch.zeros(
                    num_blocks, block_size, head_size, device=device
                )
                self.v_pool[(layer, head)] = torch.zeros(
                    num_blocks, block_size, head_size, device=device
                )
```

| Feature | vLLM | NanoGPT |
|---------|------|---------|
| Purpose | Block metadata pool + free queue + cache hash map | Raw GPU tensor storage only |
| Allocation logic | Built into `BlockPool.get_new_blocks()` | Separate `BlockAllocator` class |
| Caching logic | Built into `BlockPool.cache_full_blocks()`, `touch()` | Separate `BlockCache` class |
| Null block | Reserved block 0 as placeholder | Not implemented |
| Ref counting | `touch()` increments, `free_blocks()` decrements | Not implemented |
| KV event tracking | Optional `KVCacheEvent` queue for distributed KV | Not implemented |
| Metrics | Optional `KVCacheMetricsCollector` | Not implemented |

**Key Insight:** vLLM's `BlockPool` is a **unified abstraction** that combines allocation, caching, eviction, and ref counting into one class. NanoGPT splits these into separate, simpler classes (`KVBlockPool` for storage, `BlockAllocator` for allocation, `BlockCache` for caching). vLLM's design is more complex but avoids cross-object coordination bugs.

---

### 4. Block Hashing

**vLLM: `hash_block_tokens()`** ([kv_cache_utils.py:L563-L590](../../vllm/vllm/v1/core/kv_cache_utils.py))

```python
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
```

**NanoGPT: `hash_block_tokens()`** ([nanogpt-paged-attention.py:L89-L92](../../nanogpt-paged-attention.py))

```python
def hash_block_tokens(parent_hash, token_ids):
    data = (parent_hash, tuple(token_ids))
    return hashlib.md5(str(data).encode()).digest()
```

| Feature | vLLM | NanoGPT |
|---------|------|---------|
| Hash function | Configurable (`sha256_cbor`, `xxhash_cbor`, or custom) | Hardcoded `hashlib.md5` |
| Parent chaining | `parent_block_hash` parameter, `NONE_HASH` sentinel | Same pattern, `NONE_HASH = b'\x00' * 16` |
| Extra keys | Supports LoRA, multimodal, cache salt, prompt embeds | Not supported |
| `NONE_HASH` | Random or seed-derived, 32 bytes | Fixed `b'\x00' * 16` |
| Block hash type | `NewType("BlockHash", bytes)` — type-safe | Raw `bytes` |
| Group ID | `BlockHashWithGroupId` packs hash + group ID | Not applicable (single group) |

**Key Insight:** The chained hashing logic is identical in both: `hash(parent_hash, token_ids)`. vLLM adds extensive machinery for multimodal inputs, LoRA adapters, and cache salts (for security isolation), but the core algorithm is the same. vLLM also uses a configurable hash function (defaulting to sha256 with CBOR serialization) while NanoGPT uses MD5 with string serialization — both work for correctness but vLLM's approach is faster and collision-resistant.

---

### 5. Prefix Cache Lookup

**vLLM: `BlockPool.get_cached_block()`** ([block_pool.py:L184-L209](../../vllm/vllm/v1/core/block_pool.py))

```python
def get_cached_block(self, block_hash, kv_cache_group_ids):
    cached_blocks = []
    for group_id in kv_cache_group_ids:
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, group_id
        )
        block = self.cached_block_hash_to_block.get_one_block(
            block_hash_with_group_id
        )
        if not block:
            return None
        cached_blocks.append(block)
    return cached_blocks
```

**NanoGPT: `find_cached_prefix()`** ([nanogpt-paged-attention.py:L165-L188](../../nanogpt-paged-attention.py))

```python
def find_cached_prefix(block_cache, prompt_tokens, block_size):
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
```

| Feature | vLLM | NanoGPT |
|---------|------|---------|
| Lookup | Single hash lookup per block (hash pre-computed on Request) | Re-computes hash chain per lookup call |
| Hash computation | Done eagerly when tokens arrive (`get_request_block_hasher`) | Done lazily at lookup time |
| Multi-group | Checks all KV cache groups for each block | Single group only |
| Return value | List of `KVCacheBlock` objects (metadata pointers) | Number of cached tokens (integer) |

**Key Insight:** vLLM pre-computes block hashes eagerly when tokens are appended to a request (via `get_request_block_hasher`). The hashes are stored on the `Request` object and reused for both cache lookup and cache insertion. NanoGPT recomputes the hash chain from scratch on every `find_cached_prefix()` call, which is simpler but redundant.

---

### 6. Reference Counting & Eviction

**vLLM: `BlockPool.touch()` / `free_blocks()`**

```python
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)  # O(1) mid-list removal
        block.ref_cnt += 1

def free_blocks(self, ordered_blocks, prepend=False):
    for block in blocks_list:
        block.ref_cnt -= 1
    freed_blocks = [b for b in blocks_list if b.ref_cnt == 0 and not b.is_null]
    if prepend:
        self.free_block_queue.prepend_n(freed_blocks)
    else:
        self.free_block_queue.append_n(freed_blocks)
```

**NanoGPT: `BlockCache._evict_lru()`**

```python
def _evict_lru(self):
    oldest = min(self.cache.values(), key=lambda b: b.last_access_step)
    del self.cache[oldest.block_hash]
```

| Feature | vLLM | NanoGPT |
|---------|------|---------|
| Ref counting | Explicit `ref_cnt` — block is free only when count reaches 0 | Not implemented — blocks are either cached or free |
| Eviction trigger | When allocating from free queue, cached blocks at front are evicted | When cache exceeds `max_blocks`, LRU block is evicted |
| Eviction selection | Implicit LRU via queue ordering — front of free list evicted first | Explicit `min()` scan over all cached blocks — O(n) |
| Shared blocks | Multiple requests can reference the same block (`ref_cnt > 1`) | Not supported — each request has its own block table |
| Prepend option | `free_blocks(prepend=True)` puts preempted blocks at front for fast re-eviction | Not implemented |

**Key Insight:** vLLM's ref counting is critical for production. When two requests share a cached prefix, both increment the block's `ref_cnt`. The block can't be evicted until both requests finish. NanoGPT doesn't share physical blocks between requests — each request gets its own copy of cached KV data via `load_cached_blocks_to_pool()`. This wastes memory but avoids the complexity of shared ownership.

---

### 7. Block Table (Logical → Physical Mapping)

**vLLM:** The block table is managed by `SingleTypeKVCacheManager`, which maintains a per-request list of `KVCacheBlock` objects. The model kernel reads `block_table[i].block_id` to find the physical slot.

**NanoGPT: `Request.block_table`** ([nanogpt-paged-attention.py:L237](../../nanogpt-paged-attention.py))

```python
block_table: List[int] = field(default_factory=list)  # list of physical block IDs
```

Both are semantically identical: a per-request ordered list mapping logical block index → physical block ID. vLLM wraps each entry in a `KVCacheBlock` object for ref counting; NanoGPT uses raw integers.

---

### 8. Writing KV Data to Physical Blocks

**vLLM:** KV data is written by GPU kernels (e.g., `reshape_and_cache_flash` in `csrc/`) that directly index into pre-allocated GPU tensors using the block table. The Python layer never touches KV tensor data.

**NanoGPT: `write_kv_to_pool()`** ([nanogpt-paged-attention.py:L264-L287](../../nanogpt-paged-attention.py))

```python
def write_kv_to_pool(pool, block_table, block_size, start_pos, k_new, v_new, layer, head):
    T_new = k_new.shape[1]
    for t in range(T_new):
        logical_pos = start_pos + t
        block_idx = logical_pos // block_size
        slot_idx = logical_pos % block_size
        phys_block = block_table[block_idx]
        pool.k_pool[(layer, head)][phys_block, slot_idx, :] = k_new[0, t, :]
        pool.v_pool[(layer, head)][phys_block, slot_idx, :] = v_new[0, t, :]
```

| Feature | vLLM | NanoGPT |
|---------|------|---------|
| KV write | CUDA kernel (`reshape_and_cache_flash`) | Python loop with tensor indexing |
| Indexing | Physical block table passed to kernel | `block_table[logical // block_size]` |
| Performance | GPU-accelerated, fused with attention | CPU-bound Python loop (educational) |
| Batch support | Handles entire batch in one kernel call | One request at a time |

---

## Summary: What NanoGPT Captures vs What Production Requires

| Concept | NanoGPT | vLLM | Notes |
|---------|---------|------|-------|
| Block table (logical → physical) | ✅ | ✅ | Identical concept |
| Pre-allocated GPU memory pool | ✅ | ✅ | Same tensor layout |
| Block allocator / free list | ✅ (Python list) | ✅ (doubly-linked list) | vLLM: O(1) removal |
| Chained block hashing | ✅ (MD5) | ✅ (configurable) | Same algorithm |
| Prefix cache lookup | ✅ | ✅ | vLLM pre-computes hashes |
| LRU eviction | ✅ (O(n) scan) | ✅ (O(1) from queue front) | Same policy, different perf |
| Reference counting | ❌ | ✅ | Critical for shared blocks |
| Block sharing across requests | ❌ (copies KV data) | ✅ (shared via ref_cnt) | vLLM's memory advantage |
| O(1) mid-list removal | ❌ | ✅ | Needed for `touch()` on cache hit |
| Multi-group KV cache | ❌ | ✅ | For hybrid attention (SWA, MLA) |
| LoRA / multimodal hash keys | ❌ | ✅ | Production requirement |
| Null block placeholder | ❌ | ✅ | For sparse attention patterns |
| KV cache events (distributed) | ❌ | ✅ | For P/D separation, offloading |
| GPU kernel integration | ❌ (Python indexing) | ✅ (CUDA kernels) | Performance-critical |
| Metrics / observability | ❌ | ✅ | Production monitoring |

## The Three Biggest Differences

### 1. Block Sharing via Reference Counting

This is the most fundamental difference. In vLLM, when two requests share the same prefix, they reference the **same physical block** — `ref_cnt` goes to 2. The block's KV data exists once in GPU memory.

In NanoGPT, `load_cached_blocks_to_pool()` **copies** the cached KV data into the request's own physical blocks. Two requests with the same prefix use 2× the memory. This is why NanoGPT's `BlockCache` stores KV data on the `CachedBlock` object (it needs a source to copy from), while vLLM's `KVCacheBlock` doesn't store KV data at all (the physical block IS the data).

### 2. Free List as a Linked List

vLLM's `FreeKVCacheBlockQueue` is a doubly-linked list specifically because of `touch()`. When a cached block in the free queue gets a cache hit, it needs to be removed from the free queue in O(1) — without scanning. Python's `list.remove()` is O(n), and `deque` doesn't support arbitrary removal at all. The intrusive linked list (pointers stored on the block object itself) solves this with zero allocation.

NanoGPT doesn't need this because its cached blocks and free blocks are in completely separate data structures (`BlockCache.cache` dict vs `BlockAllocator.free_blocks` list).

### 3. Eager vs Lazy Hash Computation

vLLM computes block hashes eagerly when tokens are appended, via `get_request_block_hasher()`. The hashes are stored on the `Request` object and reused for both lookup and insertion. This avoids recomputing the hash chain on every cache check.

NanoGPT recomputes the entire hash chain from block 0 every time `find_cached_prefix()` is called. For short prompts this doesn't matter, but for long prompts with many blocks, the redundant hashing would be measurable.
