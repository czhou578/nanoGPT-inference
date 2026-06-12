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

### 9. Scheduler

**vLLM: `Scheduler`** ([scheduler.py:L67-L954](../../vllm/vllm/v1/core/sched/scheduler.py))

The scheduler is the top-level loop that decides which requests get GPU time at each step. vLLM's scheduler is ~2,300 lines and handles dozens of production concerns (LoRA, multimodal, speculative decoding, P/D disaggregation, structured output, async scheduling). NanoGPT's scheduler is ~90 lines and captures the essential scheduling algorithm.

#### 9a. Request State Machine

**vLLM: `RequestStatus`** ([request.py:L299-L341](../../vllm/vllm/v1/request.py))

```python
class RequestStatus(enum.IntEnum):
    WAITING = auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = auto()
    WAITING_FOR_REMOTE_KVS = auto()
    WAITING_FOR_STREAMING_REQ = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()
    FINISHED_ABORTED = auto()
    FINISHED_IGNORED = auto()
    FINISHED_ERROR = auto()
    FINISHED_REPETITION = auto()
```

**NanoGPT: `Request.status`** ([nanogpt-paged-attention.py:L233](../../nanogpt-paged-attention.py))

```python
status: str = "waiting"  # "waiting" -> "prefilling" -> "active" -> "done"
```

| Feature | vLLM | NanoGPT |
|---------|------|---------| 
| Status type | `IntEnum` with 12 states | String with 4 states |
| Waiting sub-states | 4 variants (grammar init, remote KV, streaming, normal) | 1 (`"waiting"`) |
| Finished sub-states | 6 variants (stopped, length-capped, aborted, ignored, error, repetition) | 1 (`"done"`) |
| Prefill vs decode distinction | No explicit phase — uses `num_computed_tokens < num_tokens` | Explicit `"prefilling"` vs `"active"` states |
| Preemption tracking | Dedicated `PREEMPTED` status + `num_preemptions` counter | Resets to `"waiting"` directly |

**Key Insight:** vLLM eliminates the concept of "prefilling" and "decoding" as separate phases. Instead, every request simply has a `num_computed_tokens` counter, and the scheduler assigns however many new tokens it can. This unified model naturally handles chunked prefill (where a "prefilling" request gets partial tokens across multiple steps), resumed requests after preemption, and speculative decoding (where extra tokens need computing). NanoGPT uses explicit `"prefilling"` → `"active"` transitions, which is easier to reason about but doesn't generalize as cleanly.

---

#### 9b. Scheduling Loop Structure

**vLLM: `Scheduler.schedule()`** ([scheduler.py:L348-L954](../../vllm/vllm/v1/core/sched/scheduler.py))

```python
def schedule(self) -> SchedulerOutput:
    # Phase 1: Schedule RUNNING requests first
    while req_index < len(self.running) and token_budget > 0:
        num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)
        new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
        if new_blocks is None:
            # Preempt lowest-priority running request and retry
            self._preempt_request(preempted_req)
        ...

    # Phase 2: Schedule WAITING requests with remaining budget
    while self.waiting and token_budget > 0:
        num_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)
        num_new_tokens = request.num_tokens - num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)
        new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
        ...
```

**NanoGPT: `Scheduler.schedule()`** ([nanogpt-paged-attention.py:L450-L465](../../nanogpt-paged-attention.py))

```python
def schedule(self, step: int):
    self._maybe_admit(step)     # promote waiting → prefilling if memory allows
    self._maybe_preempt()       # evict if over memory budget

    prefill_req = self.prefilling[0] if self.prefilling else None
    decode_reqs = list(self.active)

    return prefill_req, decode_reqs
```

| Feature | vLLM | NanoGPT |
|---------|------|---------| 
| Scheduling priority | Running requests first, then waiting | Admit first, then preempt |
| Token budget enforcement | Explicit `token_budget` countdown in both phases | Token budget checked only at admission |
| Chunked prefill | Native — request gets `min(remaining_tokens, token_budget)` per step | External — the caller (`interleaved_generate`) computes the chunk size |
| Multiple prefill requests per step | Yes — multiple waiting requests can be admitted in one step | No — at most 1 prefilling request at a time |
| Fused prefill + decode | Implicit — running and waiting requests share the same token budget | Explicit — `assemble_fused_batch()` builds a combined batch |
| Output structure | `SchedulerOutput` dataclass with block IDs, token counts, encoder inputs | `(prefill_req, decode_reqs)` tuple |
| Admission + allocation | Combined — `allocate_slots()` is called during scheduling | Separate — `_maybe_admit()` allocates blocks, `maybe_allocate_block()` happens later |

**Key Insight:** vLLM's two-phase design (running first, waiting second) ensures that **existing requests always make progress** before new ones are admitted. This prevents starvation. NanoGPT's simpler approach calls `_maybe_admit()` before checking running requests, which works in the educational context but doesn't guarantee that running requests are prioritized.

---

#### 9c. Admission Control

**vLLM** (waiting loop in `schedule()`, [scheduler.py:L563-L855](../../vllm/vllm/v1/core/sched/scheduler.py)):

```python
# Waiting scheduling checks:
if len(self.running) == self.max_num_running_reqs:
    break                              # batch size limit

num_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)
num_new_tokens = request.num_tokens - num_computed_tokens
num_new_tokens = min(num_new_tokens, token_budget)

new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
if new_blocks is None:
    break                              # memory limit
```

**NanoGPT: `Scheduler._maybe_admit()`** ([nanogpt-paged-attention.py:L403-L431](../../nanogpt-paged-attention.py)):

```python
def _maybe_admit(self, step):
    candidate = self.waiting[0]
    blocks_needed = (prompt_len + self.block_size - 1) // self.block_size

    if self.block_allocator.num_free < blocks_needed:
        return                             # memory check

    num_cached = find_cached_prefix(self.block_cache, candidate.prompt_tokens, ...)
    actual_compute = prompt_len - num_cached
    needed_compute = min(actual_compute, self.token_budget)

    if self.current_compute_tokens + needed_compute > self.token_budget:
        return                             # budget check

    candidate.block_table = self.block_allocator.allocate_n(blocks_needed)
```

| Feature | vLLM | NanoGPT |
|---------|------|---------| 
| Memory check | `allocate_slots()` returns `None` if insufficient blocks | `block_allocator.num_free < blocks_needed` |
| Budget check | `min(num_new_tokens, token_budget)` | `current_compute_tokens + needed_compute > token_budget` |
| Prefix cache integration | `get_computed_blocks()` returns already-cached token count | `find_cached_prefix()` walks the hash chain |
| Block allocation timing | Allocated during scheduling (inside `allocate_slots`) | All blocks allocated up front at admission |
| Partial admission | Yes — request gets only what fits in the budget (chunked prefill) | No — entire block allocation happens or nothing |
| Multiple admissions per step | Yes — loop continues until budget exhausted | No — admits at most one request per step |
| LoRA constraint | Checks `max_loras` limit before admitting | Not applicable |
| Encoder budget | Separate `encoder_compute_budget` for multimodal inputs | Not applicable |

**Key Insight:** vLLM allocates blocks **lazily per-step** — a request only gets the blocks it needs for this step's tokens. NanoGPT allocates **all blocks up-front** when the request is admitted. vLLM's approach is more memory-efficient (a long prompt only consumes blocks as it's actually prefilled), but NanoGPT's approach avoids mid-generation allocation failures.

---

#### 9d. Preemption

**vLLM: `Scheduler._preempt_request()`** ([scheduler.py:L961-L981](../../vllm/vllm/v1/core/sched/scheduler.py)):

```python
def _preempt_request(self, request):
    self.kv_cache_manager.free(request)
    self.encoder_cache_manager.free(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    request.num_preemptions += 1
    self.waiting.prepend_request(request)
```

Preemption is triggered **inline** during the running scheduling loop when `allocate_slots()` returns `None`:

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
    if new_blocks is not None:
        break
    # Preempt lowest-priority running request
    preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
    self._preempt_request(preempted_req)
```

**NanoGPT: `Scheduler._maybe_preempt()`** ([nanogpt-paged-attention.py:L434-L448](../../nanogpt-paged-attention.py)):

```python
def _maybe_preempt(self):
    kv_used = sum(len(req.prompt_tokens) + req.num_generated 
                  for req in self.active + self.prefilling)

    while self.active and kv_used > self.max_kv_tokens:
        victim = max(self.active, key=lambda r: (r.priority, -r.arrival_time))
        victim.clear_cache(self.block_allocator)
        victim.prefill_cursor = 0
        victim.generated_tokens = []
        heapq.heappush(self.waiting, (*key, victim.id, victim))
        ...
```

| Feature | vLLM | NanoGPT |
|---------|------|---------| 
| Trigger | Memory allocation failure (`allocate_slots()` returns `None`) | KV token count exceeds `max_kv_tokens` threshold |
| Timing | During running request scheduling (demand-driven) | After admission, before scheduling (proactive) |
| Victim selection | Lowest priority, then latest arrival (FCFS) or configurable | Lowest priority, then earliest arrival |
| State reset | `num_computed_tokens = 0`, KV freed | `prefill_cursor = 0`, `generated_tokens = []`, blocks freed |
| Requeue position | `prepend_request()` — front of waiting queue (high priority re-admission) | `heappush()` — sorted by priority/arrival into waiting queue |
| Preemption counter | `num_preemptions` tracked for metrics | Not tracked |
| Preempt-and-retry | Yes — after preempting, retries `allocate_slots()` for the original request | No — preemption and admission are separate phases |
| Cached block preservation | Freed blocks go to end of free queue (may retain cache data) | Blocks fully freed; cached data may persist in `BlockCache` |

**Key Insight:** vLLM's preemption is **demand-driven and surgical** — it only preempts when a specific request can't get blocks, and it retries immediately after freeing. NanoGPT's preemption is **threshold-driven** — it checks a global memory watermark and preempts proactively. vLLM's approach is more efficient (preempt only what's necessary), while NanoGPT's is simpler to reason about.

---

#### 9e. Request Queue

**vLLM: `RequestQueue`** ([request_queue.py](../../vllm/vllm/v1/core/sched/request_queue.py))

```python
class FCFSRequestQueue(deque[Request], RequestQueue):
    def add_request(self, request): self.append(request)
    def pop_request(self): return self.popleft()
    def prepend_request(self, request): self.appendleft(request)

class PriorityRequestQueue(RequestQueue):
    def __init__(self): self._heap: list[Request] = []
    def add_request(self, request): heapq.heappush(self._heap, request)
    def pop_request(self): return heapq.heappop(self._heap)
```

**NanoGPT: waiting queue** ([nanogpt-paged-attention.py:L396-L398](../../nanogpt-paged-attention.py))

```python
def add_request(self, req):
    key = self._sort_key(req)
    heapq.heappush(self.waiting, (*key, req.id, req))
```

| Feature | vLLM | NanoGPT |
|---------|------|---------| 
| Queue abstraction | `RequestQueue` ABC with two implementations | Inline heap with tuple keys |
| FCFS implementation | `deque` — O(1) front/back operations | `heapq` with `(0, arrival_time, id, req)` keys |
| Priority implementation | Min-heap on `Request.__lt__` (priority, arrival_time, request_id) | `heapq` with `(priority, arrival_time, id, req)` keys |
| Skipped requests | Separate `skipped_waiting` queue for blocked requests | Not implemented |
| Request ordering | `Request.__lt__` defines canonical ordering | Manual sort key construction |
| Polymorphism | Factory function `create_request_queue(policy)` | `_sort_key()` method switches on policy string |

---

#### 9f. Output Processing

**vLLM: `Scheduler.update_from_output()`** ([scheduler.py:L1299-L1500](../../vllm/vllm/v1/core/sched/scheduler.py))

vLLM has a dedicated 200-line method that processes model outputs after each forward pass:

```python
def update_from_output(self, scheduler_output, model_runner_output):
    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
        generated_token_ids = sampled_token_ids[req_index]
        
        # Handle spec decoding rejections
        if scheduled_spec_token_ids and generated_token_ids:
            num_rejected = num_draft_tokens - (len(generated_token_ids) - 1)
            request.num_computed_tokens -= num_rejected
        
        # Check stop conditions (EOS, length cap, repetition)
        new_token_ids, stopped = self._update_request_with_output(request, ...)
        
        # Build EngineCoreOutput per request
        outputs[request.client_index].append(EngineCoreOutput(...))
```

**NanoGPT** (inline in `interleaved_generate`, [nanogpt-paged-attention.py:L936-L960](../../nanogpt-paged-attention.py)):

```python
# Decode output processing
logits_decode = logits[:len(decode_reqs), -1, :]
probs = F.softmax(logits_decode, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)

for i, req in enumerate(decode_reqs):
    req.generated_tokens.append(idx_next[i].item())
    if req.is_done:
        scheduler.complete(req)

# Prefill completion
if prefill_req.is_fully_prefilled:
    scheduler.promote(prefill_req)
```

| Feature | vLLM | NanoGPT |
|---------|------|---------| 
| Separation | Dedicated `update_from_output()` method on Scheduler | Inline in the generation loop |
| Stop conditions | EOS, stop tokens, length cap, repetition detection, abort | `num_generated >= max_new_tokens` only |
| Spec decode handling | Adjusts `num_computed_tokens` by number of rejected tokens | Not applicable |
| Logprobs | Extracted and attached to output per request | Not extracted |
| Structured output | Grammar state advanced per token | Not applicable |
| Output format | `EngineCoreOutput` per request, routed by `client_index` | Direct mutation of `Request.generated_tokens` |
| Sampling | Done by model runner (separate process) | Inline `softmax` → `multinomial` |
| KV cache update | Implicit (model runner writes to GPU memory via kernels) | Explicit `disassemble_paged_fused()` scatters KV back to pool |

**Key Insight:** In vLLM, the scheduler **never touches model outputs directly** — sampling happens in the model runner process, and the scheduler only processes the resulting token IDs. This clean separation enables async scheduling (overlapping scheduling step N+1 with model execution step N). In NanoGPT, sampling and scheduling are interleaved in the same loop, which is simpler but couples the two concerns.

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
| Unified prefill/decode scheduling | ❌ (explicit phases) | ✅ (`num_computed_tokens` model) | Enables chunked prefill naturally |
| Running-first scheduling | ❌ (admit first) | ✅ (running first, then waiting) | Prevents starvation |
| Demand-driven preemption | ❌ (threshold-based) | ✅ (allocation failure triggers) | More efficient |
| Lazy block allocation | ❌ (all blocks up-front) | ✅ (per-step allocation) | Better memory utilization |
| Multiple prefill per step | ❌ (one at a time) | ✅ (budget-limited) | Higher throughput |
| Async scheduling | ❌ | ✅ (overlaps with model execution) | Production latency |
| Scheduler/sampler separation | ❌ (inline) | ✅ (separate processes) | Enables async + PP |

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
