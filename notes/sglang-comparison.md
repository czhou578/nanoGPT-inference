# SGLang RadixCache vs NanoGPT Radix Tree — Side-by-Side Comparison

This document compares the radix tree prefix caching in [SGLang](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) with our educational implementation in [nanogpt-radix-tree-.py](../nanogpt-radix-tree-.py).

The goal is to show how the same RadixAttention concepts map between a production system (~800 lines in `radix_cache.py` + ~370 lines in `base_prefix_cache.py`) and a minimal educational one (~170 lines of tree code).

---

## Architecture Overview

### SGLang's Radix Cache Stack

```
┌─────────────────────────────────────────────┐
│  RadixCache (radix_cache.py)                │ ← Top-level API
│    Extends BasePrefixCache + KVCacheEvents  │
├─────────────────────────────────────────────┤
│  TreeNode                                    │ ← Node metadata
│    children, key, value, lock_ref,           │
│    hit_count, priority, host_value           │
├─────────────────────────────────────────────┤
│  RadixKey                                    │ ← Edge label abstraction
│    token_ids, extra_key, bigram mode         │
│    Exponential-search matching               │
├─────────────────────────────────────────────┤
│  token_to_kv_pool_allocator                  │ ← Physical KV memory pool
│    Alloc/free of KV cache indices            │
├─────────────────────────────────────────────┤
│  evictable_leaves (set)                      │ ← Leaf-first LRU eviction
│    Priority-aware heap eviction              │
└─────────────────────────────────────────────┘
```

### NanoGPT's Radix Tree Stack

```
┌─────────────────────────────────────────────┐
│  RadixTree (nanogpt-radix-tree-.py)         │ ← Top-level API
│    match_prefix, insert, eviction            │
├─────────────────────────────────────────────┤
│  RadixNode                                   │ ← Node metadata
│    children, token_ids, kv_data, lock_ref    │
├─────────────────────────────────────────────┤
│  Per-request KV cache (dict)                 │ ← KV data on request
│    kv_cache[(layer, head)] = (k, v)          │
└─────────────────────────────────────────────┘
```

---

## Component-by-Component Comparison

### 1. Node Metadata

**SGLang: `TreeNode`** ([radix_cache.py:L206-L267](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py))

```python
class TreeNode:
    counter = 0

    def __init__(self, id=None, priority=0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None   # KV cache indices (int64)
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()
        self.hit_count = 0
        self.host_ref_counter = 0                    # CPU offload lock
        self.host_value: Optional[torch.Tensor] = None
        self.hash_value: Optional[List[str]] = None  # per-page SHA256
        self.priority = priority                     # priority-aware eviction
        self.id = TreeNode.counter
```

**NanoGPT: `RadixNode`** ([nanogpt-radix-tree-.py:L119-L126](../nanogpt-radix-tree-.py))

```python
class RadixNode:
    def __init__(self):
        self.children: Dict[int, RadixNode] = {}
        self.parent: Optional[RadixNode] = None
        self.token_ids: Tuple[int, ...] = ()
        self.kv_data: Optional[Dict] = None   # actual KV tensors stored here
        self.lock_ref: int = 0
        self.last_access_time: int = 0
```

| Feature | SGLang | NanoGPT |
|---------|--------|---------| 
| Edge label | `RadixKey` object (supports bigram, extra_key, page alignment) | Raw `tuple[int, ...]` |
| KV data storage | `value` = `torch.Tensor` of **indices** into a shared GPU memory pool | `kv_data` = dict of actual KV **tensors** stored on the node |
| Children dict | `defaultdict(TreeNode)` — auto-creates on access | `Dict[int, RadixNode]` — standard dict |
| Access tracking | `time.monotonic()` — real wall-clock time | Integer step counter |
| Hit counting | `hit_count` — tracks how often a node is matched | Not implemented |
| Priority | Per-node `priority` for priority-aware eviction | Not implemented |
| Host offload | `host_value` + `host_ref_counter` for CPU ↔ GPU tiering | Not implemented |
| Per-page hashing | `hash_value: List[str]` — SHA256 per page, computed lazily | Not implemented |
| Node ID | Global auto-incrementing `counter` | Not implemented |

**Key Insight:** SGLang's `TreeNode` stores KV cache **indices** (pointers into a shared GPU memory pool), not the actual tensors. This means multiple requests referencing the same tree node share the same physical GPU memory — zero copies. NanoGPT stores the actual KV tensors on each node, and `load_from_radix_tree()` **clones** them onto each request's private cache. This is simpler but wastes GPU memory when multiple requests share a prefix.

---

### 2. Edge Labels (RadixKey)

**SGLang: `RadixKey`** ([radix_cache.py:L60-L203](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py))

```python
class RadixKey:
    __slots__ = ("token_ids", "extra_key", "is_bigram")

    def __init__(self, token_ids: array[int], extra_key=None, is_bigram=False):
        self.token_ids = token_ids     # array('q', [...]) — compact C array
        self.extra_key = extra_key     # LoRA ID, cache salt, etc.
        self.is_bigram = is_bigram     # EAGLE speculative decoding mode

    def match(self, other, page_size=1) -> int:
        # Exponential search + binary search for divergence point
        # O(log(prefix_len)) C-level slice comparisons
        ...

    def child_key(self, page_size=1):
        # Hashable dict key for child lookup, namespaced by extra_key
        ...

    def page_aligned(self, page_size) -> RadixKey:
        # Truncate to page boundary
        ...
```

**NanoGPT:** No dedicated edge label type — uses raw `tuple[int, ...]` for `token_ids` and `int` for child dict keys.

| Feature | SGLang | NanoGPT |
|---------|--------|---------| 
| Data type | `array('q', ...)` — C-level int64 array | Python `tuple[int, ...]` |
| Matching algorithm | Exponential search + binary search — O(log n) slice compares | Linear scan — O(n) per-token Python loop |
| Page alignment | `page_aligned(page_size)` truncates to page boundary | Manual `(matched // block_size) * block_size` |
| Namespace isolation | `extra_key` separates LoRA adapters, cache salts | Not supported |
| Bigram mode | `is_bigram=True` for EAGLE speculative decoding | Not supported |
| Child key | `child_key(page_size)` — supports multi-token page keys | `token_ids[0]` — single token |
| Memory | `__slots__` — minimal per-object overhead | Standard object |

**Key Insight:** SGLang's `RadixKey.match()` uses an **exponential search** algorithm for prefix matching — it gallops in doubling windows using C-level `array` slice comparisons, then binary searches the divergence window. This avoids per-token Python loops entirely. NanoGPT uses a straightforward `while` loop comparing one token at a time, which is clear but O(n) in Python.

---

### 3. Prefix Matching

**SGLang: `RadixCache._match_prefix_helper()`** ([radix_cache.py:L631-L655](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py))

```python
def _match_prefix_helper(self, node, key):
    access_time = time.monotonic()
    node.last_access_time = access_time
    child_key = key.child_key(self.page_size)

    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        child.last_access_time = access_time
        prefix_len = child.key.match(key, page_size=self.page_size)
        if prefix_len < len(child.key):
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)
    return value, node
```

**NanoGPT: `RadixTree.match_prefix()`** ([nanogpt-radix-tree-.py:L218-L252](../nanogpt-radix-tree-.py))

```python
def match_prefix(self, token_ids):
    node = self.root
    matched = 0
    while matched < len(token_ids):
        next_token = token_ids[matched]
        child = node.children.get(next_token)
        if child is None: break

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

        matched += edge_match_len
        node = child
    return node, matched
```

| Feature | SGLang | NanoGPT |
|---------|--------|---------| 
| Return value | `(List[torch.Tensor], TreeNode)` — concatenated KV indices + last node | `(RadixNode, int)` — last node + match count |
| Edge matching | `RadixKey.match()` — exponential search | Per-token `while` loop |
| Access time update | `time.monotonic()` on every traversed node | Not updated during match (only `lock_ref`) |
| Split on partial match | Yes — identical logic | Yes — identical logic |
| Page alignment | Input key pre-aligned to `page_size` | Manual alignment after the call |
| Empty key handling | Returns pre-allocated empty `MatchResult` | Returns `(root, 0)` |

**Key Insight:** The core algorithm is identical: walk the tree, compare edges, split on partial match. The differences are in the return type (SGLang returns KV indices directly usable by the GPU; NanoGPT returns a node that must be walked to extract KV data) and in the matching performance (C-level exponential search vs Python loop).

---

### 4. Node Splitting

**SGLang: `RadixCache._split_node()`** ([radix_cache.py:L657-L677](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py))

```python
def _split_node(self, key, child, split_len):
    new_node = TreeNode(priority=child.priority)
    new_node.hit_count = child.hit_count
    new_node.children = {key[split_len:].child_key(self.page_size): child}
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len].clone()
    child.parent = new_node
    child.key = child.key[split_len:]
    child.value = child.value[split_len:].clone()
    new_node.parent.children[key.child_key(self.page_size)] = new_node
    # Split hash_value too
    new_node.hash_value, child.hash_value = split_node_hash_value(...)
    return new_node
```

**NanoGPT: `RadixTree._split_node()`** ([nanogpt-radix-tree-.py:L150-L182](../nanogpt-radix-tree-.py))

```python
def _split_node(self, child, split_len):
    new_mid = RadixNode()
    new_mid.token_ids = child.token_ids[:split_len]
    new_mid.parent = child.parent
    new_mid.last_access_time = child.last_access_time
    new_mid.lock_ref = child.lock_ref

    if child.kv_data is not None:
        new_mid.kv_data = {}
        new_child_kv = {}
        for (layer, head), (k, v) in child.kv_data.items():
            new_mid.kv_data[(layer, head)] = (
                k[:, :split_len, :].clone(), v[:, :split_len, :].clone()
            )
            new_child_kv[(layer, head)] = (
                k[:, split_len:, :].clone(), v[:, split_len:, :].clone()
            )
        child.kv_data = new_child_kv

    child.token_ids = child.token_ids[split_len:]
    child.parent = new_mid
    new_mid.children[child.token_ids[0]] = child
    new_mid.parent.children[new_mid.token_ids[0]] = new_mid
    return new_mid
```

| Feature | SGLang | NanoGPT |
|---------|--------|---------| 
| Value split | `value[:split_len].clone()` — slicing a 1D index tensor | Per-(layer, head) 3D tensor slicing + clone |
| Hash split | `split_node_hash_value()` preserves per-page hashes | Not applicable |
| Hit count | Inherited by new mid-node | Not tracked |
| Priority | Inherited by new mid-node | Not tracked |
| Cost | O(split_len) — single tensor clone | O(n_layer × n_head × split_len) — many tensor clones |

**Key Insight:** Because SGLang stores KV as a 1D index tensor (`torch.int64`), splitting a node is a single cheap tensor slice. NanoGPT stores actual KV data, so splitting requires cloning every `(layer, head)` pair — O(n_layer × n_head) tensor allocations. This is a direct consequence of the "indices vs data" design choice.

---

### 5. Insertion

**SGLang: `RadixCache._insert_helper()`** ([radix_cache.py:L687-L740](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py))

```python
def _insert_helper(self, node, key, value, priority=0, chunked=False):
    while len(key) > 0 and child_key in node.children.keys():
        node = node.children[child_key]
        prefix_len = node.key.match(key, page_size=self.page_size)
        total_prefix_length += prefix_len
        key = key[prefix_len:]
        value = value[prefix_len:]
        if prefix_len < len(node.key):
            new_node = self._split_node(node.key, node, prefix_len)
            node = new_node
        ...
    if len(key):
        new_node = TreeNode(priority=priority)
        new_node.parent = node
        new_node.key = key
        new_node.value = value.clone()
        node.children[child_key] = new_node
        self.evictable_size_ += len(key)
        self._update_leaf_status(node)
        self._update_leaf_status(new_node)
    return total_prefix_length
```

**NanoGPT: `RadixTree.insert()`** ([nanogpt-radix-tree-.py:L184-L206](../nanogpt-radix-tree-.py))

```python
def insert(self, token_ids, kv_data_full, block_size):
    node, matched = self.match_prefix(token_ids)
    if matched == len(token_ids): return

    remaining = token_ids[matched:]
    new_node = RadixNode()
    new_node.token_ids = tuple(remaining)
    new_node.parent = node

    new_node.kv_data = {}
    for (layer, head), (k, v) in kv_data_full.items():
        new_node.kv_data[(layer, head)] = (
            k[:, matched:matched + len(remaining)].clone(),
            v[:, matched:matched + len(remaining)].clone(),
        )
    node.children[remaining[0]] = new_node
```

| Feature | SGLang | NanoGPT |
|---------|--------|---------| 
| Approach | Iterative walk + inline split | `match_prefix()` first, then append remainder |
| Duplicate handling | Walks existing prefix, returns `total_prefix_length` for dedup freeing | `if matched == len(token_ids): return` — skip |
| Evictable size tracking | `self.evictable_size_ += len(key)` incremented inline | Not tracked |
| Leaf status | `_update_leaf_status()` maintains `evictable_leaves` set | Not tracked |
| Hit count | `_inc_hit_count()` on every traversed node | Not tracked |
| Priority propagation | `node.priority = max(node.priority, priority)` along path | Not applicable |
| Chunked insert | `chunked=True` skips hit count (avoids self-inflation) | Not applicable |

**Key Insight:** SGLang's insert is iterative and handles the common case where a prefix already exists — it walks the existing path, deduplicates indices, and only creates a new node for the truly new suffix. NanoGPT's two-phase approach (match then append) is cleaner but can't deduplicate mid-insertion or update metadata along the existing path.

---

### 6. Eviction

**SGLang: `RadixCache.evict()`** ([radix_cache.py:L545-L572](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py))

```python
def evict(self, params: EvictParams):
    leaves = list(self.evictable_leaves)
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        self._delete_leaf(x)

        # Parent may become a new leaf
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            heapq.heappush(eviction_heap, (new_priority, x.parent))
```

**NanoGPT:** No explicit eviction in the radix tree. The scheduler uses a global `max_kv_tokens` threshold and preempts entire requests when memory pressure is high.

| Feature | SGLang | NanoGPT |
|---------|--------|---------| 
| Eviction granularity | Per-node (leaf-first) | Per-request (whole request preempted) |
| Eviction trigger | `evict(num_tokens)` called when allocator runs out | `_maybe_preempt()` when KV tokens > threshold |
| Eviction strategy | Pluggable (`lru`, `priority`, custom) via `eviction_strategy` | Not applicable at tree level |
| Leaf tracking | `evictable_leaves: set` — maintained incrementally | Not tracked |
| Parent cascading | After deleting leaf, parent may become new evictable leaf | Not applicable |
| Memory freed | KV indices returned to `token_to_kv_pool_allocator` | Entire request's KV cache cleared |
| Locked nodes | `lock_ref > 0` prevents eviction | `lock_ref > 0` prevents eviction (but eviction is request-level) |

**Key Insight:** SGLang does **fine-grained, leaf-first eviction** — it can evict a single unused suffix without affecting other branches that share the same prefix. NanoGPT doesn't evict from the tree at all; instead, the scheduler preempts entire requests and clears their KV caches. The tree only grows, with nodes remaining until the process ends. This means NanoGPT's tree acts purely as a read-through cache, not a managed memory structure.

---

### 7. Lock Reference Counting

**SGLang: `inc_lock_ref()` / `dec_lock_ref()`** ([radix_cache.py:L574-L609](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py))

```python
def inc_lock_ref(self, node):
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
        node.lock_ref += 1
        self._update_leaf_status(node)
        node = node.parent
    return IncLockRefResult(delta=delta)
```

**NanoGPT: `load_from_radix_tree()` / `unlock_radix_path()`** ([nanogpt-radix-tree-.py:L82-L117, L209-L216](../nanogpt-radix-tree-.py))

```python
# Lock: increment along the path from leaf to root
for pnode in prefix_path:
    pnode.lock_ref += 1

# Unlock
def unlock_radix_path(self, request):
    for node in request._radix_path:
        node.lock_ref -= 1
```

| Feature | SGLang | NanoGPT |
|---------|--------|---------| 
| Walk direction | Leaf → root (same) | Leaf → root (same) |
| Size accounting | `evictable_size_` / `protected_size_` updated on transitions | Not tracked |
| Leaf status | `_update_leaf_status()` maintains evictable set | Not tracked |
| Path storage | Stored on `req.last_node` — only the leaf, walks up | Stored on `request._radix_path` — entire path list |
| Host locking | `host_ref_counter` for CPU offload protection | Not applicable |

---

### 8. Request Lifecycle Integration

**SGLang: `cache_finished_req()` / `cache_unfinished_req()`**

SGLang integrates the radix tree directly into the request lifecycle via two methods:

- `cache_finished_req()` — inserts the full sequence (prompt + output) into the tree, frees duplicates, decrements locks
- `cache_unfinished_req()` — inserts partial progress (chunked prefill), re-matches to get updated indices, transfers locks

**NanoGPT:** Manual integration in the generation loop:

```python
# On prefill completion:
insert_into_radix_tree(prefill_req, scheduler.radix_tree, scheduler.block_size)
scheduler.radix_tree.unlock_radix_path(prefill_req)

# On request completion:
scheduler.radix_tree.unlock_radix_path(req)
```

| Feature | SGLang | NanoGPT |
|---------|--------|---------| 
| Finished request caching | `cache_finished_req()` — handles dedup, index transfer, lock release | `insert_into_radix_tree()` — manual call in loop |
| Unfinished request caching | `cache_unfinished_req()` — incremental caching during chunked prefill | Not implemented — only inserts on prefill completion |
| Index deduplication | Frees indices that were already in tree: `free(kv_indices[protected:prefix_len])` | Not applicable (copies data, doesn't share indices) |
| Lock transfer | `dec_lock_ref(old_node)` → `inc_lock_ref(new_node)` atomically | Manual lock/unlock at separate points |
| Unaligned tail handling | `free(kv_indices[key_len:])` — frees tokens beyond page boundary | Not applicable |

---

## Summary: What NanoGPT Captures vs What Production Requires

| Concept | NanoGPT | SGLang | Notes |
|---------|---------|--------|-------|
| Radix tree structure | ✅ | ✅ | Identical trie with compressed edges |
| Prefix matching with split | ✅ | ✅ | Same algorithm |
| Node splitting | ✅ | ✅ | Same logic, different cost |
| Lock reference counting | ✅ (simple) | ✅ (with size tracking) | Same walk-to-root pattern |
| KV sharing via indices | ❌ (copies data) | ✅ (shared indices) | SGLang's memory advantage |
| Leaf-first eviction | ❌ | ✅ | Critical for memory management |
| Evictable leaf tracking | ❌ | ✅ (incremental set) | Enables O(1) leaf discovery |
| Priority-aware eviction | ❌ | ✅ (pluggable strategy) | Production scheduling |
| Page-aligned operations | ❌ (manual) | ✅ (`page_size` parameter) | GPU memory alignment |
| Exponential-search matching | ❌ (linear scan) | ✅ | Performance at scale |
| Bigram mode (EAGLE) | ❌ | ✅ | Speculative decoding support |
| LoRA / cache salt isolation | ❌ | ✅ (`extra_key`) | Multi-tenant serving |
| Host ↔ device tiering | ❌ | ✅ (`host_value`) | HiCache CPU offloading |
| Hit count tracking | ❌ | ✅ | Eviction policy input |
| Per-page hashing | ❌ | ✅ (SHA256) | Distributed KV events |
| Chunked prefill caching | ❌ | ✅ (`cache_unfinished_req`) | Incremental caching |
| Metrics / observability | ❌ | ✅ | Production monitoring |

## The Three Biggest Differences

### 1. Indices vs Data

This is the most fundamental difference. SGLang's tree nodes store a 1D `torch.int64` tensor of **indices** pointing into a shared GPU memory pool. When a request matches a prefix, it directly uses those indices — no copying. Multiple requests sharing a prefix consume zero additional GPU memory for the cached portion.

NanoGPT stores the actual KV tensors on each `RadixNode`. When `load_from_radix_tree()` finds a match, it **clones** every `(layer, head)` tensor pair onto the request's private cache. Two requests sharing a 100-token prefix duplicate all the KV data. This also makes node splitting expensive (O(n_layer × n_head) tensor allocations vs a single index slice).

### 2. Fine-Grained Eviction vs No Eviction

SGLang maintains an `evictable_leaves` set and can evict individual tree nodes when memory pressure rises. After evicting a leaf, the parent may cascade into a new evictable leaf. This enables surgical memory reclamation — unused suffixes are freed while shared prefixes remain intact.

NanoGPT has no tree-level eviction at all. The tree only grows. Memory management happens at the request level through scheduler preemption (`_maybe_preempt()`), which is a much coarser mechanism. If a prompt's prefix is cached in the tree but the tree node holds actual tensor data, that memory is never reclaimed until the process exits.

### 3. Exponential-Search Matching vs Linear Scan

SGLang's `RadixKey.match()` uses an exponential search algorithm: it gallops in doubling windows using C-level `array` slice comparisons (`t0[lo:hi] != t1[lo:hi]`), then binary-searches the divergence window. This avoids per-token Python-level iteration entirely, making it efficient for long shared prefixes (common in multi-turn conversations).

NanoGPT matches token-by-token in a Python `while` loop. For an edge with 1,000 matching tokens, that's 1,000 Python iterations. SGLang would handle this in ~10 C-level slice comparisons (exponential galloping) plus ~10 more (binary search) — a ~50× reduction in Python overhead.
