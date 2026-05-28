# Prefix Caching — Implementation Plan & Hints

## The Problem You're Solving

Look at what happens when two requests share the same system prompt:

```
Request A: "You are a helpful assistant. Translate to French: Hello"
Request B: "You are a helpful assistant. Translate to French: Goodbye"

Shared prefix: "You are a helpful assistant. Translate to French: "
```

Without prefix caching, both requests **independently prefill the entire prompt** — including
the shared prefix. Every token in "You are a helpful assistant. Translate to French: " goes
through `K = W_k × x` and `V = W_v × x` for every layer and every head, **twice**. The
resulting KV tensors are numerically identical because the same tokens attended to the same
prior context, but you computed them from scratch both times.

**Prefix caching** stores completed KV blocks in a content-addressed cache. When Request B
arrives and its prompt starts with the same tokens as Request A, the scheduler finds the
cached KV blocks, skips the prefill for those tokens, and only computes the **suffix**
(`"Goodbye"`). This directly reduces TTFT.

In production (vLLM's Automatic Prefix Caching), this cuts prefill compute by 50–90% for
workloads with shared system prompts — which is the vast majority of API deployments.

---

## Why This Matters Even at 210K Params

You won't see a meaningful wall-clock improvement on nanoGPT — the model is too small and the
prompts too short for the cache lookup overhead to pay for itself. But the concepts are
exactly what vLLM implements:

1. **Content-addressed hashing** — KV blocks are keyed by their token content, not by
   request ID or position.
2. **Chained hashes** — each block's hash includes its parent's hash, so the entire prefix
   history is captured transitively.
3. **LRU eviction** — when memory is full, the least-recently-used cached blocks are evicted
   to make room for new ones.
4. **The scheduler integrates cache hits** — cached tokens are subtracted from the work to do,
   so a fully-cached prefix means near-zero prefill cost.

The goal is to learn the architecture, not hit a perf number.

---

## Hint 1: Think in Blocks, Not Individual Tokens

Your current KV cache is per-request and per-(layer, head):

```python
req.kv_cache[(layer_idx, head_idx)] = (k_tensor, v_tensor)
# k_tensor shape: (1, T_i, head_size) — contiguous, grows by 1 each decode step
```

For prefix caching, you need to think in **fixed-size blocks** of tokens. Choose a block size
(e.g. `BLOCK_SIZE = 4` — small enough to see the mechanics at nanoGPT scale). A prompt of
12 tokens becomes 3 blocks:

```
Block 0: tokens[0:4]   → KV for positions 0, 1, 2, 3
Block 1: tokens[4:8]   → KV for positions 4, 5, 6, 7
Block 2: tokens[8:12]  → KV for positions 8, 9, 10, 11
```

Each block stores a fixed-size KV chunk: `(1, BLOCK_SIZE, head_size)` per (layer, head).
Only **full** blocks (exactly `BLOCK_SIZE` tokens) are eligible for caching. The trailing
partial block is never cached — it changes with every new decode token.

**Question to ask yourself:** Why can't you cache partial blocks?

Answer: because the block's content isn't finalized. During decode, new tokens append to the
last block until it fills up. Only then is its token content fixed and its hash meaningful.

---

## Hint 2: Content-Addressed Hashing with Parent Chains

The cache key for a block is **not** just its token IDs. It's a hash of:

```python
block_hash = hash((parent_block_hash, tuple(block_token_ids)))
```

**Why the parent hash?** Because KV values are context-dependent. Consider:

```
Request A: ["The", "cat", "sat", "on"] ["the", "mat", ".", "!"]
Request B: ["The", "dog", "sat", "on"] ["the", "mat", ".", "!"]
```

Block 1 (`["the", "mat", ".", "!"]`) has the **same token IDs** in both requests. But the
KV tensors are numerically different — in Request A, every token in Block 1 attended to
`"The cat sat on"`, while in Request B it attended to `"The dog sat on"`. The K and V
projections produce different values because the input `x` to the attention layer is different
(it was contextualized by a different prefix).

By chaining the parent hash, Block 1's hash in Request A encodes the full history through
Block 0 (`["The", "cat", "sat", "on"]`), which differs from Block 0 in Request B
(`["The", "dog", "sat", "on"]`). The two Block 1 hashes are therefore different, and the
cache correctly treats them as distinct entries.

**The transitive property:** if block `k` matches, it implies all blocks `0..k-1` also match.
A cache hit at any block guarantees the entire prefix up to that block is identical.

```python
import hashlib

NONE_HASH = b'\x00' * 16  # sentinel for the first block (no parent)

def hash_block_tokens(parent_hash, token_ids):
    """Compute a chained content hash for a KV block."""
    data = (parent_hash, tuple(token_ids))
    return hashlib.md5(str(data).encode()).digest()
```

---

## Hint 3: The Global Block Cache

You need a **global** cache that lives outside any individual request — a shared pool that
multiple requests can read from.

```python
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
```

The `BlockCache` is instantiated once and shared across all requests. It's passed to the
scheduler so the admission logic can check for cache hits before deciding how much prefill
work is needed.

---

## Hint 4: Finding Cache Hits During Admission

When a new request arrives, the scheduler needs to figure out how many of its prompt tokens
are already cached. This is done by computing block hashes from the prompt and checking
each one against the `BlockCache`:

```python
def find_cached_prefix(block_cache, prompt_tokens, block_size):
    """
    Walk the prompt left-to-right in block-sized chunks.
    Return the number of tokens that are fully cached.
    """
    num_cached_tokens = 0
    parent_hash = NONE_HASH

    for start in range(0, len(prompt_tokens), block_size):
        end = start + block_size
        if end > len(prompt_tokens):
            break  # partial block — not cacheable

        chunk = prompt_tokens[start:end]
        block_hash = hash_block_tokens(parent_hash, chunk)

        cached = block_cache.lookup(block_hash)
        if cached is None:
            break  # cache miss — everything from here on must be computed

        num_cached_tokens += block_size
        parent_hash = block_hash  # chain for the next block

    return num_cached_tokens
```

**Key insight:** you stop at the first miss. Because hashes are chained, a miss at block `k`
means every subsequent block's hash would also differ (even if the token IDs happen to match).
The prefix property is all-or-nothing up to the hit boundary.

After finding the hit length, the scheduler knows:
```
num_new_tokens = len(prompt_tokens) - num_cached_tokens
```

Only `num_new_tokens` need to be prefilled. The cached blocks' KV data is loaded directly
onto the request.

---

## Hint 5: Loading Cached KV Data onto the Request

When you find cached blocks, you need to **reconstruct the request's KV cache** from the
cached data before running the prefill of the remaining suffix:

```python
def load_cached_blocks(request, block_cache, prompt_tokens, block_size):
    """
    Load cached KV blocks onto a request and return how many tokens were cached.
    Sets request.prefill_cursor to skip past the cached portion.
    """
    parent_hash = NONE_HASH
    num_cached = 0

    for start in range(0, len(prompt_tokens), block_size):
        end = start + block_size
        if end > len(prompt_tokens):
            break

        chunk = prompt_tokens[start:end]
        block_hash = hash_block_tokens(parent_hash, chunk)
        cached = block_cache.lookup(block_hash)

        if cached is None:
            break

        # Copy cached KV data onto the request's per-(layer, head) cache
        for (layer, head), (k, v) in cached.kv_data.items():
            if (layer, head) in request.kv_cache:
                # Append to existing cache (from earlier cached blocks)
                existing_k, existing_v = request.kv_cache[(layer, head)]
                request.kv_cache[(layer, head)] = (
                    torch.cat([existing_k, k.clone()], dim=1),
                    torch.cat([existing_v, v.clone()], dim=1),
                )
            else:
                request.kv_cache[(layer, head)] = (k.clone(), v.clone())

        num_cached += block_size
        parent_hash = block_hash

    request.prefill_cursor = num_cached
    return num_cached
```

After this, the request looks like it already completed prefill for the first `num_cached`
tokens. The scheduler's chunked prefill logic takes over for the remaining tokens — it sees
`prefill_cursor > 0` and `kv_cache` already populated, just like a partially-prefilled
request from a previous chunking step.

**Nothing in Head or the model changes.** The request's cached KV is passed in as `past_kvs`
exactly like before. The model can't tell whether the KV came from a fresh prefill or from
the cache.

---

## Hint 6: Caching Newly Computed Blocks

After prefilling a request (partially or fully), the model returns new KV tensors. You need
to **commit completed blocks** to the `BlockCache` for future requests to reuse.

This happens in the scheduler loop, after the forward pass and cache disassembly:

```python
def commit_completed_blocks(request, block_cache, block_size):
    """
    After a prefill step, check if any new full blocks were completed.
    If so, insert them into the global cache.
    """
    total_tokens = len(request.prompt_tokens) + request.num_generated
    num_full_blocks = request.prefill_cursor // block_size

    # We need to track which blocks have already been committed
    # to avoid re-inserting on every step
    if not hasattr(request, '_committed_blocks'):
        request._committed_blocks = 0

    parent_hash = NONE_HASH
    for block_idx in range(num_full_blocks):
        start = block_idx * block_size
        end = start + block_size
        chunk = request.prompt_tokens[start:end]
        block_hash = hash_block_tokens(parent_hash, chunk)

        if block_idx >= request._committed_blocks:
            # Extract this block's KV slice from the request's cache
            kv_data = {}
            for (layer, head), (k, v) in request.kv_cache.items():
                kv_data[(layer, head)] = (
                    k[:, start:end, :].clone(),
                    v[:, start:end, :].clone(),
                )
            block_cache.insert(block_hash, tuple(chunk), kv_data)

        parent_hash = block_hash

    request._committed_blocks = num_full_blocks
```

**When to call this:** after each prefill chunk completes. The request's `prefill_cursor`
tells you how many tokens have been processed, so you can compute how many full blocks exist.

**Important:** you must `.clone()` the KV tensors before inserting into the cache. The
request's KV cache will continue to be modified (decode appends tokens), and you don't want
the cached block's data to be silently mutated via shared memory.

---

## Hint 7: Integrating with the Scheduler

The scheduler needs two additions:

### 7a: At admission time (`_maybe_admit`)

Before admitting a new request, check the block cache to see how much prefill work is
actually needed:

```python
def _maybe_admit(self, step):
    # ... existing admission checks (memory, batch size) ...

    candidate = self.waiting[0]  # top of heap

    # NEW: check prefix cache for hits
    num_cached = find_cached_prefix(
        self.block_cache,
        candidate.prompt_tokens,
        self.block_size,
    )

    # Actual KV cost is only the uncached portion
    actual_kv_cost = len(candidate.prompt_tokens) - num_cached

    if kv_used + actual_kv_cost > self.max_kv_tokens:
        return  # still can't fit

    # Admit and load cached blocks
    heapq.heappop(self.waiting)
    load_cached_blocks(candidate, self.block_cache, candidate.prompt_tokens, self.block_size)
    candidate.status = "prefilling"
    self.prefilling.append(candidate)
```

The admission check uses `actual_kv_cost` instead of the full prompt length. A fully-cached
prompt has `actual_kv_cost = 0` (just the trailing partial block), so it's nearly free to
admit.

### 7b: After each prefill step

Commit newly completed blocks to the cache:

```python
# In the generate loop, after processing a prefill chunk:
if prefill_req.is_fully_prefilled:
    commit_completed_blocks(prefill_req, scheduler.block_cache, BLOCK_SIZE)
    scheduler.promote(prefill_req)
```

You can also commit after each chunked prefill step (not just when fully prefilled) to make
blocks available sooner. This is what vLLM does — blocks are committed as soon as they're
complete, even if the request is still mid-prefill.

---

## Hint 8: Printing Cache Hit Diagnostics

Add logging to your generate loop so you can see prefix caching in action:

```python
# When admitting a request:
num_cached = find_cached_prefix(block_cache, req.prompt_tokens, BLOCK_SIZE)
print(f"[step {step}] Admitting req {req.id}: "
      f"{num_cached}/{len(req.prompt_tokens)} tokens cached "
      f"({num_cached // BLOCK_SIZE} blocks hit), "
      f"{len(req.prompt_tokens) - num_cached} tokens to prefill")

# When committing blocks:
print(f"[step {step}] Committed {num_new_blocks} blocks from req {req.id} to cache "
      f"(cache size: {len(block_cache.cache)}/{block_cache.max_blocks})")
```

Expected output when two requests share a prefix:

```
[step 0] Admitting req 0: 0/12 tokens cached (0 blocks hit), 12 tokens to prefill
[step 3] Committed 3 blocks from req 0 to cache (cache size: 3/64)
[step 4] Admitting req 1: 8/12 tokens cached (2 blocks hit), 4 tokens to prefill
                           ^^^ prefix cache hit!
```

---

## Test Scenarios

### Test 1: Identical prefixes

Two requests with identical prompts. The second request should reuse all complete blocks
from the first, prefilling only the trailing partial block (if any) plus zero new full blocks.

```python
requests = [
    Request(id=0, prompt_tokens=encode("To be or not to be"), max_new_tokens=20),
    Request(id=1, prompt_tokens=encode("To be or not to be"), max_new_tokens=20),
]
# req 1 should cache-hit on all full blocks from req 0
```

### Test 2: Shared prefix, different suffix

```python
shared = encode("You are a helpful assistant. ")
requests = [
    Request(id=0, prompt_tokens=shared + encode("Hello"), max_new_tokens=20),
    Request(id=1, prompt_tokens=shared + encode("Goodbye"), max_new_tokens=20),
]
# req 1 should cache-hit on the shared prefix blocks, prefill only the suffix
```

### Test 3: No shared prefix

```python
requests = [
    Request(id=0, prompt_tokens=encode("The cat sat on the mat"), max_new_tokens=20),
    Request(id=1, prompt_tokens=encode("Once upon a midnight dreary"), max_new_tokens=20),
]
# req 1 should have 0 cache hits — full prefill
```

### Test 4: Cache eviction under memory pressure

Set `max_blocks` small enough that the cache fills up. Verify that:
- LRU eviction removes the oldest-accessed block.
- New blocks are successfully inserted after eviction.
- Subsequent requests that would have hit the evicted block must re-prefill.

### Test 5: Output correctness

For all tests above, verify that the generated text is **identical** (same random seed)
whether prefix caching is enabled or disabled. The cache is a performance optimization —
it must not change the model's output.

---

## Summary of Changes from Scheduling Notebook

| Component | What Changes |
|-----------|-------------|
| New: `hash_block_tokens()` | Content-addressed block hashing with parent chains |
| New: `CachedBlock`, `BlockCache` | Global LRU cache for completed KV blocks |
| New: `find_cached_prefix()` | Walk prompt blocks, check cache hits |
| New: `load_cached_blocks()` | Copy cached KV onto a request's cache before prefill |
| New: `commit_completed_blocks()` | Insert newly-computed full blocks into the cache |
| `Scheduler._maybe_admit()` | Check cache hits to reduce actual prefill cost |
| `Scheduler.__init__()` | Add `block_cache` and `block_size` parameters |
| Generate loop | Call `commit_completed_blocks` after prefill steps |
| `Request` dataclass | Add `_committed_blocks` tracking field |
| Model / Head / assemble/disassemble | **Nothing changes** — prefix caching is pure Python above the model |

The key insight: **the model doesn't know anything about prefix caching.** It receives
KV tensors as `past_kvs` and can't tell whether they came from a fresh prefill or from
the cache. All the complexity lives in 100–150 lines of Python cache management above it.

---

## Gotchas

1. **Clone tensors before caching.** The request's KV cache is a live tensor that gets
   modified during decode (`torch.cat` appends new tokens). If you store a reference
   instead of a clone, the cached block's data will be silently corrupted as the request
   generates more tokens.

2. **Only cache full blocks.** A partial block's token content isn't finalized — it will
   change with the next decode step. Caching it would create stale entries that never match
   anything.

3. **Block size affects hit rate.** Smaller blocks = more granular matching = higher hit
   rate, but more overhead per block (more hash computations, more cache entries). A block
   size of 4–8 is reasonable for nanoGPT. In production vLLM uses 16.

4. **Chained hashes are essential.** Without the parent hash, you'd get false cache hits
   when two requests share token IDs in the same block position but have different preceding
   context. This would serve **corrupted KV data** and produce garbage output. Test 2 above
   validates this — if you break the chain, the suffix blocks would falsely match.

5. **Position alignment matters.** When loading cached KV, make sure the position embeddings
   used during the original prefill match what the request expects. Since KV is position
   dependent (through position embeddings in the input), a block cached at positions [0:4]
   can only be reused at positions [0:4]. Your current architecture handles this naturally
   because the prompt tokens are always processed from position 0.

6. **Prefix caching doesn't help decode.** It only accelerates prefill — decode tokens are
   always unique (they're generated, not shared). The optimization is purely about avoiding
   redundant KV computation for shared prompt prefixes.

---

## Implementation Checklist & Order

If you are implementing this from scratch (e.g., in `nanogpt-prefix-caching.py`), follow this exact order to build the caching layer logically without breaking the core engine:

**Step 1: Foundational Data Structures**
- Implement the hashing function `hash_block_tokens()` (ensure parent chaining is present).
- Implement the `CachedBlock` dataclass and the `BlockCache` class with its LRU `insert`, `lookup`, and `_evict_lru` logic.

**Step 2: Core Cache Operations**
- Implement `find_cached_prefix()`: Iterate through prompt blocks and stop at the first cache miss.
- Implement `load_cached_blocks()`: Fetch the KV data from the `BlockCache`, write it to the request's `kv_cache` dictionary, and advance the `prefill_cursor`.
- Implement `commit_completed_blocks()`: Slice full blocks from the request's live `kv_cache`, **clone them** (Gotcha #1), and insert them into `BlockCache`.
- Update the `Request` dataclass to include a `_committed_blocks` integer initialized to 0.

**Step 3: Modify the Scheduler Initialization**
- Update the `Scheduler.__init__` method to take and store `block_cache` and `block_size` parameters.

**Step 4: Update Scheduler Admission (`_maybe_admit`)**
- When evaluating a candidate in `_maybe_admit`, call `find_cached_prefix()` first to see how many tokens are already cached.
- Calculate `actual_kv_cost = len(candidate.prompt_tokens) - num_cached`.
- Use `actual_kv_cost` to check against `self.max_kv_tokens`.
- Once admitted, immediately call `load_cached_blocks()` to jump-start the request's `prefill_cursor` before appending it to `self.prefilling`.

**Step 5: Update the Generation Loop**
- In `scheduled_generate`, locate the section where the `prefill_req` chunk finishes and `p_req.kv_cache` is updated.
- Right after updating the `kv_cache`, call `commit_completed_blocks()` to push any newly-finished full blocks into the global cache.
- (Optional but recommended) Add print statements during admission and block commitment as detailed in Hint 8 to prove the cache is working.

**Step 6: Run Tests**
- Construct the 5 test scenarios (Identical prefixes, Shared prefixes with different suffixes, No shared prefix, Memory pressure, and Output correctness).
- Validate that the total generated text matches exactly with what a non-caching baseline would produce.
