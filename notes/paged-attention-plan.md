# PagedAttention — Implementation Plan & Hints

## The Problem You're Solving

Look at your current KV cache management. Every request owns a **contiguous tensor** per (layer, head):

```python
req.kv_cache[(layer_idx, head_idx)] = (k_tensor, v_tensor)
# k_tensor shape: (1, T_i, head_size) — grows by 1 every decode step via torch.cat
```

Every decode step does:
```python
k = torch.cat([past_k, k_new], dim=1)  # allocate a NEW (1, T+1, hs) tensor
v = torch.cat([past_v, v_new], dim=1)  # copy all old data + 1 new token
```

This has two costs that scale with sequence length:

1. **O(T) copy per step** — every `torch.cat` copies the entire cache history just to append one token.
2. **Memory fragmentation** — each request's cache is a different-sized contiguous slab. When requests finish, they leave holes that can't be reused by shorter or longer sequences.

At nanoGPT scale (32 max tokens, 210K params) this doesn't matter. At production scale (128K tokens, 70B params) it's the dominant bottleneck. The concept is what matters here.

**PagedAttention** replaces the contiguous cache with a **block table** — exactly like how an OS replaces contiguous RAM allocation with virtual memory pages. Each request gets a list of logical block indices that map to physical blocks in a shared pool. Appending a token writes into the current block's next slot; no copying. When a block fills up, allocate a fresh one from the pool.

---

## What You Already Have (Starting Point)

From your `nanogpt_interleaving.ipynb` and `nanogpt_prefix-caching.ipynb`:

- ✅ Per-request KV cache: `req.kv_cache[(layer, head)] = (k, v)` with shape `(1, T, hs)`
- ✅ `assemble_batch_cache` / `disassemble_batch_cache` — left-pads + stacks per-request caches
- ✅ `assemble_fused_batch` / `disassemble_fused_cache` — for interleaved decode+prefill
- ✅ Stateless `Head.forward()` with unified causal mask
- ✅ `BlockCache` for prefix caching (content-addressed KV blocks with LRU eviction)
- ✅ `BlockManager` skeleton in `paged_attention.py` (allocator only, no KV storage)

What's missing: the KV data itself is still stored as contiguous tensors on each request, grown via `torch.cat`. The `BlockManager` allocates block indices but doesn't store or retrieve actual KV data. The `Head` doesn't know about blocks.

---

## Hint 1: The Physical Block Pool — Where KV Data Actually Lives

Your existing `BlockManager` allocates block *indices* but has no actual KV storage. You need a global pool of **pre-allocated GPU tensors** that hold the KV data for all requests, all layers, all heads.

```python
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
        
        # Pre-allocate ALL memory upfront — no dynamic allocation during inference
        # k_pool[layer][head] = (num_blocks, block_size, head_size)
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

**Why pre-allocate?** In production, `torch.empty` / `torch.zeros` calls trigger CUDA memory allocation, which is slow and causes fragmentation. By allocating one big pool at startup, all subsequent "allocations" are just index bookkeeping — no CUDA malloc calls during inference.

**Question to ask yourself:** Why `(num_blocks, block_size, head_size)` instead of one giant `(num_blocks * block_size, head_size)` tensor? Because the block dimension lets you use `pool[block_indices]` to gather multiple non-contiguous blocks into a contiguous view with a single indexing operation. This is the paged gather.

---

## Hint 2: The Block Table — Logical → Physical Mapping

Each request maintains a **block table**: an ordered list of physical block indices that maps its logical token positions to physical pool locations.

```python
@dataclass
class PagedRequest:
    """Request with block-table-based KV cache."""
    id: int
    prompt_tokens: List[int]
    max_new_tokens: int
    generated_tokens: List[int] = field(default_factory=list)
    status: str = "waiting"
    prefill_cursor: int = 0
    
    # NEW: replaces kv_cache dict
    block_table: List[int] = field(default_factory=list)  # [phys_block_0, phys_block_1, ...]
    num_filled_slots: int = 0  # how many token slots are actually written
    
    @property
    def tokens_so_far(self):
        return self.prompt_tokens + self.generated_tokens
    
    @property
    def num_tokens_in_cache(self):
        return self.num_filled_slots
```

The mapping works like this:

```text
Logical token positions:  0  1  2  3 | 4  5  6  7 | 8  9  .  .
                          ─────────── ─────────── ───────────
Logical block:               0            1            2
Physical block:              7           14           22    (from block table)

block_table = [7, 14, 22]
num_filled_slots = 10    (slots 8-9 written, 10-11 empty in block 2)
```

To find where logical token position `t` lives:
```python
block_idx = t // block_size           # which logical block
slot_idx  = t % block_size            # which slot within that block
phys_block = block_table[block_idx]   # physical block index
# KV data lives at: pool[(layer, head)][phys_block, slot_idx, :]
```

---

## Hint 3: Writing KV Into the Pool — The Scatter Operation

When the model produces new K/V tensors during a forward pass, you need to **write** them into the correct physical block slots. This replaces `torch.cat`.

### During prefill (writing a chunk of tokens):

```python
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
        slot_idx  = logical_pos % block_size
        phys_block = block_table[block_idx]
        
        pool.k_pool[(layer, head)][phys_block, slot_idx, :] = k_new[0, t, :]
        pool.v_pool[(layer, head)][phys_block, slot_idx, :] = v_new[0, t, :]
```

### During decode (writing 1 token):

Same logic, but `T_new = 1` and `start_pos = num_filled_slots`. If the current block is full (`start_pos % block_size == 0`), allocate a new physical block first:

```python
def maybe_allocate_block(request, block_allocator, block_size):
    """Allocate a new physical block if the current one is full."""
    if request.num_filled_slots % block_size == 0:
        new_block = block_allocator.allocate_one()
        request.block_table.append(new_block)
```

**Key insight:** No copying. The old data stays in place. You just write into the next slot.


# Visualizing `write_kv_to_pool`

Let's walk through exactly what this code does using a concrete example.

## The Setup

Imagine we have a **block size of 4 tokens**. 
We are processing a **prefill chunk of 5 new tokens** (`T_new = 5`).
The request already has **2 tokens** processed (`start_pos = 2`).

### 1. The Block Table (The Map)
This request already has two physical blocks assigned to it from the pool.
```text
Logical Block Index | Physical Block Index
--------------------|---------------------
Block 0             | Block 14
Block 1             | Block 22
```
In Python: `block_table = [14, 22]`

### 2. The Physical Pool (The Destination)
The pool is a giant 3D tensor holding all KV data for everyone. We're only looking at a specific `(layer, head)` slice.

```text
Physical Block 14 (Capacity: 4)      Physical Block 22 (Capacity: 4)
[ Slot 0 ] - (filled: Token 0)       [ Slot 0 ] - (empty)
[ Slot 1 ] - (filled: Token 1)       [ Slot 1 ] - (empty)
[ Slot 2 ] - (empty)                 [ Slot 2 ] - (empty)
[ Slot 3 ] - (empty)                 [ Slot 3 ] - (empty)
```

### 3. The New Data (`k_new`, `v_new` - The Source)
The model just computed the keys and values for our 5 new tokens.
```text
k_new (shape: 1, 5, head_size):
[ t0_data, t1_data, t2_data, t3_data, t4_data ]
```

---

## Step-by-Step Execution

The function loops over each of the 5 new tokens: `for t in range(5):`

### Iteration 1: `t = 0` (The 1st new token)

1. **Calculate Global Position:**
   `logical_pos = start_pos + t`  ➔  `2 + 0 = 2`
   (This is the 3rd token overall for this request).

2. **Find the Logical Block:**
   `block_idx = logical_pos // block_size`  ➔  `2 // 4 = 0`
   (It belongs in Logical Block 0).

3. **Find the Slot Inside the Block:**
   `slot_idx = logical_pos % block_size`  ➔  `2 % 4 = 2`
   (It goes into slot 2 of that block).

4. **Lookup the Physical Address:**
   `phys_block = block_table[block_idx]`  ➔  `block_table[0] = 14`
   (Logical Block 0 maps to Physical Block 14).

5. **Write the Data!**
   Take `t0_data` from `k_new` and write it to `Physical Block 14, Slot 2`.

---

### Iteration 2: `t = 1` (The 2nd new token)

1. `logical_pos = 2 + 1 = 3`
2. `block_idx = 3 // 4 = 0` (Still Logical Block 0)
3. `slot_idx = 3 % 4 = 3` (Slot 3)
4. `phys_block = block_table[0] = 14`
5. **Write!** `t1_data` goes to `Physical Block 14, Slot 3`.

*Uh oh, Physical Block 14 is now completely full! Let's see what happens next.*

---

### Iteration 3: `t = 2` (The 3rd new token)

1. `logical_pos = 2 + 2 = 4`
2. **Find the Logical Block:**
   `block_idx = 4 // 4 = 1`
   **(Notice the jump! We crossed the block boundary and moved to Logical Block 1).**
3. **Find the Slot:**
   `slot_idx = 4 % 4 = 0`
   **(The slot index wraps back around to 0).**
4. **Lookup the Physical Address:**
   `phys_block = block_table[1] = 22`
   **(We look up the next block in our table and find Physical Block 22).**
5. **Write!** `t2_data` goes to `Physical Block 22, Slot 0`.

---

### Iteration 4: `t = 3` (The 4th new token)
1. `logical_pos = 5`
2. `block_idx = 5 // 4 = 1`
3. `slot_idx = 5 % 4 = 1`
4. `phys_block = 22`
5. **Write!** `t3_data` goes to `Physical Block 22, Slot 1`.

---

### Iteration 5: `t = 4` (The 5th new token)
1. `logical_pos = 6`
2. `block_idx = 6 // 4 = 1`
3. `slot_idx = 6 % 4 = 2`
4. `phys_block = 22`
5. **Write!** `t4_data` goes to `Physical Block 22, Slot 2`.

---

## The Resulting Physical Memory

After the loop finishes, the shared physical pool looks like this:

```text
Physical Block 14                    Physical Block 22
[ Slot 0 ] - (filled: Token 0)       [ Slot 0 ] - (filled: Token 2 / t2_data)
[ Slot 1 ] - (filled: Token 1)       [ Slot 1 ] - (filled: Token 3 / t3_data)
[ Slot 2 ] - (filled: Token 2 / t0_data) [ Slot 2 ] - (filled: Token 4 / t4_data)
[ Slot 3 ] - (filled: Token 3 / t1_data) [ Slot 3 ] - (empty)
```

Notice how a contiguous tensor of 5 new tokens (`k_new`) was automatically sliced and scattered across two completely separate physical blocks in GPU memory, just by doing simple math on the logical positions.


---

## Hint 4: Reading KV From the Pool — The Gather Operation

During attention, the `Head` needs to read the full past KV for a request. Instead of receiving a contiguous `(1, T_past, hs)` tensor, it **gathers** from scattered physical blocks.

```python
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
    
    k_parts = []
    v_parts = []
    
    # Full blocks
    for i in range(num_full_blocks):
        phys = block_table[i]
        k_parts.append(pool.k_pool[(layer, head)][phys])   # (block_size, hs)
        v_parts.append(pool.v_pool[(layer, head)][phys])
    
    # Trailing partial block
    if trailing_slots > 0:
        phys = block_table[num_full_blocks]
        k_parts.append(pool.k_pool[(layer, head)][phys, :trailing_slots])  # (trailing, hs)
        v_parts.append(pool.v_pool[(layer, head)][phys, :trailing_slots])
    
    k = torch.cat(k_parts, dim=0).unsqueeze(0)  # (1, num_filled, hs)
    v = torch.cat(v_parts, dim=0).unsqueeze(0)
    return k, v
```

**Wait — this still uses `torch.cat`!** Yes, but the difference is fundamental:

| | Contiguous cache | Paged cache |
|---|---|---|
| **Where data lives** | Per-request tensor, reallocated each step | Shared pool, written in-place |
| **What `torch.cat` does** | Copies ALL old data + new → O(T) per step | Gathers read-only views → O(T) once for the forward pass |
| **Memory layout** | Each request owns separate memory | All requests share one pool |
| **Deallocation** | Whole tensor freed | Individual blocks returned to pool |

In production vLLM, the gather is fused into a custom CUDA kernel (the paged attention kernel) that reads directly from scattered blocks without materializing a contiguous tensor. At nanoGPT scale, the explicit gather is fine.


# Visualizing `gather_kv_from_pool`

Let's walk through the exact opposite process of the scatter: reading scattered blocks from memory and gluing them back into a single, contiguous tensor that the model can understand.

## The Setup

We'll use the same request from the previous example, which now has **7 tokens total** (`num_filled = 7`) and a **block size of 4 tokens**. 

### 1. The Block Table (The Map)
```text
Logical Block Index | Physical Block Index
--------------------|---------------------
Block 0             | Block 14
Block 1             | Block 22
```
In Python: `block_table = [14, 22]`

### 2. The Physical Pool (The Source)
The KV data is sitting in these scattered physical locations:
```text
Physical Block 14                    Physical Block 22
[ Slot 0 ] - Token 0                 [ Slot 0 ] - Token 4
[ Slot 1 ] - Token 1                 [ Slot 1 ] - Token 5
[ Slot 2 ] - Token 2                 [ Slot 2 ] - Token 6
[ Slot 3 ] - Token 3                 [ Slot 3 ] - (empty)
```

### 3. The Goal (The Destination)
We need to create a single tensor containing Tokens 0 through 6 in order.

---

## Step-by-Step Execution

### 1. Identify Full vs. Partial Blocks

The function first figures out how many full blocks we have, and if there's a trailing partial block:

*   **`num_full_blocks = num_filled // block_size`**  ➔  `7 // 4 = 1`
    *(We have 1 completely full block).*
*   **`trailing_slots = num_filled % block_size`**  ➔  `7 % 4 = 3`
    *(We have 3 leftover tokens in a partial block).*

### 2. Gather the Full Blocks
```python
for i in range(1): # loops just once for i=0
    phys = block_table[0]  # phys = 14
    k_parts.append(pool.k_pool[(layer, head)][14])
```
It looks up Logical Block 0 in the table, finds Physical Block 14, and grabs **the entire block** (all 4 slots) from the pool. It adds this chunk to the `k_parts` list.

`k_parts` now holds: `[ <Tensor with Tokens 0, 1, 2, 3> ]`

### 3. Gather the Trailing Partial Block
```python
if trailing_slots > 0: # 3 > 0 is True
    phys = block_table[1]  # phys = 22
    k_parts.append(pool.k_pool[(layer, head)][22, :3]) 
```
It looks up the *next* logical block (Block 1), finds Physical Block 22, but **crucially**, it only slices the first 3 slots (`:3`). It ignores the empty Slot 3. It adds this sliced chunk to the list.

`k_parts` now holds:
`[ <Tensor with Tokens 0, 1, 2, 3>, <Tensor with Tokens 4, 5, 6> ]`

### 4. Glue Them Together (`torch.cat`)

Finally, it concatenates the pieces in the list along the sequence dimension (dimension 0 for these slices).

```python
k = torch.cat(k_parts, dim=0)
```
This glues the two tensors together seamlessly:
`<Tensor with Tokens 0, 1, 2, 3, 4, 5, 6>`

Then, it adds a batch dimension of size 1 to the front (`.unsqueeze(0)`), resulting in the final shape `(1, 7, head_size)`.

---

## Summary

The `gather_kv_from_pool` function is essentially doing this:

1.  "Give me all of Physical Block 14."
2.  "Give me the first 3 slots of Physical Block 22."
3.  "Stick them end-to-end so the model thinks it was contiguous the whole time." 

*(In a real production system like vLLM, this `torch.cat` is replaced by a custom CUDA kernel that does the math while directly reading from the scattered locations, skipping the need to create a new contiguous tensor entirely. But at our scale, explicitly gathering it first works perfectly).*

---

## Hint 5: The Block Allocator — Managing the Free Pool

Extend your existing `BlockManager` to support the operations needed:

```python
class BlockAllocator:
    """Manages physical block allocation from the pool."""
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
    
    def allocate_one(self):
        """Allocate a single physical block. Raises if pool exhausted."""
        if not self.free_blocks:
            raise RuntimeError("Block pool exhausted!")
        return self.free_blocks.pop()
    
    def allocate_n(self, n):
        """Allocate n blocks for a prefill chunk."""
        if len(self.free_blocks) < n:
            raise RuntimeError(f"Need {n} blocks, only {len(self.free_blocks)} free")
        return [self.free_blocks.pop() for _ in range(n)]
    
    def free_blocks_for_request(self, block_table):
        """Return all blocks from a request back to the free pool."""
        self.free_blocks.extend(block_table)
    
    @property
    def num_free(self):
        return len(self.free_blocks)
```

**When to allocate:**
- At **admission** (prefill start): allocate enough blocks for the prompt + some headroom
- At **decode**, when the current last block fills up: allocate 1 new block
- At **request completion**: free all blocks back to the pool

**Question to ask yourself:** How many blocks does a prompt of length `P` need?
Answer: `ceil(P / block_size)`. For `P=10, block_size=4`: `ceil(10/4) = 3` blocks.

---

## Hint 6: Modifying Head.forward() — No Changes Needed!

Here's the surprising part: **`Head.forward()` doesn't change at all.**

The `Head` receives `past_k` and `past_v` as `(B, T_past, hs)` tensors. It doesn't know or care whether those tensors came from:
- A contiguous `torch.cat` chain (current approach)
- A paged gather from scattered blocks (PagedAttention)
- A prefix cache lookup (your existing prefix caching)

The gather happens **outside** the model, in the batch assembly functions. The model sees the same interface:

```python
logits, _, new_kvs = model(batch_tokens, pos=batch_positions, past_kvs=past_kvs, attn_mask=attn_mask)
```

The `past_kvs` are assembled from the paged pool instead of from per-request contiguous tensors. The model doesn't know the difference.

> **This is the same insight from your prefix caching plan:** "the model doesn't know anything about prefix caching." PagedAttention is the same — all the complexity lives in the Python cache management layer above the model.

---

## Hint 7: Rewriting assemble/disassemble for Paged KV

### Assembly (before the forward pass)

Replace `assemble_batch_cache` with a version that gathers from the pool:

```python
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
        attn_mask[i, 0, pad:] = True
    
    past_kvs = []
    for layer_idx in range(n_layer):
        block_kv = []
        for head_idx in range(n_head):
            keys, values = [], []
            for i, req in enumerate(requests):
                k, v = gather_kv_from_pool(
                    pool, req.block_table, block_size, 
                    req.num_filled_slots, layer_idx, head_idx
                )
                # Left-pad if needed
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
```

### Disassembly (after the forward pass)

Instead of storing the returned KV back onto `req.kv_cache`, **scatter** the new KV entries into the pool:

```python
def disassemble_paged_cache(requests, new_kvs, pad_lengths, pool, block_size):
    """
    Scatter new KV data from model output back into the paged pool.
    Each request gets 1 new KV entry (decode token).
    """
    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
            for i, req in enumerate(requests):
                pad = pad_lengths[i]
                # The new token's KV is at the last position (after stripping pad)
                k_new = batched_k[i:i+1, -1:, :]  # (1, 1, hs)
                v_new = batched_v[i:i+1, -1:, :]
                
                write_kv_to_pool(
                    pool, req.block_table, block_size,
                    req.num_filled_slots,  # position of the new token
                    k_new, v_new, layer_idx, head_idx
                )
    
    # Update filled counts (once, not per layer/head)
    for req in requests:
        req.num_filled_slots += 1
```

For the **fused** disassembly (interleaved decode + prefill), the prefill row writes `chunk_size` tokens instead of 1:

```python
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
```

---

## Hint 8: Integrating with the Scheduler

### At admission (`_maybe_admit`):

```python
def _maybe_admit(self, step):
    # ... existing checks ...
    
    candidate = self.waiting[0]
    prompt_len = len(candidate.prompt_tokens)
    blocks_needed = (prompt_len + self.block_size - 1) // self.block_size
    
    # Check if pool has enough free blocks
    if self.block_allocator.num_free < blocks_needed:
        return  # can't fit
    
    # Admit and allocate blocks
    heapq.heappop(self.waiting)
    candidate.block_table = self.block_allocator.allocate_n(blocks_needed)
    candidate.num_filled_slots = 0
    candidate.status = "prefilling"
    self.prefilling.append(candidate)
```

### At completion:

```python
def complete(self, req):
    self.active.remove(req)
    req.status = "done"
    # Return blocks to pool
    self.block_allocator.free_blocks_for_request(req.block_table)
```

### During decode (new block when current fills):

```python
# In the generate loop, before building the decode batch:
for req in decode_reqs:
    maybe_allocate_block(req, scheduler.block_allocator, block_size)
```

---

## Hint 9: Unifying with Prefix Caching (Optional)

Your existing prefix caching stores KV data inside `CachedBlock.kv_data` — a dict of `(layer, head) → (k, v)` tensors. With PagedAttention, cached blocks can instead store **physical block indices** that point into the shared pool.

```python
@dataclass
class CachedBlock:
    block_hash: bytes
    token_ids: tuple
    phys_block: int         # physical block index in the pool (replaces kv_data dict)
    last_access_step: int = 0
```

When a cache hit occurs, instead of copying KV tensors onto the request, you just add the physical block index to the request's block table:

```python
def load_cached_blocks_paged(request, block_cache, prompt_tokens, block_size):
    """Load cached blocks by sharing physical block references."""
    parent_hash = NONE_HASH
    num_cached = 0
    
    for start in range(0, len(prompt_tokens), block_size):
        end = start + block_size
        if end > len(prompt_tokens): break
        
        chunk = prompt_tokens[start:end]
        block_hash = hash_block_tokens(parent_hash, chunk)
        cached = block_cache.lookup(block_hash)
        if cached is None: break
        
        # Just reference the same physical block — no data copying!
        request.block_table.append(cached.phys_block)
        num_cached += block_size
        parent_hash = block_hash
    
    request.num_filled_slots = num_cached
    request.prefill_cursor = num_cached
    return num_cached
```

**This is copy-on-write.** Multiple requests can share the same physical block (read-only prefix). If a request needs to modify a shared block (unlikely for prompt-only blocks), you'd copy it first. At nanoGPT scale, prompts are never modified after prefill, so simple sharing works.

> **Important:** Shared blocks must NOT be freed when one request completes. You need reference counting or rely on the `BlockCache` to track which physical blocks are in use.

---

## Test Scenarios

### Test 1: Output equivalence

Run the same requests through both the old contiguous-cache `interleaved_generate` and the new paged version with the same random seed. Outputs must be **identical**. This validates that paging didn't change the attention computation.

### Test 2: Block allocation lifecycle

```python
# Track block allocation across the lifecycle of a request
req = PagedRequest(id=0, prompt_tokens=encode("Hello world!"), max_new_tokens=10)
# Prompt is 12 tokens, block_size=4 → needs 3 blocks

# After prefill: 3 blocks allocated, 12 slots filled
assert len(req.block_table) == 3
assert req.num_filled_slots == 12

# After 4 decode steps: slots 12-15 fill block 3, then slot 16 triggers block 4
assert len(req.block_table) == 5  # 3 (prompt) + 1 (filled) + 1 (new)
assert req.num_filled_slots == 16

# After completion: all blocks returned to pool
initial_free = allocator.num_free
scheduler.complete(req)
assert allocator.num_free == initial_free + 5
```

### Test 3: Memory reuse

```python
# Two sequential requests should reuse the same physical blocks
req_a = PagedRequest(id=0, prompt_tokens=encode("Hi"), max_new_tokens=5)
# ... run to completion, blocks freed ...
blocks_used_a = set(req_a.block_table)

req_b = PagedRequest(id=1, prompt_tokens=encode("Go"), max_new_tokens=5)
# ... run to completion ...
blocks_used_b = set(req_b.block_table)

# Blocks from req_a should be reused by req_b
assert blocks_used_a & blocks_used_b  # non-empty intersection
```

### Test 4: Pool exhaustion and preemption

Set `num_blocks` small enough that the pool fills up. Verify that:
- New requests are blocked from admission when the pool is full
- When a request completes and frees blocks, waiting requests can proceed

### Test 5: Prefix sharing (if implementing Hint 9)

Two requests with the same prompt prefix should share physical blocks:
```python
shared = encode("You are a helpful assistant. ")
req_a = PagedRequest(id=0, prompt_tokens=shared + encode("Hello"), max_new_tokens=10)
req_b = PagedRequest(id=1, prompt_tokens=shared + encode("Goodbye"), max_new_tokens=10)

# After both are admitted, shared prefix blocks should be the same physical blocks
shared_blocks = len(shared) // block_size
assert req_a.block_table[:shared_blocks] == req_b.block_table[:shared_blocks]
```

---

## Summary of Changes from Interleaving Notebook

| Component | What Changes |
|-----------|-------------|
| New: `KVBlockPool` | Pre-allocated GPU tensor pool for all KV data |
| New: `BlockAllocator` | Replaces simple free list in `paged_attention.py` with full alloc/free lifecycle |
| New: `gather_kv_from_pool()` | Reads scattered blocks into contiguous tensors for attention |
| New: `write_kv_to_pool()` | Writes new KV into physical block slots (replaces `torch.cat`) |
| `Request` dataclass | `kv_cache` dict → `block_table` list + `num_filled_slots` int |
| `assemble_batch_cache` | Gathers from pool instead of from `req.kv_cache` |
| `disassemble_batch_cache` | Scatters into pool instead of storing on `req.kv_cache` |
| `disassemble_fused_cache` | Same scatter logic, handles variable tokens per row |
| `Scheduler._maybe_admit` | Allocates blocks from pool instead of just checking KV token count |
| `Scheduler.complete` | Frees blocks back to pool |
| Generate loop | Calls `maybe_allocate_block` before decode steps |
| `Head.forward()` | **No changes** — receives the same `(B, T_past, hs)` interface |
| `GPTLanguageModel` | **No changes** |
| Prefix caching (optional) | `CachedBlock` stores `phys_block` index instead of KV tensor dict |

---

## Recommended Build Order

```
1. KVBlockPool + BlockAllocator          ← the memory infrastructure
2. write_kv_to_pool / gather_kv_from_pool ← scatter/gather ops
3. PagedRequest dataclass                ← replace kv_cache with block_table
4. assemble_paged_cache                  ← gather from pool for batched forward pass
5. disassemble_paged_cache               ← scatter back to pool after forward pass
6. Integrate with scheduled_generate     ← non-interleaved version first (simpler)
7. Test: output equivalence vs contiguous ← validate correctness
8. Integrate with interleaved_generate   ← fused version
9. (Optional) Unify with prefix caching  ← share physical blocks
```

---

## Gotchas

1. **Don't forget to allocate blocks before prefill.** The block table must have enough blocks allocated before `write_kv_to_pool` tries to index into it. For a prompt of length `P`, you need `ceil(P / block_size)` blocks allocated *before* the first forward pass.

2. **`num_filled_slots` must be updated after KV write, not before.** The write position is `num_filled_slots` (the next empty slot). If you increment first, you'll skip a slot and write to the wrong position.

3. **Decode block boundary.** When `num_filled_slots % block_size == 0`, the current block is full. You must allocate a new block *before* writing the next token. Check this at the start of each decode step, not after.

4. **Gather creates a copy, not a view.** The `torch.cat` in `gather_kv_from_pool` creates a new contiguous tensor. This is fine — the model needs contiguous input. But don't try to write back to it expecting the pool to be updated; use `write_kv_to_pool` for writes.

5. **Shared prefix blocks and freeing.** If two requests share physical blocks (prefix caching), don't free those blocks when one request completes. You need either reference counting or a rule like "only free blocks that aren't in the BlockCache." The simplest approach: skip Hint 9 initially and implement paging without prefix sharing first.

6. **The pool size determines max concurrent tokens.** With `num_blocks=64` and `block_size=4`, you can hold at most 256 tokens across all active requests. Size the pool based on `max_kv_tokens / block_size`.

7. **Position embedding limit.** Your position embedding table has `block_size=32` entries (positions 0–31). This limits total sequence length to 32 regardless of how many blocks you have. PagedAttention doesn't change this — it's a model architecture constraint.
