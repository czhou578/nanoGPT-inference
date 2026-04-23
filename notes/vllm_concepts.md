# How vLLM Implements Prefix Caching, Context Window Management & Batching

All code is in `vllm/vllm/` (the checked-out vLLM submodule).

---

## 1. Prefix Caching in Practice

### Core idea
Every "full" KV-cache block gets a **chained SHA/xxhash fingerprint** that captures its token content **and** its parent's hash. Two requests that share the same prompt prefix will produce identical block hashes, so the scheduler can reuse already-computed KV blocks instantly.

---

### `v1/core/kv_cache_utils.py` — hashing machinery

```python
# Line 36 — type alias: a hash is just bytes
BlockHash = NewType("BlockHash", bytes)

# Lines 535-562 — chain hash: hash(parent_hash, token_ids, extra_keys)
def hash_block_tokens(
    hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None
) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = NONE_HASH           # random seed (line 87)
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
```

Only **full** blocks are hashed. Partial trailing blocks are never cached — which is why a 100%-hit prompt still recomputes the last token (line 201, `kv_cache_manager.py`).

**Extra keys** extend the hash for multimodal, LoRA, or per-request salt:
```python
# Lines 497-532 kv_cache_utils.py
def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx, start_mm_idx):
    # Combines: mm_feature identifier + offset, LoRA name, cache_salt, prompt-embed hash
```

---

### `v1/core/block_pool.py` — the hash-to-block lookup table

```python
# Lines 57-127 — BlockHashToBlockMap: dict[BlockHash → KVCacheBlock | dict[id, Block]]
# Allows multiple physical blocks with the same logical content (dedup is intentionally off)

# Lines 184-209 — cache lookup by hash
def get_cached_block(self, block_hash, kv_cache_group_ids) -> list[KVCacheBlock] | None:

# Lines 211-320 — commit a freshly computed block into the cache
def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, ...):
    blk.block_hash = block_hash_with_group_id
    self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
```

---

### `v1/core/kv_cache_manager.py` — the scheduler-facing API

`num_new_computed_tokens` is the number of tokens from prompt that were *already* found in the cache.

```python
# Lines 176-216 — returns cached blocks & a computed-token count for a new request
def get_computed_blocks(self, request) -> tuple[KVCacheBlocks, int]:
    computed_blocks, num_new_computed_tokens = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
    )
```

```python
# Lines 257-427 — allocate_slots() integrates prefix hits into the token budget
# cached hits are *not* recomputed; only new tokens need GPU time
```

---

### `v1/core/sched/scheduler.py` — scheduler integration

```python
# Lines 610-652 — on first admission of a waiting request:
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
# num_new_tokens = request.num_tokens - num_computed_tokens
# so cached tokens are simply subtracted from the work to do
```

**LRU eviction**: The `FreeKVCacheBlockQueue` (`kv_cache_utils.py` lines 158-366) is a doubly-linked list. When allocating new blocks, the pool pops from the front (LRU). If a popped block has a hash it is evicted from `BlockHashToBlockMap` at that moment (`block_pool.py` line 354).

---

## 2. Context Window Management: Sliding Windows, KV Eviction, Long-Context Trade-offs

### `v1/kv_cache_interface.py` — the spec types

Three concrete window strategies, each encoding different memory budgets:

| Class | Key field | Memory formula |
|---|---|---|
| `FullAttentionSpec` (line 148) | `sliding_window: int \| None` | `ceil(max_model_len / block_size) * page_bytes` |
| `SlidingWindowSpec` (line 333) | `sliding_window: int` | Only `sliding_window - 1 + max_batched_tokens` tokens ever needed |
| `ChunkedLocalAttentionSpec` (line 313) | `attention_chunk_size: int` | One chunk worth of tokens max |

```python
# SlidingWindowSpec.max_memory_usage_bytes (lines 336-355)
num_tokens = min(self.sliding_window - 1 + max_num_batched_tokens, max_model_len)
return (cdiv(num_tokens, self.block_size) + 1) * self.page_size_bytes
# The +1 handles misalignment at the window boundary
```

---

### `v1/core/single_type_kv_cache_manager.py` — the eviction/skip logic

**`get_num_skipped_tokens`** computes how many old tokens are now outside the window and whose blocks can be freed:

```python
# SlidingWindowManager (lines 582-608)
def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
    return max(0, num_computed_tokens - self.sliding_window + 1)
    # e.g. sliding_window=4, 7 tokens computed → skip 4 tokens (blocks 0-3 freed)

# ChunkedLocalAttentionManager (lines 717-761)
def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
    return (num_computed_tokens // self.attention_chunk_size) * self.attention_chunk_size
    # e.g. chunk=8, 13 tokens → skip 8 (entire first chunk freed)
```

**`remove_skipped_blocks`** (lines 359-400) — called every scheduler step:
```python
def remove_skipped_blocks(self, request_id, total_computed_tokens):
    num_skipped_blocks = num_skipped_tokens // self.block_size
    for i in range(num_skipped_blocks - 1, -1, -1):
        if blocks[i] == self._null_block:
            break                             # already freed in a prior step
        removed_blocks.append(blocks[i])
        blocks[i] = self._null_block          # replace with sentinel
    self.block_pool.free_blocks(removed_blocks)  # returned to free_block_queue
```

**Prefix cache + sliding window interaction** — `find_longest_cache_hit` in `SlidingWindowManager` (lines 486-580) returns `[NULL, NULL, ..., block_k, block_k+1]` — null sentinels for out-of-window blocks, real cached blocks only for the in-window suffix. This means prefix caching still works for sliding-window models (you reuse the tail, not the head).

---

### `v1/core/kv_cache_utils.py` — memory budget check

```python
# Lines 665-716 — binary search for the longest context that fits in available GPU RAM
def estimate_max_model_len(vllm_config, kv_cache_spec, available_memory) -> int:
    while left <= right:
        mid = (left + right) // 2
        if fits_in_memory(mid):
            result = mid; left = mid + 1
        else:
            right = mid - 1
```

This is the "when to truncate" decision point — vLLM refuses to start if `max_model_len` requires more KV RAM than available.

---

## 3. Batching at the Application Layer

### Architecture: requests go in → EngineCore batches them

```
Client (HTTP)
  └─ AsyncLLM.generate()          # async_llm.py L523
       └─ add_request()           # L282 — enqueues into per-request queue
            └─ engine_core.add_request_async()   # sends to background process
  
Background asyncio task (output_handler):
  └─ engine_core.get_output_async()             # pulls from EngineCore
       └─ output_processor.process_outputs()    # fan-out to per-request generators
```

### `v1/engine/async_llm.py` — the async batching pump

```python
# Lines 634-704 — single background task, never blocks the API server
async def output_handler():
    while True:
        outputs = await engine_core.get_output_async()   # waits for one batch
        # process up to VLLM_V1_OUTPUT_PROC_CHUNK_SIZE outputs per event loop tick
        for start in range(0, num_outputs, chunk_size):
            processed = output_processor.process_outputs(outputs_slice, ...)
            if end < num_outputs:
                await asyncio.sleep(0)   # yield to event loop between chunks
```

Multiple concurrent `generate()` calls add requests to the **same** EngineCore queue. The scheduler drains it opportunistically every step.

### `v1/core/sched/scheduler.py` — the continuous batching scheduler

```python
# Lines 383-552 — RUNNING requests (already have KV blocks):
while req_index < len(self.running) and token_budget > 0:
    num_new_tokens = min(num_new_tokens, token_budget)
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is None:
        preempted_req = self.running.pop()   # evict lowest-priority request
        self._preempt_request(preempted_req)

# Lines 563-800 — WAITING requests (first prefill, chunked):
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:
        break
    num_new_tokens = min(request.num_tokens - num_computed_tokens, token_budget)
    # chunked_prefill: partial prefill fills remaining token_budget
```

Key knobs:
- `max_num_scheduled_tokens` (`token_budget`) — total tokens in one forward pass
- `max_num_running_reqs` — max concurrent sequences in the batch
- `long_prefill_token_threshold` — cap new tokens per long-prefill request per step
- `enable_chunked_prefill` — whether waiting requests can fill leftover budget

**Preemption** (line 500): when a new request can't get blocks, the *lowest-priority* running request is evicted and put back in the waiting queue. Its KV blocks are freed; it'll re-prefill (or hit prefix cache) on re-admission.

### `v1/core/sched/request_queue.py` — scheduling policies

```python
class SchedulingPolicy(Enum):
    FCFS = "fcfs"       # arrival order (default)
    PRIORITY = "priority"  # explicit priority field on the request
```

The `priority` + `arrival_time` tuple breaks ties deterministically.

---

## Summary Map

| Concept | Primary files | Key classes/functions |
|---|---|---|
| Prefix caching | `kv_cache_utils.py`, `block_pool.py`, `kv_cache_manager.py` | `hash_block_tokens`, `BlockHashToBlockMap`, `get_computed_blocks` |
| KV eviction (LRU) | `kv_cache_utils.py`, `block_pool.py` | `FreeKVCacheBlockQueue`, `_maybe_evict_cached_block` |
| Sliding window | `kv_cache_interface.py`, `single_type_kv_cache_manager.py` | `SlidingWindowSpec`, `SlidingWindowManager.get_num_skipped_tokens`, `remove_skipped_blocks` |
| Chunked local attention | same | `ChunkedLocalAttentionSpec`, `ChunkedLocalAttentionManager` |
| Max context memory check | `kv_cache_utils.py` | `estimate_max_model_len`, `check_enough_kv_cache_memory` |
| Continuous batching | `sched/scheduler.py` | `Scheduler.schedule()` |
| Async request intake | `engine/async_llm.py` | `AsyncLLM.generate()`, `output_handler` |
| Chunked prefill | `sched/scheduler.py` | `token_budget`, `enable_chunked_prefill`, `long_prefill_token_threshold` |

---

## Deep Dives

### Why does each block hash need to include its parent's hash?

**Short answer: KV values are not determined by token IDs alone — they depend on every token that came before them.**

In a transformer, the K and V tensors for a token at position `i` are computed from that token's embedding **after** it has been contextualized by attention over all prior positions. Two sequences that happen to share the same tokens in block `N` but diverge somewhere in an earlier block will produce completely different K/V tensors in block `N`. If you keyed the cache on block `N`'s token IDs alone you would get cache collisions between logically unrelated sequences and serve corrupted outputs.

Concrete counter-example without parent hash:

```
Request A prefix:  [ "The cat sat" ] [ "on the mat" ]
Request B prefix:  [ "The dog sat" ] [ "on the mat" ]
```

Block 2 (`"on the mat"`) has the same token IDs in both requests. But in request A, every token in block 2 attended to `"The cat sat"` when its K/V was computed; in request B it attended to `"The dog sat"`. The stored K/V tensors are numerically different even though the token IDs look the same.

By making each block's hash a function of `(parent_block_hash, this_block_token_ids)`:

```python
# kv_cache_utils.py line 560
hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
```

the hash for block 2 in request A encodes the entire history back to block 1, which encodes `"The cat sat"`. The same position in request B encodes `"The dog sat"` in its ancestor hash. The two block-2 hashes are therefore different, so the cache correctly treats them as distinct entries.

The chain also means a cache hit at block `k` implies a hit at every block `0..k-1` — the prefix is fully and transitively captured. This is exactly the property needed: if block `k` matches, all its ancestors matched first.

---

### Why does a 100%-cache-hit prompt still recompute the last token (or sometimes the whole last block)?

There are two related constraints at work:

#### Constraint 1 — you always need a forward pass to get output logits

The KV cache stores **key and value tensors** for past tokens. It does *not* store the transformer's **output logits** (the probability distribution over the vocabulary used to sample the next token). Logits are produced by the final linear layer applied to the *query* side of the last token.

So even if every single prompt token has a cached KV entry, vLLM still needs to run at least one token through the full forward pass to obtain the logits it needs to generate the first output token. There is no shortcut — the cached K/V context is fed in via paged attention, but the query (and hence the output) for the final position must be freshly computed.

#### Constraint 2 — only *full* blocks are ever cached, and block alignment forces rounding

`kv_cache_manager.py` line 201:
```python
# When all tokens hit the cache, we must recompute the last token
# to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
max_cache_hit_length = request.num_tokens - 1
```

This caps the maximum cache hit at `num_tokens - 1`, guaranteeing at least 1 token is left "new". But there is a second rounding effect: `allocate_slots()` requires `num_computed_tokens` to be **block-size aligned**. If `num_tokens - 1` is not a multiple of `block_size`, it gets rounded *down* to the nearest block boundary, which could push an entire additional block into the "recompute" bucket.

Example with `block_size = 16` and a 48-token prompt where all tokens would otherwise hit:

```
Blocks:  [ 0–15 ] [ 16–31 ] [ 32–47 ]
                              ^^ block 3: 16 tokens that cached

max_cache_hit_length = 47   (prompt_len - 1)
round down to block boundary → 32
→ tokens 32–47 are NOT treated as cached
→ entire block 3 is recomputed (16 tokens, not just 1)
```

The comment in the source acknowledges this:
> "This can trigger recomputation of an entire block, rather than just the single last token, because `allocate_slots()` requires `num_computed_tokens` to be block-size aligned. Removing this limitation could slightly improve performance in the future."

In summary: you always recompute at least the last token to get logits, and the block-alignment requirement can silently extend that to the entire final block.

---

### What exactly is the token budget?

**Short answer: it's a cap on the total number of tokens the GPU processes in a single forward pass, shared across every active request in the batch.**

#### Why a budget at all?

The GPU doesn't care how many *sequences* are in a batch — it cares about how many *tokens* it has to process. The matrix multiplications inside each transformer layer are proportional to the total token count across all requests, not to the number of distinct sequences. More total tokens → more compute → longer step time.

`max_num_batched_tokens` (the physical cap, set in `config/scheduler.py` line 49) is derived from how much GPU memory you can use for the input tensor and intermediate activations in a single forward pass. It's pre-allocated at startup:

```python
# gpu_input_batch.py line 103
self.max_num_batched_tokens = max_num_batched_tokens
# The input_ids tensor, position tensors, block tables, etc. are all
# allocated to this fixed size at startup to avoid dynamic allocation.
```

`max_num_scheduled_tokens` (the scheduler's working cap, `scheduler.py` line 56) is usually equal to `max_num_batched_tokens`, but is set smaller when speculative decoding is active:

```python
# vllm.py line 1388-1390
if self.scheduler_config.max_num_scheduled_tokens is None:
    self.scheduler_config.max_num_scheduled_tokens = (
        max_num_batched_tokens - scheduled_token_delta
    )
# scheduled_token_delta = max_new_draft_slots_per_seq * max_num_seqs
# The drafter will inject extra tokens mid-batch; leave headroom for them.
```

#### Token budget vs. max_num_seqs

These are two independent limits that both apply simultaneously:

| Limit | What it constrains | Typical value |
|---|---|---|
| `max_num_scheduled_tokens` | Total tokens across ALL requests in the batch | 8192–32768 |
| `max_num_seqs` | Number of distinct sequences in the batch | 128–512 |

You can hit either limit first depending on the workload:
- **Prefill-heavy**: a single long prompt (e.g. 4096 tokens) consumes half the token budget by itself. You might have only 2–3 sequences before the budget runs out.
- **Decode-heavy**: each sequence contributes exactly 1 token per step. With `max_num_seqs=256` and `max_num_scheduled_tokens=8192`, you'd hit the sequence limit (256) long before the token budget (8192).

#### How the budget is consumed in `schedule()`

```python
# scheduler.py line 367
token_budget = self.max_num_scheduled_tokens   # starts full each step

# --- RUNNING pass (already have KV blocks) ---
while req_index < len(self.running) and token_budget > 0:
    num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
    num_new_tokens = min(num_new_tokens, token_budget)   # can't exceed budget
    # ...allocate KV slots, schedule request...
    token_budget -= num_new_tokens                        # deduct

# --- WAITING pass (new admissions) ---
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    num_new_tokens = request.num_tokens - num_computed_tokens
    num_new_tokens = min(num_new_tokens, token_budget)   # chunk if needed
    # ...allocate KV slots, admit request...
    token_budget -= num_new_tokens
```

Running requests are served first. Whatever budget is left after serving all running requests is available for newly admitted requests (or their first chunk of prefill).

#### Chunked prefill: the budget as a slicing tool

Without chunked prefill, a waiting request either fits in full or blocks. With `enable_chunked_prefill=True` (default), the scheduler slices a long prompt into however many tokens the remaining budget allows:

```
Budget = 8192
Running decode requests consume: 200 tokens (200 seqs × 1 token each)
Remaining budget: 7992

New request arrives with 20,000-token prompt.
→ Schedule first 7992 tokens of prompt this step.
→ Request stays in WAITING, num_computed_tokens = 7992.
→ Next step: serve its next chunk alongside decode traffic.
```

This is why decoding and prefill can interleave in the same forward pass — they share the same flat token budget. The `long_prefill_token_threshold` config (line 80) further limits how many tokens a *single* long prompt can consume per step, preventing one giant request from starving all decode traffic:

```python
# scheduler.py line 674-676
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold   # hard cap per request per step
```

#### The invariant enforced at the end of `schedule()`

After all scheduling decisions are made, there is a final assertion (scheduler.py line 859):

```python
assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
```

This is the hard guarantee — the batch handed to the GPU worker will never exceed the pre-allocated tensor size. Violating it would cause a buffer overflow in the input tensors that were statically allocated at startup.

---

### How do we balance the number of forward passes against the tokens per pass?

**Short answer: you are operating inside a roofline model. Too few tokens per pass wastes memory bandwidth; too many tokens per pass wastes GPU compute and hurts latency. vLLM's scheduler implicitly navigates this without exposing it directly to users.**

---

#### The two bottlenecks a GPU can hit

Every forward pass involves two types of work:

1. **Loading weights from GPU memory (memory bandwidth bound)**
   - For each transformer layer, *all* weight matrices must be streamed from HBM into compute registers.
   - The weight data is the same regardless of how many tokens are in the batch: `L × D × 4D × bytes_per_param`.
   - This cost is *fixed per forward pass*, not per token.

2. **Matrix multiplications (compute bound)**
   - Each token in the batch causes multiplications through those weight matrices.
   - FLOP count scales with `T × L × D²` (roughly, ignoring heads and MLPs).
   - More tokens = more FLOPs, proportionally.

vLLM's `perf.py` tracks both precisely. For attention alone (`AttentionMetrics.get_num_flops_breakdown`, line 432):

```python
{
  "qkv_proj": 2 * T * D * (q + 2*kv) * d * L,   # scales with T (tokens)
  "attn_qk":  2 * q * TC * d * L,                # TC = sum(tokens × context_len)
  "out_proj":  2 * T * D * q * d * L,            # scales with T
}
```

And the memory read cost (`get_read_bytes_breakdown`, line 458):
```python
"qkv_weight": D * (q + 2*kv) * d * weight_byte_size * L  # fixed per pass
```

The weight reads are the same whether `T=1` or `T=8192`. The FLOPs scale linearly with `T`.

---

#### Arithmetic intensity and why it determines the regime

**Arithmetic intensity** = FLOPs ÷ bytes read from memory.

When intensity is low (few FLOPs relative to weight bytes loaded), the GPU is sitting idle waiting for the memory system to stream weights — this is the **memory-bandwidth-bound** regime. When intensity is high (lots of FLOPs per byte loaded), the GPU's compute units are the bottleneck — this is the **compute-bound** regime.

The crossover point ("ridge point" in roofline terminology) for an H100 is roughly:
```
Peak FLOPS / Peak HBM Bandwidth = 990 TFLOPs / 3.35 TB/s ≈ 295 FLOPs per byte
```

For a token to push the GPU into the compute-bound regime, you need enough tokens in the batch that the FLOPs per weight-byte read exceeds ~295. For a 7B-parameter model:
- **T = 1 (single decode token)**: intensity ≈ 2 FLOPs/byte → heavily memory-bound, GPU is ~1% utilized
- **T = 512 (mixed batch)**: intensity ≈ 1024 FLOPs/byte → compute-bound, GPU is well-utilized
- **T = 8192 (large prefill)**: fully compute-bound, saturates the tensor cores

This is why pure decode (one token per request per step) is wasteful — the GPU loads the entire model every step just to execute a tiny amount of math.

---

#### What happens at each extreme

**Extreme A: Many forward passes, very few tokens each (decode-only, small batch)**

```
T = 10 requests × 1 token = 10 tokens per forward pass
```

- Weight bytes loaded per token: `params_bytes / 10` — almost the entire model per token
- GPU utilization: nearly 0% on compute units; 100% spent waiting on HBM reads
- Output: 10 tokens generated per pass
- Throughput (tokens/second): very low, bottlenecked on memory BW
- Latency (per request): good — each request gets a response every step
- **This is the classic decode bottleneck.** Even 256 decoding requests × 1 token = 256 tokens, which for a 70B model may still be memory-bandwidth bound.

**Extreme B: Very few forward passes, huge numbers of tokens each (prefill-only, giant prompts)**

```
T = 8192 tokens from 1 request's prompt
```

- GPU fully compute-bound, high utilization
- Output: *0 tokens generated until the prefill is done* — the user sees no partial output
- One forward pass might take 200ms; TTFT (time to first token) is 200ms
- Any new request that arrives has to wait until this pass and all subsequent decode steps complete
- **This is the prefill starvation problem.** New requests queue up with 0 throughput from their perspective while an existing prefill monopolizes the GPU.

---

#### How vLLM's scheduler balances them

The scheduler doesn't compute arithmetic intensity directly, but its mechanisms achieve the same balance:

**1. Shared token budget with RUNNING-first priority**

Decode requests (1 token each) consume their budget first. If there are 256 running decode requests:
```
256 × 1 = 256 tokens consumed → 7936 tokens of budget remaining
```
Those 7936 tokens are available for prefill of new requests. Prefill and decode co-exist in the same forward pass, amortizing the weight-load cost across both.

**2. Chunked prefill prevents prefill domination**

Without chunking, a single 64K-token prompt would occupy the entire budget for ~8 steps (~8 forward passes) before any other request could proceed. With `enable_chunked_prefill=True`, it gets at most `remaining_budget` tokens per step, so decode requests are never fully blocked.

**3. `long_prefill_token_threshold` prevents a single request from starving decode**

```python
# scheduler.py line 674-676  
if 0 < threshold < num_new_tokens_for_this_request:
    num_new_tokens_for_this_request = threshold
```

Even if 7936 tokens of budget remain, a single long prefill won't take all 7936 — it's capped at `threshold` (e.g. 4096), leaving room for other waiting requests to be partially scheduled too.

**4. `max_num_seqs` prevents the opposite: too many decode requests at once**

If you have 2048 running decode requests × 1 token = 2048 tokens, that's fine for memory bandwidth utilization (still low arithmetic intensity) but you're now consuming a lot of KV cache memory to hold all 2048 contexts. `max_num_seqs` caps this to avoid thrashing the KV pool.

---

#### The metrics vLLM actually tracks for this balance

`stats.py` and `perf.py` expose exactly the values needed to diagnose which regime you're in:

| Metric | What it tells you | Where it's tracked |
|---|---|---|
| `num_running_reqs` | Concurrent decode sequences | `SchedulerStats.num_running_reqs` (line 174) |
| `num_generation_tokens` | Output tokens produced per step | `IterationStats.num_generation_tokens` (line 330) |
| `num_prompt_tokens` | Prefill tokens processed per step | `IterationStats.num_prompt_tokens` (line 345) |
| `time_to_first_tokens_iter` | TTFT per request | `IterationStats.time_to_first_tokens_iter` (line 336) |
| `inter_token_latencies_iter` | TPOT per request | `IterationStats.inter_token_latencies_iter` (line 337) |
| `num_flops_per_gpu` | Actual FLOPs dispatched | `PerfStats.num_flops_per_gpu` (perf.py line 97) |
| `num_read_bytes_per_gpu` | Memory bandwidth consumed | `PerfStats.num_read_bytes_per_gpu` (perf.py line 98) |

The ratio `num_flops_per_gpu / num_read_bytes_per_gpu` in `PerfStats` is the arithmetic intensity of that step. When this number is low (below the GPU's ridge point), you are memory-bound and should increase batch size or add more decode concurrency. When it's high, you are compute-bound and latency is being determined by raw FLOP throughput.

---

### What are per-request generators?

**Short answer: each active request gets its own mini async queue—a `RequestOutputCollector`—and `generate()` is an async Python generator that awaits tokens from that queue one at a time, independent of every other request.**

---

#### The problem they solve

`AsyncLLM` runs two concurrent activities on the same asyncio event loop:

1. **`output_handler` (background task)** — pulls batches of outputs from the `EngineCore` subprocess. Each batch contains new tokens for *every* request scheduled in that step. It must distribute these to the right callers without blocking.
2. **`generate()` (per-request task)** — called once per HTTP request by the API server. Must yield tokens back to the HTTP connection as they arrive, without knowing or caring about other requests.

If you used a single shared queue you'd need the caller to filter out tokens from other requests, adding per-token O(N) fan-out logic on the hot path. Instead, vLLM uses one queue per request.

---

#### `RequestOutputCollector`: the per-request slot

`output_processor.py` lines 45–106:

```python
class RequestOutputCollector:
    def __init__(self, output_kind, request_id):
        self.output: RequestOutput | Exception | None = None
        self.ready = asyncio.Event()          # wakes up the awaiting generate()

    def put(self, output):                    # called by output_handler — non-blocking
        if self.output is None:
            self.output = output
            self.ready.set()
        elif ...:
            self.output.add(output, aggregate=self.aggregate)  # merge deltas

    async def get(self):                      # called by generate() — suspends until ready
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        return output
```

The key design decisions:

- **`put` is synchronous and non-blocking.** The `output_handler` task never awaits when distributing tokens. It calls `put()` on each request's collector and immediately moves on to the next.
- **`get` suspends the caller coroutine via `asyncio.Event`.** The `generate()` loop awaits `q.get()`, releasing the event loop to other tasks while waiting for the next token.
- **Delta merging**: if `output_handler` produces two tokens before `generate()` calls `get()` again, the second token is merged into the first with `output.add(output, aggregate=True)`. The consumer always sees at most one pending item.

---

#### `generate()`: the async generator loop

`async_llm.py` lines 523–593:

```python
async def generate(self, prompt, sampling_params, request_id, ...) -> AsyncGenerator:
    # 1. Register a RequestOutputCollector for this request.
    q = await self.add_request(request_id, prompt, sampling_params, ...)

    # 2. Loop until the request is done, yielding each token to the API caller.
    finished = False
    while not finished:
        # Try non-blocking first (avoids task switching under load).
        out = q.get_nowait() or await q.get()
        finished = out.finished
        yield out    # <-- this is what the HTTP streaming handler receives
```

`generate()` is an **async generator function** — the `yield` keyword turns it into something the API server can iterate with `async for token in engine.generate(...)`. Each iteration suspends until `output_handler` puts a new `RequestOutput` into the collector.

**Critically, each `generate()` call has its own independent `q`** (a `RequestOutputCollector`), so 200 concurrent HTTP requests run 200 independent async generator coroutines, each suspended on their own `asyncio.Event`, with zero interaction between them. The `output_handler` task fans out in a plain Python loop — no locking, no shared state.

---

#### Full data flow for one token

```
[EngineCore subprocess]
  → GPU forward pass produces tokens for requests A, B, C
  → sends EngineCoreOutputs batch over multiprocessing queue

[output_handler task, async_llm.py]
  → receives batch
  → for each output in batch:
      req_state = output_processor.request_states[req_id]
      output = req_state.make_request_output(new_token_ids, ...)
      req_state.queue.put(output)    ← non-blocking, sets asyncio.Event

[generate() for request A, suspended on q.get()]
  → asyncio.Event fires
  → out = q.get() returns RequestOutput
  → yield out  → HTTP SSE frame sent to the client
  → loop, suspend again on next q.get()
```

The output_handler and each generate() run as separate asyncio coroutines. Because `asyncio` is single-threaded, they don't truly run in parallel — but they interleave: `output_handler` does one batch distribution pass, then each generate() gets a chance to `yield` before the event loop returns control to `output_handler` for the next batch.

---

### Why do we cap the number of concurrent sequences, and is it about VRAM?

**Yes, but it's more than just VRAM. There are four separate limits at work, and `max_num_seqs` is the single knob they all flow through.**

---

#### Limit 1 — KV cache blocks (the VRAM you're thinking of)

Every active sequence holds KV blocks in GPU VRAM. Each block stores the K and V tensors for `block_size` tokens across all layers:

```
bytes_per_block = 2 × num_layers × num_kv_heads × head_dim × block_size × bytes_per_element
```

For Llama-3-8B (`num_layers=32, num_kv_heads=8, head_dim=128, block_size=16, bf16`):
```
2 × 32 × 8 × 128 × 16 × 2 ≈ 4.2 MB per block
```

A request generating 2048 tokens needs `2048 / 16 = 128 blocks` ≈ **537 MB**. With 256 concurrent sequences each generating long sequences, that's potentially **137 GB** just for KV cache — well beyond a single GPU. The block pool derived from `gpu_memory_utilization` (typically 90% of remaining VRAM after model weights) is finite, and the scheduler will not admit new requests if there aren't enough free blocks to handle the full expected output.

The number of GPU blocks is computed at startup in `gpu_worker.py` line 520 — after a profiling run loads the model and activations into VRAM, whatever VRAM is left (× `gpu_memory_utilization`) is converted into KV blocks.

#### Limit 2 — Block table tensor (statically pre-allocated)

Each request needs a row in the **block table** — a tensor that maps sequence position → physical block ID, used by the attention kernel. This table is statically allocated at startup to `max_num_seqs × max_model_len / block_size`. Every slot is allocated even if it goes unused.

```python
# block_table.py line 76
self.block_ids = torch.zeros(
    self.max_num_batched_tokens,   # flat layout
    dtype=torch.int64
)
```

If `max_num_seqs=2048` and `max_model_len=32768/block_size=16`, the table pre-allocates `2048 × 2048 = 4M` int64 entries ≈ **32 MB** on GPU. Not huge, but fixed — you can't hold more requests than the table has rows.

#### Limit 3 — Sampler and logit tensors (per-request output tensors)

At the end of each forward pass, the sampler runs on `num_active_seqs × vocab_size` logit values. For `vocab_size=128256` (Llama 3.1) and 512 sequences:
```
512 × 128256 × 4 bytes ≈ 262 MB
```

The error message in `gpu_model_runner.py` line 5618–5625 confirms:
```python
except RuntimeError as e:
    if "out of memory" in str(e):
        raise RuntimeError(
            "CUDA out of memory occurred when warming up sampler with "
            f"{num_reqs} dummy requests. Please try lowering "
            "`max_num_seqs` or `gpu_memory_utilization` ..."
        )
```

If you set `max_num_seqs` too high, the sampler warmup itself OOMs — before any real requests have even arrived.

#### Limit 4 — Per-step scheduling overhead (CPU)

For each step, `schedule()` iterates over `self.running` (a list of all active requests) to compute token budgets, check preemption, and allocate KV slots. This loop runs on CPU while the GPU runs the previous step:

```python
# scheduler.py line 367
token_budget = self.max_num_scheduled_tokens
for req in self.running:             # ← O(N) over all active requests
    num_new_tokens = ...
    token_budget -= num_new_tokens
    ...
```

At low sequence counts (≤256), this is negligible. At 2048+ sequences, the CPU scheduling loop can start to lag behind the GPU, creating idle GPU cycles. `max_num_seqs` prevents this by bounding N.

---

#### Summary: the four concurrent-sequence constraints

| Constraint | What it limits | Where enforced |
|---|---|---|
| **KV block pool** | Total token capacity across all active sequences | `block_pool.py`, `kv_cache_manager.py` |
| **Block table size** | Max rows in the physical block mapping tensor | `block_table.py` (pre-allocated at startup) |
| **Sampler logit tensor** | Output memory for `num_seqs × vocab_size` | `gpu_model_runner.py` warmup |
| **Scheduler loop latency** | CPU O(N) overhead per step | `scheduler.py` `schedule()` |

`max_num_seqs` caps all four at once. The practical binding constraint at typical deployment sizes (batch ≤ 512) is almost always **KV VRAM**: once the block pool is exhausted, the scheduler preempts low-priority requests (swapping their KV blocks out or simply recomputing them), which is expensive. Setting `max_num_seqs` conservatively avoids thrashing.
