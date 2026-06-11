# Sliding Window KV Eviction — Implementation Plan & Hints

## Base File: `nanogpt-scheduling.py`

**Output file:** `nanogpt-sliding-window.py`

### Why this base file?

`nanogpt-scheduling.py` is the ideal starting point because it has exactly the right infrastructure:

- **Per-request KV caches** — `req.kv_cache[(layer, head)] = (k, v)` with shape `(1, T_i, head_size)`, where `T_i` grows unbounded. This is the thing we're going to cap.
- **Scheduler with memory budgets** — `max_kv_tokens` already limits total KV memory. Sliding window makes this budget go further by keeping each request's cache bounded.
- **Preemption** — `_maybe_preempt()` evicts requests when memory is over budget. Sliding window reduces the frequency of preemption because each request uses less memory.
- **Continuous batching** — `assemble_batch_cache()` / `disassemble_batch_cache()` already handle variable-length caches across requests. Sliding window doesn't break this.
- **Chunked prefill** — Prefill cursor + position embeddings are already wired correctly.

We don't need paged attention, prefix caching, speculative decoding, or radix trees — those add complexity without illuminating the core concept. Sliding window is a pure **memory management** optimization that sits between the scheduler and the KV cache.

---

## The Problem You're Solving

Right now, every request's KV cache grows without bound during decode. If a request generates 500 tokens on top of a 100-token prompt, its KV cache holds all 600 token positions — even though the model may only need the most recent ~200 to produce coherent output.

This has three consequences:

1. **Memory waste** — Old KV entries for tokens that barely influence attention consume GPU HBM that could hold more concurrent requests.
2. **Unnecessary preemption** — The scheduler's `max_kv_tokens` budget fills up faster, triggering preemption and forcing expensive re-prefills.
3. **Slower batch assembly** — `assemble_batch_cache()` left-pads to the longest cache in the batch. Unbounded caches mean more padding and larger tensors.

### The insight

In most transformer models, attention is **local** — the softmax distribution concentrates heavily on recent tokens. For NanoGPT trained on Shakespeare, the model primarily conditions on the last ~50–100 characters of context. Tokens from 300 positions ago contribute essentially zero attention weight.

Sliding window exploits this: keep only the last `W` KV entries per request, and discard everything older. The memory cost per request becomes **bounded by `W`** instead of growing linearly with sequence length.

### How vLLM does it

vLLM's `SlidingWindowManager` (in `v1/core/single_type_kv_cache_manager.py`) implements this with block-level granularity:

```python
# SlidingWindowManager.get_num_skipped_tokens()
def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
    return max(0, num_computed_tokens - self.sliding_window + 1)
```

Old blocks outside the window are replaced with `NULL_BLOCK` sentinels and freed back to the pool. Our implementation will be simpler — we'll slice the KV tensors directly since we don't use paged attention in this file.

---

## Hint 1: Add a `sliding_window` Parameter to the Scheduler

The sliding window size `W` is a system-level configuration, not a model parameter. Add it to the `Scheduler` constructor alongside `max_kv_tokens`:

```python
class Scheduler:
    def __init__(self, policy='fcfs', max_batch_size=4, token_budget=16,
                 max_kv_tokens=22, sliding_window=None):
        ...
        self.sliding_window = sliding_window  # None = no window (keep all), int = max KV entries
```

When `sliding_window is None`, the system behaves identically to vanilla `nanogpt-scheduling.py`. This lets you A/B test windowed vs. full-cache in the same benchmark suite.

**Question to ask yourself:** Should the window include both prompt and generated tokens? Or only generated tokens? Think about what happens if `W < len(prompt_tokens)`.

The window should include both prompt and generated tokens. If W < len(prompt_tokens), the prompt will be truncated. This is acceptable behavior for sliding window, as it reflects the fact that the model can only attend to the last W tokens.

---

## Hint 2: Write an `evict_kv_cache()` Function

The core of sliding window is a single function that trims a request's KV cache to at most `W` entries:

```python
def evict_kv_cache(request, window_size):
    """
    Trim the request's KV cache to keep only the last `window_size` entries.
    
    Before: kv_cache[(layer, head)] = (k, v) with shape (1, T, hs)
    After:  kv_cache[(layer, head)] = (k, v) with shape (1, min(T, W), hs)
    """
    if window_size is None:
        return  # no window configured

    for (layer, head), (k, v) in request.kv_cache.items():
        T = k.shape[1]
        if T > window_size:
            request.kv_cache[(layer, head)] = (
                k[:, -window_size:, :],   # keep the LAST W entries
                v[:, -window_size:, :],
            )
```

This is the `[:, -W:, :]` slice — Python's negative indexing makes it a one-liner. The old tensor entries are now unreferenced and will be garbage-collected.

**Key insight:** We always keep the **most recent** `W` entries, not the first `W`. The most recent tokens have the strongest influence on the attention distribution. This is exactly what vLLM's sliding window does at the block level.

---

## Hint 3: Where to Call Eviction in the Generate Loop

Eviction should happen **after each decode step writes new KV entries but before the next batch assembly**. The natural place is right after `disassemble_batch_cache()`:

```python
# In scheduled_generate(), after the decode block:
disassemble_batch_cache(scheduler.active, new_kvs, pad_lengths)

# NEW: trim KV caches to sliding window size
if scheduler.sliding_window is not None:
    for req in scheduler.active:
        evict_kv_cache(req, scheduler.sliding_window)

# ... existing token generation and completion logic ...
```

**Why here?** At this point, the new KV entries from the decode step have been written back to per-request caches (`disassemble_batch_cache` just ran). If we evict now, the next call to `assemble_batch_cache` will see shorter caches → smaller batched tensors → less memory usage.

**Should we also evict during prefill?** For chunked prefill, the KV cache can grow to the full prompt length (`len(prompt_tokens)`) before any eviction. You could evict after each prefill chunk too, but be careful — evicting mid-prefill means the model computes attention for later prompt tokens without access to early prompt tokens. This changes the output. Decide whether this is acceptable for your use case (it usually is for long prompts where early tokens have low attention weight anyway).

---

## Hint 4: Fix the Position Embedding Problem

This is the subtle bug you'll hit. Right now, position embeddings are computed from the KV cache length:

```python
# In the decode path of scheduled_generate():
batch_positions = torch.tensor(
    [[len(req.tokens_so_far) - 1] for req in scheduler.active],
    device=device
)
```

After eviction, `req.kv_cache[(0,0)][0].shape[1]` (the KV cache length) is `W`, but the **true position** of the current token in the sequence is still `len(req.tokens_so_far) - 1` (which could be much larger than `W`).

**This is already handled correctly** in `nanogpt-scheduling.py` because positions are computed from `req.tokens_so_far` (the logical sequence), not from the KV cache shape. The position embedding uses the absolute position, and the KV cache is just the set of past K/V vectors the model attends over. These are decoupled.

However, **the `pos` value must not exceed `block_size - 1`** (since that's the max positional embedding index). If you generate more tokens than `block_size`, you'll get an index-out-of-bounds error on `self.position_embedding_table(pos)`. This is a pre-existing limitation of NanoGPT's learned positional embeddings (RoPE would fix it, but that's a separate optimization).

**Verify:** Make sure your test prompts + max_new_tokens don't exceed `block_size` total. With `block_size=64`, a 20-token prompt + 40 generated tokens = 60, which is safe.

---

## Hint 5: Update the Memory Budget Calculation

The scheduler's `_maybe_admit()` currently estimates KV usage as:

```python
kv_tokens_used = sum(len(req.prompt_tokens) + req.num_generated for req in self.active)
```

With sliding window, the actual KV memory per request is capped at `min(total_tokens, W)`:

```python
def _effective_kv_tokens(self, req):
    """How many KV entries this request actually holds (after eviction)."""
    total = len(req.prompt_tokens) + req.num_generated
    if self.sliding_window is not None:
        return min(total, self.sliding_window)
    return total

def _maybe_admit(self, step):
    ...
    kv_tokens_used = sum(self._effective_kv_tokens(r) for r in self.active)
    ...
```

This is critical — without this fix, the scheduler overestimates memory usage and unnecessarily refuses to admit new requests. The whole point of sliding window is to pack more requests into the same memory budget.

Similarly, update `_maybe_preempt()` to use `_effective_kv_tokens()`:

```python
def _maybe_preempt(self):
    kv_tokens_used = sum(self._effective_kv_tokens(r) for r in self.active + self.prefilling)
    ...
```

---

## Hint 6: The Attention Mask Interaction

In `assemble_batch_cache()`, the attention mask marks which KV positions are valid (True) vs. padding (False). Sliding window doesn't change this logic — all positions in a windowed cache are valid. The left-padding in `assemble_batch_cache` already handles different-length caches across requests.

However, there's a subtlety: after eviction, different requests may have different cache lengths even if they've generated the same number of tokens (because their prompts had different lengths and some are still within the window while others have been trimmed). This is already handled correctly by the existing `pad_lengths` computation.

**No changes needed to `assemble_batch_cache` or `disassemble_batch_cache`.**

---

## Hint 7: Benchmark Design — Proving the Value

Design benchmarks that show the three key benefits of sliding window:

### Benchmark 1: Memory Savings

Compare `max(kv_cache_length)` across requests with and without sliding window:
```
Without window: request generates 50 tokens on 20-token prompt → KV size = 70
With window=32: same request → KV size = 32 (54% reduction)
```

### Benchmark 2: Higher Throughput via Reduced Preemption

Set up a scenario with tight `max_kv_tokens` that forces preemption without a window, but doesn't with a window:
```python
requests = [
    Request(id=0, prompt_tokens=encode("ROMEO: " * 3), max_new_tokens=30),
    Request(id=1, prompt_tokens=encode("JULIET: " * 3), max_new_tokens=30),
    Request(id=2, prompt_tokens=encode("NURSE: " * 3), max_new_tokens=30),
]

# Without window: total KV grows to ~90 tokens/request × 3 = 270 → preemption fires at max_kv=200
# With window=32: total KV caps at 32 × 3 = 96 → no preemption needed
```

Measure: total wall-clock time, number of preemptions, number of re-prefills.

### Benchmark 3: Quality vs. Window Size Sweep

The key tradeoff. Generate text with different window sizes and compare output quality:
```
window_sizes = [8, 16, 32, 64, None]  # None = no eviction (full cache)
```

For each window size:
- Generate 50 tokens from the same prompt with the same random seed
- Compare: at which window size does the output diverge from the full-cache output?
- Measure: token-by-token agreement rate with the no-eviction baseline

For NanoGPT on Shakespeare, you'll likely find that `W ≥ 32` produces nearly identical output to full-cache, while `W ≤ 8` diverges significantly. This demonstrates the **locality of attention** property.

### Benchmark 4: Effective Batch Capacity

With a fixed `max_kv_tokens` budget, how many concurrent requests can the scheduler hold?
```
Without window: max_kv_tokens=200, each request uses ~60 tokens → 3 concurrent requests
With window=32: max_kv_tokens=200, each request uses ≤32 tokens → 6 concurrent requests (2× more)
```

---

## Hint 8: What This DOESN'T Handle (Scope Limitations)

To keep the implementation clean, this plan intentionally omits:

1. **Importance-based eviction** — We always evict the oldest tokens (FIFO). A smarter approach would track cumulative attention scores and keep high-attention tokens regardless of position. This is the H₂O (Heavy-Hitter Oracle) algorithm from the research literature.

2. **Sink tokens** — The first few tokens of a sequence (especially special tokens or sentence-initial tokens) often receive disproportionately high attention even at long distances. Production systems like StreamingLLM keep a small number of "attention sinks" (the first ~4 tokens) even when evicting the rest of the window. You could add this as a follow-up.

3. **Block-level eviction** — vLLM evicts at the block granularity (groups of `block_size` tokens), not individual tokens. This is because paged attention organizes memory in blocks. Since we're not using paged attention in this file, token-level slicing is simpler and cleaner.

4. **Interaction with prefix caching** — If a cached prefix is loaded from the block cache / radix tree, and the sliding window evicts part of it, those prefix blocks should ideally remain in the cache for other requests. This interaction is complex — leave it for a future file that combines both.

---

## Summary of Changes vs. `nanogpt-scheduling.py`

| Component | What Changes |
|-----------|-------------|
| `Scheduler.__init__` | Add `sliding_window` parameter |
| `Scheduler._effective_kv_tokens()` | New method — returns `min(total_tokens, W)` |
| `Scheduler._maybe_admit()` | Use `_effective_kv_tokens()` for budget check |
| `Scheduler._maybe_preempt()` | Use `_effective_kv_tokens()` for budget check |
| `evict_kv_cache()` | **New function** — trims KV tensors to last `W` entries |
| `scheduled_generate()` | Call `evict_kv_cache()` after each decode step |
| Model / Head / assemble / disassemble | **Nothing changes** — eviction is pure Python above the model |
| Benchmarks | New benchmark file for window size sweep + memory + throughput |

The key insight: **the model doesn't know anything about sliding window**. The KV tensors it receives are just shorter. All eviction logic lives in ~20 lines of Python between the scheduler and the model.

---

## Recommended Implementation Order

1. **Step 1: Copy `nanogpt-scheduling.py` → `nanogpt-sliding-window.py`**
   - Update the module docstring to describe sliding window KV eviction.
   - Change the import to use a new benchmark file (e.g., `sliding_window_benchmark_runs.py`).

2. **Step 2: Add the `sliding_window` parameter (Hint 1)**
   - Add it to `Scheduler.__init__`.
   - Add `_effective_kv_tokens()` helper method.

3. **Step 3: Implement `evict_kv_cache()` (Hint 2)**
   - A standalone function that takes a `Request` and `window_size`.
   - Iterates over `request.kv_cache` and slices each tensor to `[:, -W:, :]`.

4. **Step 4: Wire eviction into the generate loop (Hint 3)**
   - Call `evict_kv_cache()` after `disassemble_batch_cache()` in the decode block.
   - Optionally, also call it after each prefill chunk (document the tradeoff).

5. **Step 5: Update memory budgeting (Hint 5)**
   - Replace raw `len(prompt_tokens) + num_generated` with `_effective_kv_tokens()` in both `_maybe_admit()` and `_maybe_preempt()`.

6. **Step 6: Write the benchmark suite (Hint 7)**
   - Create `benchmarks/sliding_window_benchmark_runs.py`.
   - Implement the four benchmark scenarios: memory savings, preemption reduction, quality sweep, batch capacity.

7. **Step 7: Verify correctness**
   - With `sliding_window=None`, output must be **identical** to `nanogpt-scheduling.py` (same random seed, same tokens).
   - With `sliding_window=block_size` (= 64, the maximum context), output should still be identical (the window is as large as the max sequence).
   - With `sliding_window < block_size`, output will diverge — verify it degrades gracefully, not catastrophically.

---

## The Conceptual Map

```
Full KV cache (no eviction):

  Position:  [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] ...
  KV cache:   K₀  K₁  K₂  K₃  K₄  K₅  K₆  K₇  K₈  K₉  K₁₀  K₁₁  K₁₂ ...
                                                                  ↑
                                                          attention query

  Memory: O(n) — grows linearly with sequence length

─────────────────────────────────────────────────────────

Sliding window (W=6):

  Position:  [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] ...
  KV cache:                                       K₇  K₈  K₉  K₁₀  K₁₁  K₁₂
                                                   ↑───── window ────↑
                                                                  ↑
                                                          attention query

  Memory: O(W) — bounded, regardless of sequence length
  Evicted: K₀ through K₆ — freed, no longer in GPU memory

─────────────────────────────────────────────────────────

The tradeoff:

  Attention weights for token at position 12:

  Full cache:     [0.01, 0.01, 0.00, 0.00, 0.02, 0.01, 0.03, 0.05, 0.07, 0.15, 0.20, 0.25, 0.20]
  Window (W=6):                                         [0.05, 0.07, 0.17, 0.22, 0.27, 0.22]
                                                         ↑ renormalized over remaining entries

  The attention mass on evicted tokens (positions 0–6) was only ~8% total.
  Redistributing it across the window barely changes the output distribution.
```
