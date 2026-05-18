# Token Budget & Decode-Prefill Interleaving — Implementation Plan & Hints

## The Problem You're Solving

Look at what happens in your current `nanogpt_chunked-prefill.ipynb`. The prefill chunk and the decode batch run as **two separate `model()` calls** within the same step:

```
step 5:
  call 1: model(prefill_chunk)     ← 8 tokens for the prefilling request
  call 2: model(decode_batch)      ← 3 tokens (one per active decode request)
```

This works correctly, but it's **two forward passes per step**. On a real GPU, each forward pass has fixed overhead (kernel launches, memory transfers). A production inference engine (vLLM) **interleaves** decode and prefill tokens into a **single forward pass**, because the model doesn't care whether a token in the batch is a decode token or a prefill token — it just sees an input tensor of shape `(B, T)`.

**Decode-prefill interleaving** merges both types of work into one `model()` call per step. The token budget constrains total tokens per step, decode requests get first priority (they're cheap and already own KV memory), and the remaining budget goes to prefill chunks.

---

## What You Already Have (Starting Point)

Your `continuous_batching_generate` in `nanogpt_chunked-prefill.ipynb` already has:

- ✅ Token budget (`token_budget` parameter)
- ✅ Decode-first priority (`remaining_budget = token_budget - len(active_requests)`)
- ✅ Chunked prefill with `prefill_cursor`
- ✅ `assemble_batch_cache` / `disassemble_batch_cache` for batching decode requests
- ✅ Per-request KV cache on the `Request` object

What's missing: the prefill chunk and decode tokens go through **separate** `model()` calls. Your goal is to fuse them into **one** call.

---

## Hint 1: Visualize the Fused Input Tensor

Imagine step 5 has 3 active decode requests + 1 prefilling request getting an 8-token chunk. The fused input looks like:

```
token_budget = 16
decode requests: A (position 22), B (position 15), C (position 9)  → 3 tokens
prefill chunk:   D (tokens at positions 0..7)                       → 8 tokens
total: 11 tokens ≤ 16 ✓

T_max = max(1, 8) = 8   (longest row in the batch)

batch_tokens (B=4, T=8):
Row 0 (decode A): [ PAD, PAD, PAD, PAD, PAD, PAD, PAD, tok_A ]  ← 1 token, left-padded
Row 1 (decode B): [ PAD, PAD, PAD, PAD, PAD, PAD, PAD, tok_B ]  ← 1 token, left-padded
Row 2 (decode C): [ PAD, PAD, PAD, PAD, PAD, PAD, PAD, tok_C ]  ← 1 token, left-padded
Row 3 (prefill D):[ D_0, D_1, D_2, D_3, D_4, D_5, D_6, D_7 ]  ← 8 tokens, no padding

batch_positions (B=4, T=8):
Row 0: [ 0, 0, 0, 0, 0, 0, 0, 22 ]   ← position 22 for decode token
Row 1: [ 0, 0, 0, 0, 0, 0, 0, 15 ]
Row 2: [ 0, 0, 0, 0, 0, 0, 0,  9 ]
Row 3: [ 0, 1, 2, 3, 4, 5, 6,  7 ]   ← positions 0-7 for prefill chunk
```

**Each row can have a different number of "real" tokens.** Decode rows have T=1 (left-padded), the prefill row has T=chunk_size. They're all padded to the same `T_max` so they fit in a single tensor.

**Question to ask yourself:** Why left-pad instead of right-pad? Because model logits for the "next token" are always taken from the **last position** (`logits[:, -1, :]`). Left-padding keeps all real tokens right-aligned, so the last position is always meaningful.

---

## Hint 2: The KV Cache Assembly Gets Trickier

In your current code, `assemble_batch_cache` only handles decode requests (all with T=1). Now you need to assemble a cache that includes:

- **Decode requests:** have a populated `kv_cache` — same as before
- **Prefilling request (continuation chunk):** may have a partial `kv_cache` from earlier chunks, OR no cache at all (first chunk)

The key difference: when a prefill request has `past_k` of shape `(1, T_past, hs)` and its chunk has `T_chunk` tokens, the model will output `new_k` of shape `(1, T_past + T_chunk, hs)`. But decode requests output `new_k` of shape `(1, T_past_i + 1, hs)`. After the fused forward pass, you need to strip padding from **each row independently** during disassembly.

**Approach:** Extend `assemble_batch_cache` to accept a mixed list of requests where one request may contribute more than 1 token. You'll need to track how many tokens each row contributed so disassembly knows how to un-pad.

```python
def assemble_fused_batch(decode_reqs, prefill_req, chunk_size):
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
```

---

## Hint 3: Handling the First Prefill Chunk (No Past Cache)

When the prefilling request has no `kv_cache` yet (first chunk, `prefill_cursor == 0`), it cannot contribute to `assemble_batch_cache` because there's nothing to assemble. But it still needs to participate in the fused forward pass.

**Two approaches:**

### Approach A: Fake an empty cache
Create a zero-length "past" cache for the prefilling request: `(1, 0, hs)` tensors. Then `torch.cat([past_k, k], dim=1)` with `T_past=0` is a no-op. This lets the prefill request go through the same `past_k is not None` path as decode requests.

But wait — your `Head` uses the `past_k is not None` path for decode (no causal mask) and the `past_k is None` path for prefill (with causal mask). A prefill chunk with `T > 1` **needs** the causal mask. This is the core tension.

### Approach B: Separate masking per row (the correct solution)

Instead of branching on `past_k is None`, branch on `T`. If `T > 1` for a row, that row needs causal masking. If `T == 1`, it doesn't. In a fused batch, different rows have different `T` values (padded to the same `T_max`, but the real token counts differ).

**The practical shortcut for your nanoGPT:** always apply the causal mask for all rows, even decode rows. For a decode row with T=1, the causal mask `tril[:1, :1]` is just `[[1]]` — it has no effect. The causal mask only matters when `T > 1`. So you can safely use the causal-masked path (the `else` branch in `Head.forward()`) for the fused batch, and pass the assembled cache as `past_k`/`past_v`.

Wait — that means the prefilling request's first chunk (no cache) goes through the `else` branch too, which doesn't concatenate past_k. So for the first chunk, you need to assemble the cache with an empty `(1, 0, hs)` past.

**Simplest fix**: always pass `past_k`, and always apply causal masking. This unifies both code paths.

**Question to ask yourself:** What does `self.tril[:T, :T]` look like when `T = T_max = 8` but a decode row's real token is only at position 7? The pad positions are position 0-6, which get zeroed out anyway by the attention mask. So the causal mask is harmless for those rows.

---

## Hint 4: Modifying the Head to Support Fused Batches

Your current `Head.forward()` has two branches:
1. `past_k is not None` → decode path (no causal mask, just attend to full cache + new token)
2. `past_k is None` → prefill path (causal mask)

For the fused forward pass, you need a **single path** that handles both:

```python
def forward(self, x, past_k=None, past_v=None, attn_mask=None):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)

    if not self.training:
        # Always concatenate past (past may be empty: T_past = 0)
        if past_k is not None:
            k = torch.cat([past_k, k], dim=1)  # (B, T_past + T, hs)
            v = torch.cat([past_v, v], dim=1)

        T_full = k.shape[1]  # T_past + T

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T_full)

        # Build attention mask: causal within new tokens + attend to all past
        # For decode rows (T_new=1), this naturally collapses to "attend to everything"
        # For prefill rows (T_new>1), this enforces causality within the chunk
        causal_mask = torch.ones(T, T_full, device=x.device, dtype=torch.bool)
        # Mask future positions within the new tokens
        if T > 1:
            new_token_mask = self.tril[:T, :T]  # (T, T) lower-triangular
            causal_mask[:, -T:] = new_token_mask  # right side: causal within chunk
        # Expand for batch: (1, T, T_full) → (B, T, T_full)
        causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)

        # Combine with padding mask if provided
        if attn_mask is not None:
            # attn_mask: (B, 1, T_past) — True = valid past position
            # Extend with valid=True for the new T positions
            new_valid = torch.ones(B, 1, T, device=x.device, dtype=torch.bool)
            full_pad_mask = torch.cat([attn_mask, new_valid], dim=-1)  # (B, 1, T_full)
            causal_mask = causal_mask & full_pad_mask

        wei = wei.masked_fill(~causal_mask, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v

        return out, k, v
```

**The key insight:** this single code path handles decode (T=1, causal mask is trivially `[[True]]`) and prefill (T>1, causal mask is lower-triangular) uniformly. The `T > 1` guard on the causal mask is an optimization — for T=1 the mask is already all-True.

> **⚠️ This is the only model-level change.** Everything else is pure Python scheduling logic.

---

## Hint 5: The Fused Scheduler Loop

Your new `interleaved_generate` function follows this structure each step:

```
1. Scheduler decides: prefill_req, decode_reqs = scheduler.schedule(step)

2. Compute budget allocation:
   B_decode = len(decode_reqs)
   remaining_budget = token_budget - B_decode
   chunk_size = 0

3. If remaining_budget > 0 and prefill_req exists:
   tokens_left = len(prefill_req.prompt_tokens) - prefill_req.prefill_cursor
   chunk_size = min(remaining_budget, tokens_left)

4. Build the FUSED input:
   T_max = max(1, chunk_size) if both exist, else 1 or chunk_size
   For each decode request: 1 token, left-padded to T_max
   For the prefill request: chunk_size tokens (may also need left-padding if chunk_size < T_max)

5. Build the FUSED cache:
   all_reqs = decode_reqs + ([prefill_req] if prefill_req else [])
   Assemble cache from all of them (prefill req may have partial cache or empty)

6. Single model call:
   logits, _, new_kvs = model(batch_tokens, pos=batch_positions, past_kvs=past_kvs, attn_mask=attn_mask)

7. Disassemble:
   Split new_kvs back to per-request storage (accounting for padding)

8. Sample from logits[:, -1, :] for decode rows
   If prefill is now fully complete: sample first decode token, promote to active
   If prefill is partial: just update the request's cache and cursor

9. Check for completed decode requests → remove from active
```

**The critical difference from your current code:** step 6 is a SINGLE `model()` call instead of two.

---

## Hint 6: A Simpler Starting Point — Build Up Incrementally

Don't try to write the fused version from scratch. Instead, evolve your existing `continuous_batching_generate` in three stages:

### Stage 1: Extract the assembly logic into a helper
Pull the "build batch_tokens + batch_positions + past_kvs" code into a reusable function. Keep the two-call structure.

### Stage 2: Fuse when prefill_req has existing cache
When `prefill_req.kv_cache` is non-empty (continuation chunks), add it as an extra row in the decode batch. This is the easy case — the prefill row has a cache to assemble, just like decode rows. The only difference is it contributes `chunk_size` tokens instead of 1.

### Stage 3: Handle the first chunk (no cache)
For the very first chunk (`prefill_cursor == 0`, empty `kv_cache`), create dummy zero-length cache entries: `(1, 0, hs)` for each (layer, head). Then it participates in assembly just like the other rows.

```python
# Initialize empty cache for the prefill request's first chunk
if not prefill_req.kv_cache:
    for layer_idx in range(n_layer):
        for head_idx in range(n_head):
            prefill_req.kv_cache[(layer_idx, head_idx)] = (
                torch.empty(1, 0, head_size, device=device),
                torch.empty(1, 0, head_size, device=device),
            )
```

---

## Hint 7: Position Tensor Construction

Each row's positions depend on its type and history:

```python
positions = []
for req in decode_reqs:
    # Decode: position = total tokens so far - 1 (0-indexed)
    pos_val = len(req.tokens_so_far) - 1
    # Left-pad with 0s to T_max
    row = [0] * (T_max - 1) + [pos_val]
    positions.append(row)

if prefill_req:
    cursor = prefill_req.prefill_cursor  # value BEFORE advancing
    # Positions for the chunk: cursor, cursor+1, ..., cursor+chunk_size-1
    chunk_positions = list(range(cursor, cursor + chunk_size))
    # Left-pad if chunk_size < T_max
    padding = [0] * (T_max - chunk_size)
    positions.append(padding + chunk_positions)

batch_positions = torch.tensor(positions, device=device)  # (B, T_max)
```

**Constraint:** positions must not exceed `block_size - 1` (your position embedding table size). At nanoGPT scale with `block_size=32`, this limits total sequence length.

---

## Hint 8: Attention Mask for the Fused Batch

The attention mask now combines **two concerns**:

1. **Padding mask:** which cache positions are real vs zero-padded (same as before)
2. **Cross-request isolation:** decode row A should NOT attend to prefill row D's cache (they're different sequences batched together)

Your current architecture handles #2 automatically because the cache is batched per-row — row A's `past_k[i=0]` only contains row A's history. The padding mask handles #1. So the masking logic is the same as your current `assemble_batch_cache`, just extended to include the prefill request's cache.

The **new concern** is the causal mask within the prefill chunk (Hint 4). Each prefill token should only attend to:
- All past cache positions (everything before this chunk)
- Earlier tokens in the same chunk (causal within the chunk)
- NOT future tokens in the chunk

This is handled by the modified `Head.forward()` from Hint 4.

---

## Test Scenarios

### Test 1: Regression — identical output to two-call version
Run the same requests through both your current `continuous_batching_generate` and the new `interleaved_generate` with the same random seed. Outputs should be **identical**. This validates that fusing didn't change semantics.

### Test 2: Single decode + single prefill, both in one call
One active decode request + one 12-token prompt being prefilled with `token_budget=8`. Verify:
- Step 0: 1 decode token + 7 prefill tokens = 8 ≤ budget ✓
- Step 1: 1 decode token + 5 remaining prefill tokens = 6 ≤ budget ✓
- Step 2: prefill fully consumed, promoted to active. Now 2 decode requests.

### Test 3: Forward pass count
Instrument the model with a call counter. Verify that each step makes **exactly one** `model()` call (vs two in the current implementation). This is the whole point.

```python
call_count = 0
_orig_forward = model.forward
def counting_forward(*args, **kwargs):
    global call_count
    call_count += 1
    return _orig_forward(*args, **kwargs)
model.forward = counting_forward
```

### Test 4: Budget arithmetic
With `token_budget=10`, 5 active decode requests, and a 20-token prompt to prefill:
- Decode consumes 5 tokens → remaining budget = 5
- Prefill gets min(5, 20) = 5 tokens this step
- Total = 10 ≤ budget ✓
- Next step: 5 decode + min(5, 15) = 5 prefill = 10 ✓
- Takes 4 steps to fully prefill (5+5+5+5 = 20)

### Test 5: Edge case — budget fully consumed by decode
With `token_budget=4` and 4 active decode requests, `remaining_budget = 0`. No prefill work happens this step. The prefilling request waits. Verify it resumes next step when a decode request finishes and frees budget.

---

## Summary of Changes from Chunked Prefill Notebook

| Component | What Changes |
|-----------|-------------|
| `Head.forward()` | Unified code path with causal mask that works for both T=1 and T>1 (see Hint 4) |
| `assemble_batch_cache` → `assemble_fused_batch` | Extended to include the prefilling request alongside decode requests; handles variable token counts per row |
| `disassemble_batch_cache` | Updated to handle rows with different numbers of new tokens (1 for decode, chunk_size for prefill) |
| Main generate loop | **Single** `model()` call per step instead of two |
| `Request` dataclass | No changes |
| Model (GPTLanguageModel) | No changes (Head changes are internal to attention) |
| `continuous_batching_generate` | Replaced with `interleaved_generate` |

The key insight: **the model already supports variable-length rows in a batch.** You pad to `T_max` and mask out the padding. The only real model change is unifying the causal mask logic so it works for both decode rows (T=1) and prefill rows (T>1) in the same batch.

---

## Gotchas

1. **KV cache shape mismatch during assembly.** Decode requests have `T_past_i` tokens in their cache. The prefill request may have `T_past_prefill` tokens (from earlier chunks) or 0 tokens (first chunk). `assemble_batch_cache`'s padding needs to handle this varying `T_past` across rows.

2. **Causal mask interaction with padding.** When applying the causal mask within the prefill chunk, make sure the mask also zeroes out pad positions on the left. Otherwise the model attends to zero-valued pad tokens, which subtly corrupts the softmax distribution.

3. **Disassembly: different numbers of new tokens per row.** After the fused forward pass, decode row `i` produced `(T_past_i + 1)` cache entries. The prefill row produced `(T_past_prefill + chunk_size)` entries. Both need their left-padding stripped. Track `pad_lengths` per row during assembly and use them during disassembly.

4. **Logit extraction.** `logits[:, -1, :]` gives you the last-position logits for every row. For decode rows, this is the next token prediction (correct). For the prefill row, this is the logit after the last token in the chunk — which is only the "first generated token" if the chunk completes the prefill. If the chunk is partial, you don't sample from it; you just cache the KV and move on. Be careful not to sample from a partial prefill row.

5. **Empty decode batch.** When there are no active decode requests (only a prefilling request), the batch is just the prefill chunk. This degenerates to a standard prefill call. Make sure your code handles `len(decode_reqs) == 0` gracefully.
