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

Edited interleave-plan.md
Viewed interleave-plan.md:140-161

Yes, you have understood it perfectly!

To summarize exactly as you said:

1. **The First Mask (`causal_mask`)**: Enforces **causality**. It ensures that no token can look ahead at future tokens. It only cares about *position* relative to the current token.
2. **The Second Mask (`attn_mask` / `full_pad_mask`)**: Enforces **validity**. It ensures that the model ignores empty padding tokens that were added just to make the tensor shapes match up. It only cares about *whether a token is real or fake*.

By combining them with a logical AND (`&`), the final rule becomes: **"A token can only be attended to if it is BOTH in the past/present AND is a valid, real token."**

Viewed interleave-plan.md:137-160

Here is a step-by-step graphical visualization of how the tensors transform. 

Let's use a concrete example where:
* **`T_past = 4`** (We have 4 tokens previously processed in the KV cache)
* **`T = 3`** (We are processing a chunk of 3 new tokens in the current step)
* **`T_full = 7`** (`T_past + T`)
* **`B = 2`** (Batch size of 2)

I will use `██` for **True** (allowed to attend) and `░░` for **False** (masked out/prevented from attending).

---

### Step 1: Initialize all-True mask
```python
causal_mask = torch.ones(T, T_full, device=x.device, dtype=torch.bool)
```
We start with a shape of `(3, 7)`. Initially, every new token is allowed to look at every token in the entire `T_full` sequence.
```text
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [ ██ , ██ , ██ , ██    |   ██ , ██ , ██ ]
New Token 1   [ ██ , ██ , ██ , ██    |   ██ , ██ , ██ ]
New Token 2   [ ██ , ██ , ██ , ██    |   ██ , ██ , ██ ]
```

### Step 2: Grab the lower-triangular mask
```python
new_token_mask = self.tril[:T, :T] 
```
We grab a `(3, 3)` square matrix. This enforces the rule: "Token 0 can only see Token 0. Token 1 can see Tokens 0 & 1. Token 2 can see Tokens 0, 1, & 2."
```text
                 New Tokens (T)
                 0    1    2
New Token 0   [ ██ , ░░ , ░░ ]
New Token 1   [ ██ , ██ , ░░ ]
New Token 2   [ ██ , ██ , ██ ]
```

### Step 3: Apply the lower-triangular mask to the right side
```python
causal_mask[:, -T:] = new_token_mask 
```
We take the `(3, 3)` matrix from Step 2 and paste it over the rightmost 3 columns (`-T:`) of the `causal_mask` from Step 1. The past tokens remain untouched (all `True`).
```text
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [ ██ , ██ , ██ , ██    |   ██ , ░░ , ░░ ]
New Token 1   [ ██ , ██ , ██ , ██    |   ██ , ██ , ░░ ]
New Token 2   [ ██ , ██ , ██ , ██    |   ██ , ██ , ██ ]
```
*Notice how Token 0 can look at all past tokens + itself, but is blocked from looking at future tokens 1 and 2.*

### Step 4: Expand for the Batch Dimension
```python
causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)
```
Right now, `causal_mask` is just a 2D grid of shape `(T, T_full)`. But our attention weights `wei` have a shape of `(B, T, T_full)`. 

1. **`unsqueeze(0)`**: Adds a batch dimension to the front, changing the shape to `(1, 3, 7)`.
2. **`expand(B, -1, -1)`**: Broadcasts that single mask across the entire batch (size `B=2`), creating a final shape of `(2, 3, 7)`. It effectively duplicates the mask so every sequence in the batch uses it.

```text
Batch Item 0 (e.g., Sequence A)
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [ ██ , ██ , ██ , ██    |   ██ , ░░ , ░░ ]
New Token 1   [ ██ , ██ , ██ , ██    |   ██ , ██ , ░░ ]
New Token 2   [ ██ , ██ , ██ , ██    |   ██ , ██ , ██ ]

-----------------------------------------------------------

[ignoring loop detection]

Here is a step-by-step graphical visualization of how the tensors transform. 

Let's use a concrete example where:
* **`T_past = 4`** (We have 4 tokens previously processed in the KV cache)
* **`T = 3`** (We are processing a chunk of 3 new tokens in the current step)
* **`T_full = 7`** (`T_past + T`)
* **`B = 2`** (Batch size of 2)

I will use `T` for **True** (allowed to attend) and `F` for **False** (masked out/prevented from attending).

---

### Step 1: Initialize all-True mask
```python
causal_mask = torch.ones(T, T_full, device=x.device, dtype=torch.bool)
```
We start with a shape of `(3, 7)`. Initially, every new token is allowed to look at every token in the entire `T_full` sequence.
```text
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [  T ,  T ,  T ,  T    |    T ,  T ,  T ]
New Token 1   [  T ,  T ,  T ,  T    |    T ,  T ,  T ]
New Token 2   [  T ,  T ,  T ,  T    |    T ,  T ,  T ]
```

### Step 2: Grab the lower-triangular mask
```python
new_token_mask = self.tril[:T, :T] 
```
We grab a `(3, 3)` square matrix. This enforces the rule: "Token 0 can only see Token 0. Token 1 can see Tokens 0 & 1. Token 2 can see Tokens 0, 1, & 2."
```text
                 New Tokens (T)
                 0    1    2
New Token 0   [  T ,  F ,  F ]
New Token 1   [  T ,  T ,  F ]
New Token 2   [  T ,  T ,  T ]
```

### Step 3: Apply the lower-triangular mask to the right side
```python
causal_mask[:, -T:] = new_token_mask 
```
We take the `(3, 3)` matrix from Step 2 and paste it over the rightmost 3 columns (`-T:`) of the `causal_mask` from Step 1. The past tokens remain untouched (all `True`).
```text
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [  T ,  T ,  T ,  T    |    T ,  F ,  F ]
New Token 1   [  T ,  T ,  T ,  T    |    T ,  T ,  F ]
New Token 2   [  T ,  T ,  T ,  T    |    T ,  T ,  T ]
```
*Notice how Token 0 can look at all past tokens + itself, but is blocked from looking at future tokens 1 and 2.*

### Step 4: Expand for the Batch Dimension
```python
causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)
```
Right now, `causal_mask` is just a 2D grid of shape `(T, T_full)`. But our attention weights `wei` have a shape of `(B, T, T_full)`. 

1. **`unsqueeze(0)`**: Adds a batch dimension to the front, changing the shape to `(1, 3, 7)`.
2. **`expand(B, -1, -1)`**: Broadcasts that single mask across the entire batch (size `B=2`), creating a final shape of `(2, 3, 7)`. It effectively duplicates the mask so every sequence in the batch uses it.

```text
Batch Item 0 (e.g., Sequence A)
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [  T ,  T ,  T ,  T    |    T ,  F ,  F ]
New Token 1   [  T ,  T ,  T ,  T    |    T ,  T ,  F ]
New Token 2   [  T ,  T ,  T ,  T    |    T ,  T ,  T ]

-----------------------------------------------------------
Batch Item 1 (e.g., Sequence B)
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [  T ,  T ,  T ,  T    |    T ,  F ,  F ]
New Token 1   [  T ,  T ,  T ,  T    |    T ,  T ,  F ]
New Token 2   [  T ,  T ,  T ,  T    |    T ,  T ,  T ]
```

Later on, when this `causal_mask` is combined with the `attn_mask` (which handles padding across the batch), any positions corresponding to padding in Batch Item 0 or 1 will also be set to `F`.

=============================================================


Edited interleave-plan.md
Viewed interleave-plan.md:143-169

[ignoring loop detection]

Here is a graphical explanation of how the padding mask (`attn_mask`) is integrated with the `causal_mask`. 

This step handles the fact that in a batch, some sequences might be shorter than others and have been padded with empty tokens in their KV cache history. The model must be prevented from paying attention to these meaningless padded tokens.

Let's build on the previous example:
* **`T_past = 4`**
* **`T = 3`**
* **`B = 2`**

Imagine our batch has two sequences with different past histories:
* **Sequence A** (Batch Item 0): Only has **2 real tokens** in the past cache, and **2 padding tokens** (left-padded).
* **Sequence B** (Batch Item 1): Has **4 real tokens** in the past cache (no padding).

### Step 1: The Input `attn_mask`
The `attn_mask` passed into the forward pass only covers the *past* tokens (`T_past`). It has shape `(B, 1, T_past)`.
```text
                 Past Tokens (T_past)
                 0    1    2    3    
Sequence A    [  F ,  F ,  T ,  T ]   <-- Positions 0 and 1 are padding!
Sequence B    [  T ,  T ,  T ,  T ]   <-- All valid tokens
```

### Step 2: Create `new_valid`
```python
new_valid = torch.ones(B, 1, T, device=x.device, dtype=torch.bool)
```
The code assumes all new tokens being passed in are valid for the purpose of this specific mask step. We create an all-`True` mask of shape `(2, 1, 3)` for the new chunk.
```text
                 New Tokens (T)
                 0    1    2
Sequence A    [  T ,  T ,  T ]
Sequence B    [  T ,  T ,  T ]
```

### Step 3: Concatenate to make `full_pad_mask`
```python
full_pad_mask = torch.cat([attn_mask, new_valid], dim=-1)  # (B, 1, T_full)
```
We stick them together horizontally to get a mask that covers the entire `T_full` (7 tokens).
```text
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
Sequence A    [  F ,  F ,  T ,  T    |    T ,  T ,  T ]
Sequence B    [  T ,  T ,  T ,  T    |    T ,  T ,  T ]
```

### Step 4: Logical AND with the `causal_mask`
```python
causal_mask = causal_mask & full_pad_mask
```
Now, we overlay (Logical AND) this padding mask onto the `causal_mask` we built earlier. 

* If a position is `T` in both, it stays `T`. 
* If a position is `F` in either mask, it becomes `F`.

Because `full_pad_mask` has a dimension of `1` in the middle `(B, 1, T_full)`, PyTorch automatically broadcasts (copies) it downwards across all `T` rows of the causal mask.

Here is the final result:

```text
Batch Item 0 (Sequence A - Has Padding)
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [  F ,  F ,  T ,  T    |    T ,  F ,  F ]
New Token 1   [  F ,  F ,  T ,  T    |    T ,  T ,  F ]
New Token 2   [  F ,  F ,  T ,  T    |    T ,  T ,  T ]
                ^^^^^^^^
           (Padding is completely blocked from attention)

-----------------------------------------------------------
Batch Item 1 (Sequence B - No Padding)
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [  T ,  T ,  T ,  T    |    T ,  F ,  F ]
New Token 1   [  T ,  T ,  T ,  T    |    T ,  T ,  F ]
New Token 2   [  T ,  T ,  T ,  T    |    T ,  T ,  T ]
                ^^^^^^^^
           (All past tokens remain attendable)
```

**Why this matters:** Without this step, `New Token 0` in Sequence A would average in the values of the empty padding tokens at indices 0 and 1, which would completely corrupt the model's predictions.

*(Note: If you have left-padded tokens in the new `T` tokens themselves during a fused decode-prefill batch, as mentioned in "Gotchas" #2 of your plan, you would also need to ensure those specific left-padded positions in the new `T` chunk are set to `False` in the mask, rather than using an all-`True` `new_valid` mask.)*

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

## Errors:

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_4739/1371704712.py in <cell line: 0>()
     11 
     12     xb, yb = get_batch('train')
---> 13     logits, loss, _ = model(xb, yb)  # _ discards the cache during training
     14     optimizer.zero_grad(set_to_none=True)
     15     loss.backward()

/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1774             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1775         else:
-> 1776             return self._call_impl(*args, **kwargs)
   1777 
   1778     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1785                 or _global_backward_pre_hooks or _global_backward_hooks
   1786                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1787             return forward_call(*args, **kwargs)
   1788 
   1789         result = None

/tmp/ipykernel_4739/1855062281.py in forward(self, idx, targets, pos, past_kvs, attn_mask)
    222         new_kvs = []
...
--> 127             out, nk, nv = h(x, pk, pv, attn_mask=attn_mask)
    128             outputs.append(out)
    129             new_kvs.append((nk, nv))

TypeError: cannot unpack non-iterable NoneType object

Reason: We didn't include training path in forward method

---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_4739/1605021355.py in <cell line: 0>()
     18     Request(id=2, prompt_tokens=encode("KING HENRY:\n"), max_new_tokens=10),
     19 ]
---> 20 s_two = scheduled_generate(model, reqs_two_call, policy="fcfs",
     21                            token_budget=16, max_kv_tokens=256)
     22 

/tmp/ipykernel_4739/1937282480.py in scheduled_generate(model, requests, policy, token_budget, max_kv_tokens)
    288                 idx_next = torch.multinomial(probs, num_samples=1)
    289 
--> 290                 disassemble_batch_cache(scheduler.active, new_kvs, pad_lengths)
    291 
    292                 for i, req in enumerate(decode_reqs):

/tmp/ipykernel_4739/1937282480.py in disassemble_batch_cache(requests, new_kvs, pad_lengths)
    131         for head_idx, (batched_k, batched_v) in enumerate(block_kv):
    132             for i, req in enumerate(requests):
--> 133                 pad = pad_lengths[i]
    134                 req.kv_cache[(layer_idx, head_idx)] = (
    135                     batched_k[i : i + 1, pad:, :],      # (1, T_i + 1, hs)

IndexError: list index out of range

Reason: 

pad_lengths was built from decode_reqs (via build_tok_pos_kv(decode_reqs)), but you're passing scheduler.active which may be longer — because the prefill block above just promoted a request into scheduler.active via scheduler.promote(prefill_req).

Fix: Change scheduler.active to decode_reqs:

# BEFORE (bug):
disassemble_batch_cache(scheduler.active, new_kvs, pad_lengths)
# AFTER (fix):
disassemble_batch_cache(decode_reqs, new_kvs, pad_lengths)

AcceleratorError                          Traceback (most recent call last)
/tmp/ipykernel_4739/2338838789.py in <cell line: 0>()
     26 print(f"  Expected prefill chunks: ~{len(prompt) // 8} steps\n")
     27 
---> 28 s = interleaved_generate(model, reqs, policy="fcfs",
     29                          token_budget=8, max_kv_tokens=256)
     30 

/tmp/ipykernel_4739/283559998.py in interleaved_generate(model, requests, policy, token_budget, max_kv_tokens)
     64 
     65                 for i, req in enumerate(decode_reqs):
---> 66                     req.generated_tokens.append(idx_next[i].item())
     67                     req._last_token = idx_next[i : i + 1]
     68                     if req.is_done:

AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Fix: 

# BEFORE (bug):
prompt = prompt[:block_size]  # 32 tokens, no room for decode!
# AFTER (fix):
prompt = prompt[:block_size - 5]  # 27 tokens + 5 generated = 32 ≤ block_size
