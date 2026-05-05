# Chunked Prefill — Implementation Plan & Hints

## The Problem You're Solving

Look at what happens in your current `continuous_batching_generate` when a new request arrives:

```
step 3: req 2 arrives with prompt_tokens = 12 tokens
        → you run model(prompt) with ALL 12 tokens in one shot
        → decode batch (req0, req1) gets ZERO forward passes this step
```

The prefill hogs the entire step. With a 12-token prompt, that's fine. But imagine a prompt with 1000 tokens — active decode requests would stall for the entire duration of that prefill. In a real server, this causes **latency spikes** for every in-flight request.

**Chunked prefill** splits long prompts into pieces and processes them alongside decode tokens in the same forward pass, so decode requests never get starved.

---

## Hint 1: Add a Token Budget

The scheduler needs a new knob: `token_budget` — the **maximum total tokens processed per forward pass**.

Each step, you have two types of work competing for that budget:
- **Decode tokens**: 1 token per active request (these are cheap, high priority)
- **Prefill tokens**: remaining chunk of a partially-prefilled request

Think about what order to prioritize. In vLLM's implementation, decode tokens always get first dibs on the budget. Why? A decode request already has VRAM allocated for its KV cache. Starving it wastes that memory.

**Question to ask yourself:** If you have 5 active decode requests and `token_budget = 16`, how many tokens are left for prefill work?

Answer: 11?

---

## Hint 2: Track Prefill Progress on the Request

Your current `Request` dataclass has no concept of "partially prefilled." It goes straight from `waiting` → `active`. You need a way to know:
- How many prompt tokens have already been processed?
- How many are left?

Think about adding a field like `prefill_cursor` — an integer pointing to where you are in `prompt_tokens`. A request is fully prefilled when `prefill_cursor == len(prompt_tokens)`.

This also means you need a new status between `waiting` and `active`. Something like `prefilling` — the request has started processing its prompt but isn't done yet.

**The states become:** `waiting` → `prefilling` → `active` → `done`

---

## Hint 3: Chunked Prefill Changes the Forward Pass Shape

In your current code, prefill and decode are **separate** model calls:
```
# Current approach:
model(full_prompt)           # prefill — standalone call, T = prompt_len
model(batch_of_1_tokens)     # decode  — separate call, T = 1 per request
```

With chunked prefill, you want to run a **single forward pass** that processes both prefill chunks and decode tokens together. Think about what the input tensor looks like:

```
token_budget = 16
active decode requests: 3 (uses 3 tokens)
remaining budget: 13 tokens for prefill
request in prefilling state: prompt has 25 tokens, cursor at 0

This step:  process prompt[0:13]  → 13 prefill tokens
Next step:  process prompt[13:25] → 12 prefill tokens (+ 3 decode = 15 ≤ 16 ✓)
Step after: request is fully prefilled, sample first token, joins decode batch
```

**Key insight:** Each "row" in the batch can have a different number of tokens. One row might be a 13-token prefill chunk, another might be a 1-token decode. You already handle variable lengths with padding in `assemble_batch_cache` — the same idea applies here.

---

## Hint 4: Position Indices Matter

When you send `prompt[0:13]` in step N and `prompt[13:25]` in step N+1, the positions must be correct:
- First chunk: positions `[0, 1, 2, ..., 12]`
- Second chunk: positions `[13, 14, 15, ..., 24]`

For decode tokens, the position is still `len(tokens_so_far) - 1`.

Your model's `forward()` already accepts a `pos` argument. You'll need to construct a `pos` tensor where each row has the right positions for its chunk/token.

**Constraint to remember:** your position embedding table has `block_size` entries. Make sure no position exceeds `block_size - 1`.

---

Here is the critical part: If you don't explicitly provide the pos tensor, your nanoGPT model's forward function defaults to torch.arange(T). Since T=12 for this chunk, it would assign positions [0, 1, 2, ..., 11] to these tokens!

That would ruin the model's understanding. It would think P13 is the first word of the sentence.

# Row 0: Decode request A (sending its 18th token, T=1) left padded with 0s
# Row 1: Chunk request B (sending tokens 13 to 24, T=12)

# input tokens (batch_tokens):
[[ PAD_TOK, PAD_TOK, PAD_TOK, ..., PAD_TOK, Decode_Tok_A ],
 [ B_P13,   B_P14,   B_P15,   ..., B_P23,   B_P24      ]]

# pos tensor (batch_positions):
[[ 0,       0,       0,       ..., 0,       17         ],  # Row A: position 17 for the decode token
 [ 13,      14,      15,      ..., 23,      24         ]]  # Row B: positions 13-24 for the chunk




## Hint 5: KV Cache for Partial Prefills

After processing `prompt[0:13]`, the model returns KV cache entries for those 13 positions. You need to **store these on the request** even though it's not done prefilling yet.

On the next step, when you process `prompt[13:25]`, you must pass the stored partial cache as `past_kvs` so the attention can see the earlier tokens.

This is exactly the same pattern as normal decode — `torch.cat([past_k, new_k], dim=1)` — except instead of appending 1 new token, you're appending a chunk of 12 tokens.

**Nothing in your `Head` needs to change.** The prefill-with-past path and the decode-with-past path use the same code branch. The only difference is `T > 1` vs `T = 1`.

---

## Hint 6: The New Scheduler Loop

Your new `chunked_prefill_generate` should follow this structure each step:

```
1. Count decode tokens needed:  B_active = len(active_requests)
2. Remaining prefill budget:    prefill_budget = token_budget - B_active
3. If prefill_budget > 0 and there's a request in prefilling state:
       Compute chunk_size = min(prefill_budget, tokens_remaining_for_this_request)
       Process the chunk
4. Build the combined forward pass:
       - Decode requests contribute 1 token each
       - Prefilling request contributes chunk_size tokens
5. Run model(combined_input, pos=combined_positions, past_kvs=..., attn_mask=...)
6. Extract logits, sample next tokens for decode requests
7. If prefilling request is now fully prefilled, sample its first token too
8. Disassemble caches back to per-request storage
```

**The tricky part:** building the combined input tensor. Decode requests have 1 token, the prefilling request has `chunk_size` tokens. You need to pad them into a single `(B, T_max)` tensor where `T_max = chunk_size` (decode tokens get left-padded to the same length).

**Alternative simpler approach:** process the prefill chunk as a separate row in the batch but in the **same** forward pass (which your model already supports via batching).

---

## Hint 7: When Does a Prefilling Request Join the Decode Batch?

A request transitions from `prefilling` → `active` when:
1. All prompt tokens have been processed (`prefill_cursor == len(prompt_tokens)`)
2. You've sampled the first generated token from the final chunk's logits

After that, it's just a regular decode request like before.

**Edge case:** What if the prompt fits entirely within the budget? Then it's identical to your current full prefill — process all tokens in one step, sample, and go straight to `active`. Your chunked prefill code should handle this as a natural special case (chunk_size == len(prompt_tokens)).

---

## Hint 8: Attention Mask Complexity

With mixed prefill + decode in one forward pass, the attention mask gets more complex:

- **Decode tokens** (T=1): attend to their full past cache + themselves. No causal masking needed (it's a single token).
- **Prefill chunk** (T>1): needs **causal masking within the chunk** (token 3 shouldn't see tokens 4+), AND should attend to the past cache if it's a continuation chunk.

Your current `Head` already handles both cases — the `past_k is not None` branch for decode, and the causal-mask branch for prefill. But if you're mixing them in one batch, each row needs its own masking behavior.

**Simplest approach:** process the prefill chunk and decode tokens as separate model calls within the same step (not truly "fused"). This loses some GPU efficiency but gets the scheduling right. Fusing them into a single forward pass is an optimization you can add later.

---

## Test Scenarios

### Test 1: Short prompts (no chunking needed)
Use prompts shorter than `token_budget`. Behavior should be identical to your current continuous batching. This is your regression test.

### Test 2: Long prompt that requires chunking
Create a request with a prompt of 20 tokens and `token_budget = 8`. It should take 3 steps to fully prefill (8 + 8 + 4 tokens), with decode requests getting service on every step.

### Test 3: Verify decode latency isn't spiked
With chunked prefill, active decode requests should get a forward pass **every step**, even while a large prefill is in progress. Print per-step timing to confirm no single step takes dramatically longer than others.

### Test 4: Mixed arrivals
Multiple requests arriving at different times with different prompt lengths. Some need chunking, some don't. All should complete with correct output and cache shapes.

---

## Summary of Changes from Continuous Batching

| Component | What Changes |
|-----------|-------------|
| `Request` dataclass | Add `prefill_cursor` field and `prefilling` status |
| Scheduler loop | Replace "prefill all at once" with "budget-aware chunked prefill" |
| Batch assembly | Handle rows with different token counts (chunk vs single decode token) |
| `assemble_batch_cache` | May need to handle requests with partial caches |
| `Head` / model | **Nothing** — the forward pass already supports variable-length inputs with past caches |
