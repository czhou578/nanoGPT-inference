# Continuous Batching Hints for NanoGPT

## The Core Problem to Solve

Your current `generate_with_cache` serves **one request at a time**. `B` in your batch dim is always 1 during generation. Continuous batching means multiple independent requests share that batch dimension, but they can **arrive and finish at different times**.

---

## Hint 1: What is a "request"?

Right now you don't have the concept of a request — think about what state each in-flight generation needs to carry:
- Its token IDs generated so far
- How many tokens it still needs to produce (`max_new_tokens`)
- Its own KV cache entries (currently baked into `Head` as a single tensor for the whole batch)

Ask yourself: **if request 0 finishes after 20 tokens but request 1 needs 100, what happens to row 0 in the batch?**

---

Row 0 is evicted from the batch, and request 1 continues to generate tokens.

## Hint 2: The KV cache is the hard part

Your `Head.key_cache` has shape `(B, T, hs)` — one contiguous tensor for the whole batch. With continuous batching, different requests have different sequence lengths. You have two options to think through:

1. **Per-request caches** (dict keyed by request ID) — each request gets its own `(1, T_i, hs)` tensor, and you `torch.cat` them along dim=0 before the attention math. What do you do when the `T_i` values differ across requests?
2. **Padded batch cache** — pre-allocate `(max_batch, max_seq_len, hs)` and track how many tokens each slot has actually used. How do you mask out the padding in attention?

Think about which approach is simpler to implement and which matches what vLLM does (hint: vLLM does neither — it uses paged blocks — but for a first pass, one of the above is fine).

---

KV Cache needs to be per batch, not per head. 

## Hint 3: The scheduler loop replaces the `for step in range(max_new_tokens)` loop

Your current loop runs a fixed number of steps for one request. The continuous batching loop looks more like:

```
while there are active requests OR the waiting queue is non-empty:
    1. Check the waiting queue — can any new requests join the batch?
    2. Build the input tensor from ALL active requests (each contributes 1 token)
    3. Forward pass → get logits for all active requests at once
    4. Sample next token for each request
    5. Check: did any request hit its max_new_tokens? → remove it, emit its result
    6. Go to 1
```

The key insight: **step 1 and step 5 happen every iteration**, not just at the start and end. Requests flow in and out continuously.

---

Could we use torch.zeroes? What's the difference?
Is this only for one request?
- No, for entire batch 
What does ~full mask mean?

 - python's bitwise not operator. On a bool tensor, it flips every value. 

 full_mask  = [False, False, True, True, True]
~full_mask = [True,  True,  False, False, False]
everywhere the mask says False (= padding), replace the attention score with -inf."


Notes:

In standard batching, all computations happen on a massive, unified tensor. A standard attention head expects its cached Keys (K) and Values (V) to arrive perfectly stacked in a block shaped like: [Batch Size (B), Sequence Length (T), Head Size (hs)]

However, by moving to a request-level continuous batching system (Hint 1 & 2 in your notebook), we scattered our KV caches. Now:

The KV Cache is fragmented: Instead of the model holding one huge block, each individual request holds a bunch of tiny blocks (1, T, hs) in a dictionary.
Asymmetrical sequences: Because requests arrive at different times and have different prompt lengths, their cached sequences are all different lengths (e.g., Request A has T=50, Request B has T=15).
The triple for loop's job is to act as a "packing plant": It gathers all those individual fragments, lines them up, pads the short ones so they are uniform, and squishes them perfectly back into the exact [B, T_max, hs] multidimensional block the model expects.

torch.cat expects list of tensors, not list of python ints.

## Steps

1. Check if any new requests have arrived
2. Grab new requests if any
3. Prefill new requests and get logits / predictions for first token
4. Store new kv in request's kv cache
5. Forward pass to get next token idx
6. Set status of request to active or done if done
7. Assemble batch from active requests
8. Get logits for next token
9. Sample next token for each request
10. Store new kv in request's kv cache
11. Set status of request to active or done if done
12. Repeat until all requests are done

==================================

for i, req in enumerate(active_requests):
    req.generated_tokens.append(idx_next[i].item())
    req._last_token = idx_next[i : i + 1]

## Hint 4: Think about what changes in `forward()`

Right now `forward()` takes `idx` of shape `(B, T)` where all B sequences have the same T. In continuous batching during decode, every active request contributes exactly 1 token, so `T=1` still works — but `B` changes dynamically as requests arrive/finish. The question is: **does the model care if B changes between steps?** (Hint: not really, since there are no learned parameters that depend on B.)

---

## Hint 5: Simulate arrivals, don't use an actual server

You don't need HTTP or async. Just create a list of "requests" with different prompts and different `max_new_tokens`, and have them "arrive" at different steps (e.g., request 0 arrives at step 0, request 1 at step 3, request 2 at step 5). The scheduler pulls from this queue.

---

## Suggested order of attack

1. **Define a `Request` dataclass** — prompt tokens, max_new_tokens, tokens generated so far, status
2. **Move KV cache ownership from `Head` to per-request storage** — this is the biggest refactor
3. **Write the scheduler loop** — the `while` loop described above
4. **Assemble/disassemble the batch** each step — gather active requests into a batch tensor, scatter results back
5. **Test**: submit 3 requests with different prompt lengths and different max_new_tokens, verify they all complete correctly and the outputs match single-request generation



==========================================

Edited nanogpt-continuous-batching.py
Edited nanogpt-continuous-batching.py
Edited nanogpt-continuous-batching.py
Edited nanogpt-continuous-batching.py
Viewed nanogpt-continuous-batching.py:102-134

You're looking at the `Head.forward` decode path in the notebook. Let me walk through the complete, working version from the notebook step by step, since that's the one with the attention mask logic:

```python
def forward(self, x, past_k=None, past_v=None, attn_mask=None):
    B, T, C = x.shape
    k = self.key(x)    # (B, T, hs)
    q = self.query(x)  # (B, T, hs)
    v = self.value(x)  # (B, T, hs)

    if not self.training:
        if past_k is not None:
            # ── Decode step ──
            k = torch.cat([past_k, k], dim=1)   # (B, T_past + T, hs)
            v = torch.cat([past_v, v], dim=1)

            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

            if attn_mask is not None:
                new_valid = torch.ones(B, 1, T, device=wei.device, dtype=torch.bool)
                full_mask = torch.cat([attn_mask, new_valid], dim=-1)
                wei = wei.masked_fill(~full_mask, float('-inf'))

            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
```

### Why do we even need an attention mask here?

In single-request KV cache generation, you don't need one. Every position in the cache is a real token. But in **continuous batching**, different requests have different sequence lengths, so `assemble_batch_cache` **left-pads** shorter caches with zeros to match the longest one:

```
Request 0 cache (length 10): [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]
Request 1 cache (length  7): [0,  0,  0,  t0, t1, t2, t3, t4, t5, t6]
                               ^^^^^^^^^^
                               fake padding — not real tokens!
```

Without a mask, the query would attend to those zero-padded positions as if they were real tokens, corrupting the output.

### The mask logic, step by step:

**Step 1:** `assemble_batch_cache` builds `attn_mask` of shape `(B, 1, T_past)` where `True` = real token, `False` = padding:
```
Request 0: [True, True, True, True, True, True, True, True, True, True]
Request 1: [False, False, False, True, True, True, True, True, True, True]
```

**Step 2:** After `torch.cat([past_k, k])`, the cache grew by 1 (the new token). The mask is now too short — it covers `T_past` positions but the keys have `T_past + 1`. We need to extend it:
```python
new_valid = torch.ones(B, 1, T, device=wei.device, dtype=torch.bool)  # T=1 during decode
full_mask = torch.cat([attn_mask, new_valid], dim=-1)  # (B, 1, T_past + 1)
```
This appends a `True` for the new token (which is always real, never padding):
```
Request 0: [True, True, True, True, True, True, True, True, True, True, True]
Request 1: [False, False, False, True, True, True, True, True, True, True, True]
```

**Step 3:** Apply the mask to the attention scores:
```python
wei = wei.masked_fill(~full_mask, float('-inf'))
```
Everywhere the mask is `False` (padding), the attention score becomes `-inf`. After softmax, `-inf` becomes `0.0`, meaning the query completely ignores those padding positions.

### Why is there no mask in the `past_k is None` (prefill) branch?

During prefill, there's no batching of different requests together — each request is prefilled individually with `B=1`. There's no padding, so the regular causal `tril` mask is sufficient.