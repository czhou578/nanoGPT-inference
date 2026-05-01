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

## Hint 2: The KV cache is the hard part

Your `Head.key_cache` has shape `(B, T, hs)` — one contiguous tensor for the whole batch. With continuous batching, different requests have different sequence lengths. You have two options to think through:

1. **Per-request caches** (dict keyed by request ID) — each request gets its own `(1, T_i, hs)` tensor, and you `torch.cat` them along dim=0 before the attention math. What do you do when the `T_i` values differ across requests?
2. **Padded batch cache** — pre-allocate `(max_batch, max_seq_len, hs)` and track how many tokens each slot has actually used. How do you mask out the padding in attention?

Think about which approach is simpler to implement and which matches what vLLM does (hint: vLLM does neither — it uses paged blocks — but for a first pass, one of the above is fine).

---

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
