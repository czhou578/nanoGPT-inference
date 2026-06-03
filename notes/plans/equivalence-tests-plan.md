# Correctness Equivalence Tests — Full Inference Engine

These tests prove that every optimization layer in your NanoGPT inference engine produces **identical logits/outputs** to the simplest baseline. Each test isolates one optimization and compares it against its "dumb but obviously correct" counterpart.

---

## Test 1: Full-Recompute Logits == KV-Cached Incremental Logits

### What it catches
KV cache threading bugs, position embedding errors, causal mask issues.

### How to implement
1. Pick a prompt (8–12 tokens from `val_data`). Generate ~5 tokens greedily (argmax).
2. **Cached path**: Prefill prompt → `past_kvs`. Then loop: feed `(1,1)` token with `pos=[[cache_len]]` and `past_kvs`, collect `logits[0, -1, :]` each step.
3. **Recompute path**: Concatenate `prompt + all_generated`. Feed entire sequence in one `forward()` (no cache). Extract `logits[0, t, :]` at positions `len(prompt)-1, len(prompt), ...`.
4. Assert `torch.allclose(cached_logits[i], recompute_logits[i], atol=1e-5)` at every step.

> [!TIP]
> **Hint — position indices**: Cached path uses `pos = [[cache_len]]` per step. Recompute uses `arange(0, total_len)`. The recompute logit for "what comes after position t" lives at `logits[0, t, :]`, not `logits[0, t+1, :]`.

### Where the code lives
Uses `GPTLanguageModel.forward()` directly — no benchmark module needed. Import the model from whichever `nanogpt-*.py` file you choose as the canonical one, or write a small helper that creates + trains a tiny model.

---

## Test 2: Unbatched Output == Continuously Batched Output

### What it catches
Left-padding bugs, attention mask errors, `_stack_kvs`/`_unstack_kvs` corruption, batched position computation mistakes.

### How to implement
1. Create 3–4 requests with the **same prompt** and **same `max_new_tokens`**, same seed.
2. **Unbatched**: Run each request through [run_sequential_single_request](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/single_req_cont_batching.py#L227) one at a time. Collect each request's `generated_tokens`.
3. **Batched**: Run the same workload through [run_continuous_batching](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/single_req_cont_batching.py#L306) with `max_batch_size >= num_requests` and `arrival_gap=0`.
4. Assert `sequential_tokens[i] == batched_tokens[i]` for every request.

> [!IMPORTANT]
> **Hint — RNG alignment**: Both paths must consume random draws in the **exact same order**. The sequential runner processes requests `[0, 1, 2, ...]` in order. The continuous batcher prefills them all at step 0, then decodes them together — but `_sample_next_token` is called with the **same shared generator** in the same logical order. If the generator sees draws in a different order, tokens will diverge even if the math is correct. **Fix**: Use `temperature=0` (argmax) to eliminate RNG dependence, OR seed a **per-request** generator so ordering doesn't matter.

> [!WARNING]
> **Hint — uniform lengths required**: `run_continuous_batching` asserts aligned cache lengths. Use `make_uniform_workload` with identical `prompt_len` and `max_new_tokens` for all requests. If you want to test mixed lengths, you'll need the paged attention path (Test 3).

### Key files
- [single_req_cont_batching.py](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/single_req_cont_batching.py) — `_stack_kvs`, `_unstack_kvs`, both runners

---

## Test 3: Contiguous KV Output == Paged KV Output

### What it catches
`write_kv_to_pool` / `gather_kv_from_pool` index math, block boundary bugs, `_assemble_paged_kvs` left-padding errors, block allocation logic.

### How to implement
1. Create a workload with `make_uniform_workload` from [paged_attention.py](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/paged_attention.py) (3–4 requests, same prompt length).
2. **Contiguous**: Run [run_contiguous_kv_policy](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/paged_attention.py#L432) → collect `generated_tokens` per request.
3. **Paged**: Run [run_paged_kv_policy](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/paged_attention.py#L517) with the same workload, seed, temperature → collect `generated_tokens`.
4. Assert token-for-token equality.

> [!TIP]
> **Hint — block boundary stress test**: Use `make_block_boundary_workload` which creates prompts of length 3, 4, 5, 7, 8, 9, ... — specifically chosen to land on and off `page_block_size` boundaries. This catches off-by-one errors in `num_filled_slots // block_size` vs `num_filled_slots % block_size`.

> [!IMPORTANT]
> **Hint — use `temperature=0`** for the deterministic version. Then optionally run with `temperature=1.0` and same seed to verify RNG alignment matches too (but this is fragile — argmax is the real correctness test).

### Key files
- [paged_attention.py](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/paged_attention.py) — `_gather_paged_kv`, `_write_kvs_to_pool`, `_assemble_paged_kvs`, both runners

---

## Test 4: Prefix-Cached Output == Normal Prefill Output

### What it catches
`_load_cached_prefix` / `_commit_completed_blocks` bugs, hash chaining errors, stale KV data in cache, incorrect `prefill_cursor` advancement.

### How to implement
1. Create a `make_shared_prefix_workload` with 4 requests sharing a 16-token prefix + 4-token unique suffix.
2. **No cache**: Run [run_prefix_cache_policy](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/prefix_caching.py#L294) with `use_prefix_cache=False` → collect tokens.
3. **With cache**: Run the same function with `use_prefix_cache=True` → collect tokens.
4. Assert `no_cache_tokens[i] == cached_tokens[i]` for every request.

> [!TIP]
> **Hint — the first request can't hit cache**: Request 0 always does a full prefill (nothing cached yet). Requests 1–N should hit the shared prefix blocks. So you need ≥2 requests to actually exercise the cache-load path. Verify `cached_prefix_tokens > 0` for requests after the first.

> [!WARNING]
> **Hint — suffix must be non-empty**: The benchmark's `_prefill_remaining` raises if `prefill_cursor >= len(prompt_tokens)`. Set `unique_suffix_len >= 1` so there's always an uncached tail to prefill. This is realistic — in production, the last partial block is never cached.

### Key files
- [prefix_caching.py](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/prefix_caching.py) — `_load_cached_prefix`, `_commit_completed_blocks`, `_slice_kvs`, `_concat_kvs`

---

## Test 5: Speculative Decoding Greedy == Autoregressive Greedy

### What it catches
Accept/reject logic errors, KV cache trim bugs, `prev_token` / `prev_prev_token` tracking bugs (trigram-specific), bonus token logic.

### How to implement
1. You need **greedy (argmax) versions** of both paths. Two options:
   - **(a) Recommended**: Write a thin wrapper that replaces `torch.multinomial` with `torch.argmax` in `_sample_next_token` and `_accept_reject`. Simplest: add a `greedy=True` kwarg.
   - **(b) Quick hack**: Use `temperature=0.001` (near-greedy) — but this is approximate, not exact.
2. **Autoregressive greedy**: Use [run_kv_decode_policy](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/speculative_decoding.py#L264) with argmax sampling.
3. **Speculative greedy (bigram)**: Use `run_spec_decode_policy` with argmax.
4. **Speculative greedy (trigram)**: Use [run_trigram_spec_decode_policy](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/trigram_speculative_decoding.py#L122) with argmax.
5. Assert all three produce **identical token sequences** for each request.

> [!IMPORTANT]
> **Hint — why this works**: Under greedy, the accept/reject coin flip `rand() < min(1, p/q)` always accepts when the draft matches the target argmax (because `p=1, q>0 → ratio ≥ 1`). When the draft is wrong, `p=0` → always reject → resample from `max(0, p-q)` → this is just the target distribution → argmax = target's argmax. So the output must be identical regardless of draft quality.

> [!WARNING]
> **Hint — trigram `prev_prev` tracking**: In `_draft_trigram_tokens`, the `prev_prev, prev` pair must update correctly after rejection. If accepted tokens `[a, b]` are followed by a rejection+resample `c`, then the next iteration's draft context must be `(b, c)`, not `(a, b)`. This is handled by the loop at [L1102](file:///home/colin-zhou/multimodal-inference-visualizer/nanogpt-spec-decode.py#L1102) — verify that `history` reconstruction is correct.

### Key files
- [speculative_decoding.py](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/speculative_decoding.py) — `_accept_reject`, `_draft_tokens`, `run_kv_decode_policy`, `run_spec_decode_policy`
- [trigram_speculative_decoding.py](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/trigram_speculative_decoding.py) — `_draft_trigram_tokens`, `run_trigram_spec_decode_policy`

---

## Test 6: Speculative Decoding Statistically Matches Target Sampling

### What it catches
Subtle distribution-preservation bugs in the rejection sampling math that wouldn't show up under greedy.

### How to implement
1. Fix a short prompt (4 tokens). Generate **1 token only**, repeated **2000+ times** with different seeds.
2. Collect histogram of the generated token from both `run_kv_decode_policy` and `run_trigram_spec_decode_policy`.
3. Compare with chi-squared test:
   ```python
   from scipy.stats import chisquare
   # Only compare bins where expected > 0
   stat, p_value = chisquare(observed_counts, expected_counts)
   assert p_value > 0.01
   ```

> [!TIP]
> **Hint — why 1 token**: Multi-token sequences diverge due to different RNG consumption patterns between the two paths. Testing 1 token isolates the question "does accept/reject preserve the target distribution?" from "do the RNG draws align?"

> [!WARNING]
> **Hint — this is statistical**: It will flake ~1% of the time by construction (p=0.01 threshold). Use a large sample size and consider running 3 seeds and requiring ≥2 pass.

---

## Test 7: KV Cache Trim Consistency

### What it catches
Rejected draft tokens leaking into the cache, off-by-one in `keep = cache_len_before_verify + num_accepted`.

### How to implement
1. Prefill a prompt, decode 2 tokens to build a cache.
2. Run a verify pass with 4 draft candidates → get `new_kvs`.
3. "Accept" only 2 of 4. Trim: `trimmed = trim_kv_cache(new_kvs, 2, cache_len_before)`.
4. Assert shape: `trimmed[0][0][0].shape[1] == cache_len_before + 2`.
5. Decode 1 more token with `trimmed` as cache → get `logits_A`.
6. Full recompute `prompt + 2_decoded + 2_accepted` → get `logits_B` at last position.
7. Assert `torch.allclose(logits_A, logits_B, atol=1e-5)`.

---

## Test 8: Draft Model Distribution Sanity (Trigram + Bigram)

### What it catches
Smoothing errors, normalization failures, fallback logic bugs.

### How to implement

**8a — Known distribution**: Synthetic corpus `[0,1,2,0,1,2,0,1,2]`, `vocab_size=3`. After trigram `(0,1)`, next should strongly favor `2`. Assert `get_probs(0, 1)[2] > 0.8`.

**8b — Normalization**: For 20 random `(prev, cur)` pairs, assert `get_probs(prev, cur).sum() ≈ 1.0` with `atol=1e-6`.

**8c — Fallback** (for the version in `nanogpt-trigram-spec-decode.py` with `fallback_bigram`): Pick `(a, b)` that never appears in corpus. Assert `trigram.get_probs(a, b)` equals `bigram.get_probs(b)`.

---

## Additional Tests Beyond the Original List

### Test 9: Chunked Prefill == Full Prefill

If you have [normal_chunked_prefill.py](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/normal_chunked_prefill.py), test that chunking a 16-token prompt into 4-token chunks produces the same final KV cache logits as prefilling all 16 at once.

### Test 10: Fused Batch (Interleaved Prefill+Decode) == Sequential Prefill Then Decode

If you have the [interleaving.py](file:///home/colin-zhou/multimodal-inference-visualizer/benchmarks/interleaving.py) fused batch path, verify that running one prefill + N decodes in a fused batch produces the same tokens as running them separately.

---

## Proposed File Structure

```
benchmarks/
  test_correctness_equivalence.py    # [NEW] — all tests
```

Each test is a function that takes `(model, vocab_size, device, block_size, train_data, val_data)`. A `run_all_correctness_tests()` function calls them in order and prints pass/fail.

## Dependency Summary

| Test | Imports from |
|---|---|
| 1 (logits) | Model directly |
| 2 (batching) | `single_req_cont_batching` |
| 3 (paged) | `paged_attention` |
| 4 (prefix) | `prefix_caching` |
| 5 (spec greedy) | `speculative_decoding`, `trigram_speculative_decoding` |
| 6 (spec stats) | `speculative_decoding`, `trigram_speculative_decoding` |
| 7 (trim) | Model directly, `_trim_kv_cache` |
| 8 (draft model) | `TrigramDraftModel`, `BigramDraftModel` |

## Open Questions

1. **Greedy mode**: Tests 2–5 need argmax sampling. Do you want to (a) add a `greedy` flag to the existing `_sample_next_token` helpers, or (b) write standalone greedy loops in the test file?

2. **Test runner**: `pytest` with `assert`, or a plain script with `print("PASS")`/`print("FAIL")`?

3. **Chunked prefill + interleaving**: Do you want Tests 9 and 10 included, or are the core 5 from your list (+ Tests 7–8) enough for now?
