"""
Correctness equivalence tests for the NanoGPT inference engine.

Each test isolates one optimization and proves it produces identical
logits/outputs to the simplest baseline.

Usage:
    # From the project root, after training:
    python benchmarks/test_correctness_equivalence.py

    # Or import and call with an already-trained model:
    from benchmarks.test_correctness_equivalence import run_all_correctness_tests
    run_all_correctness_tests(model, vocab_size=vocab_size, device=device,
                              block_size=block_size, train_data=train_data,
                              val_data=val_data)
"""

import torch
import torch.nn.functional as F
from scipy.stats import chi2 as chi2_dist
from benchmarks.single_req_cont_batching import _stack_kvs, _unstack_kvs
from benchmarks.paged_attention import (
    KVBlockPool, BlockAllocator,
    _write_kvs_to_pool, _gather_paged_kv, _infer_cache_shape,
)
from benchmarks.prefix_caching import (
    PrefixBlockCache, PrefixRequestSpec, PrefixRequestState,
    _load_cached_prefix, _commit_completed_blocks,
)
from benchmarks.speculative_decoding import (
    BigramDraftModel, _trim_kv_cache, _make_generator,
)
from benchmarks.trigram_speculative_decoding import TrigramDraftModel


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _greedy_token(logits):
    """Argmax over last-position logits → single token id."""
    return logits[0, -1, :].argmax().item()


def _pass_fail(name, passed):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {name}")
    return passed


def _greedy_decode(model, past_kvs, last_token, num_steps, device):
    """Decode num_steps tokens greedily (argmax), return token list."""
    generated = []
    for _ in range(num_steps):
        cache_len = past_kvs[0][0][0].shape[1]
        inp = torch.tensor([[last_token]], dtype=torch.long, device=device)
        pos = torch.tensor([[cache_len]], dtype=torch.long, device=device)
        logits, _, past_kvs = model(inp, pos=pos, past_kvs=past_kvs)
        last_token = _greedy_token(logits)
        generated.append(last_token)
    return generated, past_kvs


# ──────────────────────────────────────────────────────────────────────
# Test 1: Full-recompute logits == KV-cached incremental logits
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_recompute_vs_kv_cache(model, *, val_data, device, block_size,
                                num_decode_steps=5, prompt_len=10, **_kw):
    """
    Prefill a prompt, then decode N tokens with a KV cache.
    Separately, run the full (prompt + decoded) sequence through a single
    forward pass.  The logits at each decode position must match.
    """
    model.eval()

    # --- pick a prompt from val_data ---
    prompt = val_data[:prompt_len].tolist()
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
    positions = torch.arange(prompt_len, device=device).unsqueeze(0)

    # --- cached path: prefill then decode one-by-one ---
    logits, _, past_kvs = model(prompt_t, pos=positions)
    first_token = _greedy_token(logits)

    cached_logits = [logits[0, -1, :].clone()]   # logit that produced first_token
    generated = [first_token]

    for _ in range(num_decode_steps - 1):
        cache_len = past_kvs[0][0][0].shape[1]
        inp = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        pos = torch.tensor([[cache_len]], dtype=torch.long, device=device)
        logits, _, past_kvs = model(inp, pos=pos, past_kvs=past_kvs)
        cached_logits.append(logits[0, -1, :].clone())
        generated.append(_greedy_token(logits))

    # --- recompute path: full sequence, no cache ---
    full_seq = prompt + generated
    full_t = torch.tensor([full_seq], dtype=torch.long, device=device)
    full_pos = torch.arange(len(full_seq), device=device).unsqueeze(0)
    full_logits, _, _ = model(full_t, pos=full_pos)

    # Compare logits at each decode position
    # cached_logits[i] corresponds to full_logits[0, prompt_len - 1 + i, :]
    all_close = True
    for i in range(len(cached_logits)):
        recompute_logit = full_logits[0, prompt_len - 1 + i, :]
        if not torch.allclose(cached_logits[i], recompute_logit, atol=1e-5):
            max_diff = (cached_logits[i] - recompute_logit).abs().max().item()
            print(f"    Mismatch at decode step {i}: max_diff={max_diff:.6f}")
            all_close = False

    return _pass_fail("recompute logits == kv-cached logits", all_close)


# ──────────────────────────────────────────────────────────────────────
# Test 2: Unbatched output == Continuously batched output
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_unbatched_vs_batched(model, *, vocab_size, device, block_size, **_kw):
    """
    Prefill+decode several requests individually, then do the same thing
    with batched forward passes using _stack_kvs/_unstack_kvs.

    Greedy (argmax) decoding eliminates RNG sensitivity — any divergence
    is a real bug in the KV-stacking, position computation, or batching.
    """

    model.eval()

    num_requests = 3
    prompt_len   = min(8, block_size // 4)
    decode_steps = min(5, block_size // 4)

    torch.manual_seed(42)
    prompts = [torch.randint(0, vocab_size, (prompt_len,)).tolist()
               for _ in range(num_requests)]

    # ---- Unbatched: decode each request completely alone ----
    unbatched_tokens = []
    for prompt in prompts:
        prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
        pos = torch.arange(prompt_len, device=device).unsqueeze(0)
        logits, _, kvs = model(prompt_t, pos=pos)
        first = _greedy_token(logits)
        rest, _ = _greedy_decode(model, kvs, first, decode_steps - 1, device)
        unbatched_tokens.append([first] + rest)

    # ---- Batched: prefill individually, then decode all together ----
    batched_tokens = [[] for _ in range(num_requests)]
    all_kvs = []
    last_tokens = []

    for i, prompt in enumerate(prompts):
        prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
        pos = torch.arange(prompt_len, device=device).unsqueeze(0)
        logits, _, kvs = model(prompt_t, pos=pos)
        first = _greedy_token(logits)
        batched_tokens[i].append(first)
        last_tokens.append(first)
        all_kvs.append(kvs)

    for _ in range(decode_steps - 1):
        stacked = _stack_kvs(all_kvs)
        cache_len = all_kvs[0][0][0][0].shape[1]

        inp = torch.tensor([[t] for t in last_tokens],
                           dtype=torch.long, device=device)
        pos = torch.full((num_requests, 1), cache_len,
                         dtype=torch.long, device=device)

        logits, _, new_stacked = model(inp, pos=pos, past_kvs=stacked)
        all_kvs = _unstack_kvs(new_stacked)

        last_tokens = []
        for i in range(num_requests):
            token = logits[i, -1, :].argmax().item()
            batched_tokens[i].append(token)
            last_tokens.append(token)

    all_match = True
    for i in range(num_requests):
        if unbatched_tokens[i] != batched_tokens[i]:
            print(f"    Request {i}: unbatched={unbatched_tokens[i]}, "
                  f"batched={batched_tokens[i]}")
            all_match = False

    return _pass_fail("unbatched == batched output (argmax)", all_match)


# ──────────────────────────────────────────────────────────────────────
# Test 3: Contiguous KV output == Paged KV output
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_contiguous_vs_paged_kv(model, *, vocab_size, device, block_size, **_kw):
    """
    Decode with a normal contiguous KV cache vs writing/gathering through a
    paged KV block pool.  Greedy decoding, token-for-token comparison.

    Tests multiple prompt lengths (3, 4, 5, 7, 8) to stress block boundaries
    so off-by-one errors in num_filled_slots // page_block_size show up.
    """


    model.eval()

    page_block_size = 4
    decode_steps    = min(6, block_size // 4)

    # Vary prompt lengths to land on and off block boundaries.
    prompt_lens = [pl for pl in [3, 4, 5, 7, 8]
                   if pl + decode_steps <= block_size]
    if not prompt_lens:
        prompt_lens = [min(3, block_size - decode_steps)]

    n_layer, n_head, head_size, kv_dtype = _infer_cache_shape(
        model, vocab_size, device
    )

    torch.manual_seed(42)
    all_ok = True

    for prompt_len in prompt_lens:
        prompt = torch.randint(0, vocab_size, (prompt_len,)).tolist()
        prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
        pos = torch.arange(prompt_len, device=device).unsqueeze(0)

        # ---- Contiguous path (baseline) ----
        logits_c, _, kvs_c = model(prompt_t, pos=pos)
        first_c = _greedy_token(logits_c)
        rest_c, _ = _greedy_decode(
            model, kvs_c, first_c, decode_steps - 1, device
        )
        contiguous_gen = [first_c] + rest_c

        # ---- Paged path ----
        pool = KVBlockPool(
            64, page_block_size, n_layer, n_head, head_size, device, kv_dtype
        )
        allocator = BlockAllocator(64)

        logits_p, _, kvs_p = model(prompt_t, pos=pos)
        blocks_needed = (prompt_len + page_block_size - 1) // page_block_size
        block_table = allocator.allocate_n(blocks_needed)
        _write_kvs_to_pool(pool, block_table, page_block_size, 0, kvs_p)
        num_filled = prompt_len

        paged_gen = [_greedy_token(logits_p)]
        for _ in range(decode_steps - 1):
            # Allocate a new block if we've hit a boundary.
            if num_filled % page_block_size == 0:
                block_table.append(allocator.allocate_one())

            # Gather contiguous KV from the pool.
            # _gather_paged_kv expects an object with .block_table / .num_filled_slots
            class _Req:
                pass
            req_obj = _Req()
            req_obj.block_table = block_table
            req_obj.num_filled_slots = num_filled

            gathered = []
            for layer_idx in range(n_layer):
                layer_kv = []
                for head_idx in range(n_head):
                    k, v = _gather_paged_kv(
                        pool, req_obj, page_block_size, layer_idx, head_idx
                    )
                    layer_kv.append((k, v))
                gathered.append(layer_kv)

            inp = torch.tensor([[paged_gen[-1]]],
                               dtype=torch.long, device=device)
            p = torch.tensor([[num_filled]],
                             dtype=torch.long, device=device)
            logits_p, _, new_kvs = model(
                inp, pos=p, past_kvs=gathered
            )

            # Write only the new token's KV back to pool.
            for layer_idx in range(n_layer):
                for head_idx, (k, v) in enumerate(new_kvs[layer_idx]):
                    phys = block_table[num_filled // page_block_size]
                    slot = num_filled % page_block_size
                    pool.k_pool[(layer_idx, head_idx)][phys, slot, :] = \
                        k[0, -1, :]
                    pool.v_pool[(layer_idx, head_idx)][phys, slot, :] = \
                        v[0, -1, :]

            num_filled += 1
            paged_gen.append(_greedy_token(logits_p))

        if contiguous_gen != paged_gen:
            print(f"    prompt_len={prompt_len}: "
                  f"contiguous={contiguous_gen}, paged={paged_gen}")
            all_ok = False

    return _pass_fail("contiguous KV == paged KV (argmax)", all_ok)


# ──────────────────────────────────────────────────────────────────────
# Test 4: Prefix-cached output == Normal prefill output
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_prefix_cached_vs_normal(model, *, vocab_size, device, block_size,
                                  train_data, **_kw):
    """
    Prefill a prompt normally (one forward pass), then separately split the
    same prompt into cached-prefix blocks + remaining suffix and prefill in
    two parts.  Decode from both and compare generated tokens.

    Uses the prefix_caching module's helpers (_load_cached_prefix,
    _commit_completed_blocks) to populate and read from the block cache,
    so the test exercises the hash-chaining and KV-slicing logic.
    """


    model.eval()

    prefix_block_size = 4
    shared_prefix_len = min(12, block_size // 3)
    # Round down to a multiple of prefix_block_size so we have full blocks.
    shared_prefix_len = (shared_prefix_len // prefix_block_size) * prefix_block_size
    unique_suffix_len = min(4, block_size // 6)
    decode_steps = min(4, block_size // 6)

    if shared_prefix_len < prefix_block_size or unique_suffix_len < 1:
        return _pass_fail("prefix-cached == normal prefill (argmax)", True)

    # Build 3 prompts: same prefix, different suffixes.
    torch.manual_seed(42)
    prefix = torch.randint(0, vocab_size, (shared_prefix_len,)).tolist()
    suffixes = [torch.randint(0, vocab_size, (unique_suffix_len,)).tolist()
                for _ in range(3)]
    prompts = [prefix + s for s in suffixes]

    # ---- Baseline: full prefill, no caching ----
    baseline_tokens = []
    for prompt in prompts:
        prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
        pos = torch.arange(len(prompt), device=device).unsqueeze(0)
        logits, _, kvs = model(prompt_t, pos=pos)
        first = _greedy_token(logits)
        rest, _ = _greedy_decode(model, kvs, first, decode_steps - 1, device)
        baseline_tokens.append([first] + rest)

    # ---- Cached path: request 0 populates the cache, requests 1-2 hit it ----
    block_cache = PrefixBlockCache(max_blocks=64)
    cached_tokens = []

    for i, prompt in enumerate(prompts):
        spec = PrefixRequestSpec(
            id=i, prompt_tokens=prompt,
            max_new_tokens=decode_steps, arrival_step=i,
        )
        req = PrefixRequestState(spec=spec)
        block_cache.current_step = i

        # Load whatever prefix blocks are in the cache.
        _load_cached_prefix(req, block_cache, prefix_block_size)

        # Prefill the remaining suffix (the part not in cache).
        start = req.prefill_cursor
        end = len(prompt)
        idx = torch.tensor([prompt[start:end]], dtype=torch.long, device=device)
        pos = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0)
        logits, _, new_kvs = model(idx, pos=pos, past_kvs=req.past_kvs)
        req.past_kvs = new_kvs

        # Commit completed blocks so next request can use them.
        _commit_completed_blocks(req, block_cache, prefix_block_size)

        # Verify requests after the first actually loaded cached blocks.
        if i > 0 and req.cached_prefix_tokens == 0:
            print(f"    Request {i} didn't hit prefix cache (expected >0)")

        # Greedy decode.
        first = _greedy_token(logits)
        rest, _ = _greedy_decode(model, req.past_kvs, first, decode_steps - 1, device)
        cached_tokens.append([first] + rest)

    all_match = True
    for i in range(len(prompts)):
        if baseline_tokens[i] != cached_tokens[i]:
            print(f"    Request {i}: baseline={baseline_tokens[i]}, "
                  f"cached={cached_tokens[i]}")
            all_match = False

    return _pass_fail("prefix-cached == normal prefill (argmax)", all_match)


# ──────────────────────────────────────────────────────────────────────
# Test 5: Speculative decoding greedy == Autoregressive greedy
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_speculative_greedy_vs_autoregressive(model, *, vocab_size, device,
                                               block_size, train_data, **_kw):
    """
    Under greedy (argmax), speculative decoding must produce the exact same
    tokens as simple autoregressive decoding — regardless of draft quality.

    Why: when the draft matches the target argmax, p=1 so accept_prob ≥ 1 →
    always accept.  When it doesn't, p=0 → always reject → resample from
    max(0, target-draft) which is the target distribution → argmax = target's
    argmax.  So accept/reject is a no-op under greedy.

    This test implements the spec-decode verify loop inline (no runner import)
    and checks both bigram and trigram draft models.
    """


    model.eval()

    num_requests  = 3
    prompt_len    = min(8, block_size // 4)
    decode_tokens = min(8, block_size // 4)
    speculation_k = 4

    torch.manual_seed(42)
    prompts = [torch.randint(0, vocab_size, (prompt_len,)).tolist()
               for _ in range(num_requests)]

    # Build draft models from train_data.
    bigram  = BigramDraftModel(train_data, vocab_size=vocab_size, device=device)
    trigram = TrigramDraftModel(train_data, vocab_size=vocab_size, device=device)

    # ---- Autoregressive baseline (argmax) ----
    auto_tokens = []
    for prompt in prompts:
        prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
        pos = torch.arange(prompt_len, device=device).unsqueeze(0)
        logits, _, kvs = model(prompt_t, pos=pos)
        first = _greedy_token(logits)
        rest, _ = _greedy_decode(model, kvs, first, decode_tokens - 1, device)
        auto_tokens.append([first] + rest)

    def _greedy_spec_decode(prompt, draft_model, is_trigram):
        """Run one request through the spec-decode verify loop with argmax."""
        prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
        pos = torch.arange(len(prompt), device=device).unsqueeze(0)
        logits, _, kvs = model(prompt_t, pos=pos)
        generated = [_greedy_token(logits)]

        while len(generated) < decode_tokens:
            remaining = decode_tokens - len(generated)
            k = min(speculation_k, remaining)

            # ---- Draft k tokens greedily from the draft model ----
            candidates = []
            draft_probs_list = []
            if is_trigram:
                history = prompt + generated
                pp = history[-2] if len(history) >= 2 else history[-1]
                pv = history[-1]
                for _ in range(k):
                    probs = draft_model.get_probs(pp, pv)
                    tok = probs.argmax().item()
                    candidates.append(tok)
                    draft_probs_list.append(probs)
                    pp, pv = pv, tok
            else:
                pv = generated[-1]
                for _ in range(k):
                    probs = draft_model.get_probs(pv)
                    tok = probs.argmax().item()
                    candidates.append(tok)
                    draft_probs_list.append(probs)
                    pv = tok

            # ---- Verify: feed [last_generated] + candidates to model ----
            verify_input = [generated[-1]] + candidates
            cache_len = kvs[0][0][0].shape[1]
            inp = torch.tensor([verify_input], dtype=torch.long, device=device)
            vpos = torch.arange(
                cache_len, cache_len + len(verify_input),
                dtype=torch.long, device=device
            ).unsqueeze(0)
            v_logits, _, new_kvs = model(inp, pos=vpos, past_kvs=kvs)

            # ---- Greedy accept/reject ----
            # target_probs[i] = softmax(v_logits[0, i, :])
            # Under argmax: if draft == target_argmax → accept.
            #                otherwise → reject, take target_argmax.
            accepted = []
            for i, draft_tok in enumerate(candidates):
                target_argmax = v_logits[0, i, :].argmax().item()
                if draft_tok == target_argmax:
                    accepted.append(draft_tok)
                else:
                    # Reject: take the target's argmax at this position.
                    accepted.append(target_argmax)
                    break  # Stop at first rejection.

            num_accepted = len(accepted) - (
                1 if accepted and accepted[-1] != candidates[len(accepted) - 1] else 0
            )

            # If all k accepted and we have room, take a bonus token.
            if num_accepted == k and remaining > k:
                bonus = v_logits[0, k, :].argmax().item()
                accepted.append(bonus)

            # Trim KV cache: keep prefix + (1 for last_generated) + num_accepted
            kvs = _trim_kv_cache(
                new_kvs,
                cache_len_before_verify=cache_len,
                keep_new_tokens=1 + num_accepted,
            )

            # Record accepted tokens (don't exceed decode_tokens).
            for tok in accepted[:remaining]:
                generated.append(tok)

        return generated[:decode_tokens]

    all_match = True

    for i, prompt in enumerate(prompts):
        # Bigram spec decode
        bi_tokens = _greedy_spec_decode(prompt, bigram, is_trigram=False)
        if auto_tokens[i] != bi_tokens:
            print(f"    Request {i} bigram mismatch: auto={auto_tokens[i]}, "
                  f"spec={bi_tokens}")
            all_match = False

        # Trigram spec decode
        tri_tokens = _greedy_spec_decode(prompt, trigram, is_trigram=True)
        if auto_tokens[i] != tri_tokens:
            print(f"    Request {i} trigram mismatch: auto={auto_tokens[i]}, "
                  f"spec={tri_tokens}")
            all_match = False

    return _pass_fail(
        "speculative greedy == autoregressive greedy (argmax)", all_match
    )


# ──────────────────────────────────────────────────────────────────────
# Test 6: Speculative decoding statistically matches target sampling
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_spec_decode_distribution(model, *, vocab_size, device, block_size,
                                   train_data, **_kw):
    """
    Verify that speculative decoding preserves the target model's sampling
    distribution.  Generate a single token from the same prompt many times
    via (a) plain autoregressive sampling and (b) spec-decode, then compare
    the two histograms with a chi-squared test.

    Only 1 token is generated per trial so that RNG-consumption differences
    between the two paths don't cause divergence — this isolates the question
    "does accept/reject preserve the target distribution?"

    Runs 3 independent seeds; requires ≥2 to pass (p > 0.01) to tolerate
    the inherent ~1% flake rate of chi-squared tests.
    """


    model.eval()

    prompt_len    = min(4, block_size // 4)
    num_samples   = 2000
    speculation_k = 4
    temperature   = 1.0

    torch.manual_seed(42)
    prompt = torch.randint(0, vocab_size, (prompt_len,)).tolist()

    bigram  = BigramDraftModel(train_data, vocab_size=vocab_size, device=device)
    trigram = TrigramDraftModel(train_data, vocab_size=vocab_size, device=device)

    # Prefill once — the KV cache is the same for every sample.
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
    pos = torch.arange(prompt_len, device=device).unsqueeze(0)
    base_logits, _, base_kvs = model(prompt_t, pos=pos)
    target_dist = F.softmax(base_logits[0, -1, :] / temperature, dim=-1)

    def _sample_auto(seed):
        """Generate num_samples single tokens via autoregressive sampling."""
        gen = _make_generator(device, seed)
        counts = torch.zeros(vocab_size, device=device)
        for _ in range(num_samples):
            tok = torch.multinomial(target_dist, 1, generator=gen).item()
            counts[tok] += 1
        return counts

    def _sample_spec(seed, draft_model, is_trigram):
        """Generate num_samples single tokens via spec-decode accept/reject.

        Tests the accept/reject math in isolation: draft one candidate from the
        draft model, accept/reject against target_dist directly.  No model
        forward pass needed — this is a pure test of the sampling algorithm.
        """
        gen = _make_generator(device, seed)
        counts = torch.zeros(vocab_size, device=device)
        for _ in range(num_samples):
            # Draft one candidate from the draft model.
            if is_trigram:
                pp = prompt[-2] if len(prompt) >= 2 else prompt[-1]
                pv = prompt[-1]
                draft_probs = draft_model.get_probs(pp, pv, temperature=temperature)
            else:
                pv = prompt[-1]
                draft_probs = draft_model.get_probs(pv, temperature=temperature)

            draft_tok = torch.multinomial(draft_probs, 1, generator=gen).item()

            # Accept/reject against the known target distribution.
            q = draft_probs[draft_tok].clamp_min(1e-12)
            p = target_dist[draft_tok]
            accept_prob = (p / q).clamp(max=1.0)

            draw = torch.rand((), device=device, generator=gen)
            if draw.item() < accept_prob.item():
                output_token = draft_tok
            else:
                # Reject — resample from max(0, target - draft).
                adjusted = torch.clamp(target_dist - draft_probs, min=0)
                adj_sum = adjusted.sum()
                if adj_sum > 0:
                    adjusted = adjusted / adj_sum
                else:
                    adjusted = target_dist
                output_token = torch.multinomial(
                    adjusted, 1, generator=gen
                ).item()

            counts[output_token] += 1
        return counts

    def _chi_squared_ok(auto_counts, spec_counts, min_expected=5):
        """Return True if the two distributions match (p > 0.01)."""
        # Only compare bins where expected count > min_expected.
        total_auto = auto_counts.sum().item()
        total_spec = spec_counts.sum().item()
        expected = auto_counts * (total_spec / total_auto)

        mask = expected > min_expected
        if mask.sum() < 2:
            return True  # Not enough data to test.

        obs = spec_counts[mask].cpu().float()
        exp = expected[mask].cpu().float()

        # Chi-squared statistic.
        chi2 = ((obs - exp) ** 2 / exp).sum().item()
        dof = int(mask.sum().item()) - 1
        if dof <= 0:
            return True

        # Approximate p-value using the survival function of chi2.
        # For simplicity, use a rough threshold: chi2 / dof > 3 ≈ p < 0.01
        # for moderate dof.  More precise: use scipy if available.
        try:
            p_value = chi2_dist.sf(chi2, dof)
        except ImportError:
            # Fallback: chi2/dof > 2.5 is roughly p < 0.01 for dof > 5.
            p_value = 0.5 if chi2 / max(dof, 1) < 2.5 else 0.001
        return p_value > 0.01

    seeds = [100, 200, 300]
    pass_count = 0

    for seed in seeds:
        auto_counts = _sample_auto(seed)
        # Test with trigram (harder draft → more rejections → better stress test).
        spec_counts = _sample_spec(seed + 1, trigram, is_trigram=True)
        if _chi_squared_ok(auto_counts, spec_counts):
            pass_count += 1

    ok = pass_count >= 2
    if not ok:
        print(f"    Only {pass_count}/3 seeds passed chi-squared test")
    return _pass_fail(
        "spec-decode distribution ≈ target distribution (chi²)", ok
    )


# ──────────────────────────────────────────────────────────────────────
# Test 7: KV cache trim consistency
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_kv_cache_trim(model, *, vocab_size, device, block_size, **_kw):
    """
    Verify that _trim_kv_cache correctly discards rejected draft tokens.

    1. Prefill a prompt, decode 2 tokens → build a cache.
    2. Run a verify pass with 4 draft candidates → get new_kvs.
    3. "Accept" only 2 of 4 → trim the cache.
    4. Decode 1 more token from the trimmed cache → logits_A.
    5. Full recompute of (prompt + 2_decoded + 2_accepted) → logits_B.
    6. Assert logits_A ≈ logits_B at the last position.
    """


    model.eval()

    prompt_len     = min(6, block_size // 4)
    decode_tokens  = 2
    draft_k        = 4
    accept_count   = 2

    total_len = prompt_len + decode_tokens + 1 + draft_k + 1
    if total_len > block_size:
        return _pass_fail("KV cache trim consistency", True)

    torch.manual_seed(42)
    prompt = torch.randint(0, vocab_size, (prompt_len,)).tolist()

    # Step 1: Prefill.
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
    pos = torch.arange(prompt_len, device=device).unsqueeze(0)
    logits, _, kvs = model(prompt_t, pos=pos)

    # Step 2: Decode 2 tokens greedily.
    first = _greedy_token(logits)
    decoded = [first]
    for _ in range(decode_tokens - 1):
        cache_len = kvs[0][0][0].shape[1]
        inp = torch.tensor([[decoded[-1]]], dtype=torch.long, device=device)
        p = torch.tensor([[cache_len]], dtype=torch.long, device=device)
        logits, _, kvs = model(inp, pos=p, past_kvs=kvs)
        decoded.append(_greedy_token(logits))

    # Step 3: Simulate a verify pass — feed [last_decoded] + 4 draft candidates.
    draft_candidates = torch.randint(0, vocab_size, (draft_k,)).tolist()
    verify_input = [decoded[-1]] + draft_candidates
    cache_len_before = kvs[0][0][0].shape[1]
    inp = torch.tensor([verify_input], dtype=torch.long, device=device)
    vpos = torch.arange(
        cache_len_before, cache_len_before + len(verify_input),
        dtype=torch.long, device=device
    ).unsqueeze(0)
    _, _, new_kvs = model(inp, pos=vpos, past_kvs=kvs)

    # Step 4: "Accept" only 2 of 4 → trim.
    trimmed = _trim_kv_cache(
        new_kvs,
        cache_len_before_verify=cache_len_before,
        keep_new_tokens=1 + accept_count,  # +1 for the verify input's first token
    )

    # Verify shape.
    expected_cache_len = cache_len_before + 1 + accept_count
    actual_cache_len = trimmed[0][0][0].shape[1]
    shape_ok = actual_cache_len == expected_cache_len
    if not shape_ok:
        print(f"    Shape mismatch: expected cache_len={expected_cache_len}, "
              f"got {actual_cache_len}")

    # Step 5: Decode 1 more token from trimmed cache.
    accepted_tokens = draft_candidates[:accept_count]
    last_accepted = accepted_tokens[-1]
    inp_a = torch.tensor([[last_accepted]], dtype=torch.long, device=device)
    pos_a = torch.tensor([[actual_cache_len]], dtype=torch.long, device=device)
    logits_a, _, _ = model(inp_a, pos=pos_a, past_kvs=trimmed)

    # Step 6: Full recompute — no cache.
    full_seq = prompt + decoded + accepted_tokens + [last_accepted]
    full_t = torch.tensor([full_seq], dtype=torch.long, device=device)
    full_pos = torch.arange(len(full_seq), device=device).unsqueeze(0)
    logits_b, _, _ = model(full_t, pos=full_pos)

    logits_close = torch.allclose(
        logits_a[0, -1, :], logits_b[0, -1, :], atol=1e-5
    )
    if not logits_close:
        diff = (logits_a[0, -1, :] - logits_b[0, -1, :]).abs().max().item()
        print(f"    Logits mismatch after trim: max_diff={diff:.6f}")

    return _pass_fail("KV cache trim consistency", shape_ok and logits_close)


# ──────────────────────────────────────────────────────────────────────
# Test 8: Draft model distribution sanity (quick, no model needed)
# ──────────────────────────────────────────────────────────────────────

def test_draft_model_distributions(*, device, **_kw):
    """
    Verify trigram and bigram draft models produce valid, peaked distributions.
    """


    all_ok = True

    # ---- 8a: Known distribution (trigram) ----
    corpus = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    tri = TrigramDraftModel(corpus, vocab_size=3, device=device)
    probs_01 = tri.get_probs(0, 1)

    # After (0,1), token 2 appeared 2 times out of 2 observations (+ smoothing)
    if not (probs_01[2] > 0.5):
        print(f"    Trigram (0,1)->2 prob={probs_01[2]:.4f}, expected > 0.5")
        all_ok = False

    # ---- 8b: Normalization ----
    big_corpus = list(range(20)) * 10
    tri_big = TrigramDraftModel(big_corpus, vocab_size=20, device=device)
    bi_big = BigramDraftModel(big_corpus, vocab_size=20, device=device)

    for prev in range(0, 20, 5):
        for cur in range(0, 20, 5):
            s = tri_big.get_probs(prev, cur).sum().item()
            if abs(s - 1.0) > 1e-5:
                print(f"    Trigram normalization fail: sum={s:.6f} for ({prev},{cur})")
                all_ok = False

            s2 = bi_big.get_probs(cur).sum().item()
            if abs(s2 - 1.0) > 1e-5:
                print(f"    Bigram normalization fail: sum={s2:.6f} for ({cur})")
                all_ok = False

    return _pass_fail("draft model distributions are valid", all_ok)


# ──────────────────────────────────────────────────────────────────────
# Test 9: Chunked prefill == Full prefill
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_chunked_vs_full_prefill(model, *, vocab_size, device, block_size,
                                  **_kw):
    """
    Prefill a prompt in one shot, then separately prefill the same prompt
    in small chunks (accumulating the KV cache across forward calls).
    The logits at the last position must match.

    This catches bugs in positional-encoding threading, KV-cache concatenation
    across chunk boundaries, and off-by-one errors in prefill_cursor logic.
    """
    model.eval()

    prompt_len = min(16, block_size - 4)
    chunk_size = 4

    if prompt_len < chunk_size:
        return _pass_fail("chunked prefill == full prefill", True)

    torch.manual_seed(42)
    prompt = torch.randint(0, vocab_size, (prompt_len,)).tolist()
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
    pos_full = torch.arange(prompt_len, device=device).unsqueeze(0)

    # ---- Full prefill (one shot) ----
    logits_full, _, kvs_full = model(prompt_t, pos=pos_full)

    # ---- Chunked prefill ----
    kvs_chunked = None
    for start in range(0, prompt_len, chunk_size):
        end = min(start + chunk_size, prompt_len)
        chunk = torch.tensor([prompt[start:end]], dtype=torch.long, device=device)
        pos = torch.arange(start, end, device=device).unsqueeze(0)
        logits_chunked, _, kvs_chunked = model(
            chunk, pos=pos, past_kvs=kvs_chunked
        )

    # Compare logits at last position.
    logits_match = torch.allclose(
        logits_full[0, -1, :], logits_chunked[0, -1, :], atol=1e-5
    )
    if not logits_match:
        diff = (logits_full[0, -1, :] - logits_chunked[0, -1, :]).abs().max().item()
        print(f"    Last-position logits mismatch: max_diff={diff:.6f}")

    # Also verify KV cache shapes match.
    full_cache_len = kvs_full[0][0][0].shape[1]
    chunked_cache_len = kvs_chunked[0][0][0].shape[1]
    shape_match = full_cache_len == chunked_cache_len
    if not shape_match:
        print(f"    KV cache shape mismatch: full={full_cache_len}, "
              f"chunked={chunked_cache_len}")

    # Decode one token from each and compare.
    first_full = _greedy_token(logits_full)
    first_chunked = _greedy_token(logits_chunked)
    token_match = first_full == first_chunked
    if not token_match:
        print(f"    First decode token mismatch: full={first_full}, "
              f"chunked={first_chunked}")

    return _pass_fail(
        "chunked prefill == full prefill",
        logits_match and shape_match and token_match,
    )


# ──────────────────────────────────────────────────────────────────────
# Test 10: Fused batch (interleaved prefill+decode) == Sequential
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_fused_interleaved_vs_sequential(model, *, vocab_size, device,
                                          block_size, **_kw):
    """
    Test that packing a decode token and a prefill chunk into one batched
    forward pass (with left-padding + attn_mask) produces the same logits
    as running them as separate forwards.

    Scenario:
      - Request A: already prefilled, has a KV cache, is decoding.
      - Request B: new arrival, needs a prefill chunk.
    We run both in a fused batch (row 0 = decode, row 1 = prefill chunk)
    and compare against running each one individually.
    """
    model.eval()

    prompt_len_a = min(6, block_size // 4)
    prompt_len_b = min(8, block_size // 3)
    chunk_size   = min(4, block_size // 4)

    if prompt_len_a + 2 > block_size or chunk_size < 1:
        return _pass_fail("fused interleaved == sequential", True)

    torch.manual_seed(42)
    prompt_a = torch.randint(0, vocab_size, (prompt_len_a,)).tolist()
    prompt_b = torch.randint(0, vocab_size, (prompt_len_b,)).tolist()

    # ---- Setup request A: prefill + 1 decode step ----
    pa = torch.tensor([prompt_a], dtype=torch.long, device=device)
    pos_a = torch.arange(prompt_len_a, device=device).unsqueeze(0)
    logits_a, _, kvs_a = model(pa, pos=pos_a)
    first_a = _greedy_token(logits_a)

    # Decode one step to build up cache.
    inp_a = torch.tensor([[first_a]], dtype=torch.long, device=device)
    pos_a1 = torch.tensor([[prompt_len_a]], dtype=torch.long, device=device)
    logits_a1, _, kvs_a = model(inp_a, pos=pos_a1, past_kvs=kvs_a)
    second_a = _greedy_token(logits_a1)

    # Now A has cache_len = prompt_len_a + 1, wants to decode `second_a`.
    cache_len_a = kvs_a[0][0][0].shape[1]  # prompt_len_a + 1

    # ---- Sequential: run A's next decode and B's prefill chunk separately ----
    # A: decode one more token.
    inp_a_seq = torch.tensor([[second_a]], dtype=torch.long, device=device)
    pos_a_seq = torch.tensor([[cache_len_a]], dtype=torch.long, device=device)
    logits_a_seq, _, _ = model(inp_a_seq, pos=pos_a_seq, past_kvs=kvs_a)

    # B: prefill first chunk_size tokens.
    b_chunk = prompt_b[:chunk_size]
    inp_b_seq = torch.tensor([b_chunk], dtype=torch.long, device=device)
    pos_b_seq = torch.arange(chunk_size, device=device).unsqueeze(0)
    logits_b_seq, _, _ = model(inp_b_seq, pos=pos_b_seq)

    # ---- Fused: pack both into one forward with left-padding + attn_mask ----
    # Row 0 = decode (1 token), Row 1 = prefill (chunk_size tokens).
    # Max sequence length = chunk_size. Row 0 is left-padded.
    t_max = max(1, chunk_size)
    pad_a = t_max - 1  # padding for decode row

    # Build input ids: pad decode row on the left with zeros.
    fused_ids = torch.zeros((2, t_max), dtype=torch.long, device=device)
    fused_ids[0, -1] = second_a              # decode token at last position
    fused_ids[1, :chunk_size] = torch.tensor(b_chunk, dtype=torch.long, device=device)

    # Build positions.
    fused_pos = torch.zeros((2, t_max), dtype=torch.long, device=device)
    fused_pos[0, -1] = cache_len_a
    fused_pos[1, :chunk_size] = torch.arange(chunk_size, device=device)

    # Build KV cache: stack A's cache with an empty cache for B.
    # A has (1, cache_len_a, hs) per head. B has nothing.
    # Left-pad A's cache to match, B gets zeros up to cache_len_a.
    n_layer = len(kvs_a)
    n_head = len(kvs_a[0])
    hs = kvs_a[0][0][0].shape[-1]
    kv_dtype = kvs_a[0][0][0].dtype

    stacked_kvs = []
    for layer_idx in range(n_layer):
        layer_kv = []
        for head_idx in range(n_head):
            ka, va = kvs_a[layer_idx][head_idx]
            # B has no cache → all-zero row of same length.
            kb = torch.zeros((1, cache_len_a, hs), dtype=kv_dtype, device=device)
            vb = torch.zeros((1, cache_len_a, hs), dtype=kv_dtype, device=device)
            layer_kv.append((
                torch.cat([ka, kb], dim=0),
                torch.cat([va, vb], dim=0),
            ))
        stacked_kvs.append(layer_kv)

    # Attention mask: (B=2, 1, cache_len_a)
    # Row 0 (A): all cache positions are real → True.
    # Row 1 (B): no cache → all False (mask out all padded KV).
    attn_mask = torch.zeros((2, 1, cache_len_a), dtype=torch.bool, device=device)
    attn_mask[0, :, :] = True   # A can see its entire cache.
    # attn_mask[1] stays False — B has no real cached positions.

    # Input mask: (B=2, t_max) — which new-input positions are real.
    input_mask = torch.zeros((2, t_max), dtype=torch.bool, device=device)
    input_mask[0, -1] = True                # only the last position is real for decode
    input_mask[1, :chunk_size] = True        # first chunk_size positions are real for prefill

    logits_fused, _, _ = model(
        fused_ids, pos=fused_pos, past_kvs=stacked_kvs,
        attn_mask=attn_mask, input_mask=input_mask,
    )

    # Compare: A's decode logits (at last real position = t_max-1 for row 0).
    logits_a_fused = logits_fused[0, -1, :]
    logits_a_ok = torch.allclose(
        logits_a_seq[0, -1, :], logits_a_fused, atol=1e-5
    )
    if not logits_a_ok:
        diff = (logits_a_seq[0, -1, :] - logits_a_fused).abs().max().item()
        print(f"    Decode row (A) logits mismatch: max_diff={diff:.6f}")

    # Compare: B's prefill logits (at position chunk_size-1 for row 1).
    logits_b_fused = logits_fused[1, chunk_size - 1, :]
    logits_b_ok = torch.allclose(
        logits_b_seq[0, -1, :], logits_b_fused, atol=1e-5
    )
    if not logits_b_ok:
        diff = (logits_b_seq[0, -1, :] - logits_b_fused).abs().max().item()
        print(f"    Prefill row (B) logits mismatch: max_diff={diff:.6f}")

    return _pass_fail(
        "fused interleaved == sequential", logits_a_ok and logits_b_ok
    )


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────

def run_all_correctness_tests(model, *, vocab_size, device, block_size,
                               train_data, val_data):
    print("\n" + "=" * 60)
    print("  Correctness Equivalence Tests")
    print("=" * 60)

    kwargs = dict(
        vocab_size=vocab_size,
        device=device,
        block_size=block_size,
        train_data=train_data,
        val_data=val_data,
    )

    results = {}

    # --- Tests that need the model ---
    results["test1_recompute_vs_kv"]      = test_recompute_vs_kv_cache(model, **kwargs)
    results["test2_unbatched_vs_batched"] = test_unbatched_vs_batched(model, **kwargs)
    results["test3_contiguous_vs_paged"]  = test_contiguous_vs_paged_kv(model, **kwargs)
    results["test4_prefix_cached"]        = test_prefix_cached_vs_normal(model, **kwargs)
    results["test5_spec_decode_greedy"]   = test_speculative_greedy_vs_autoregressive(model, **kwargs)
    results["test6_spec_distribution"]    = test_spec_decode_distribution(model, **kwargs)
    results["test7_kv_trim"]             = test_kv_cache_trim(model, **kwargs)

    # --- Tests that don't need the model ---
    results["test8_draft_distributions"] = test_draft_model_distributions(**kwargs)

    # --- Additional tests ---
    results["test9_chunked_prefill"]      = test_chunked_vs_full_prefill(model, **kwargs)
    results["test10_fused_interleaved"]   = test_fused_interleaved_vs_sequential(model, **kwargs)

    # --- Summary ---
    passed = sum(results.values())
    total = len(results)
    print(f"\n  {passed}/{total} tests passed.")
    if passed < total:
        print("  Failed tests:")
        for name, ok in results.items():
            if not ok:
                print(f"    - {name}")
    print("=" * 60)

    return results
