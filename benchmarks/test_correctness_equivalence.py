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
# Test 8: Draft model distribution sanity (quick, no model needed)
# ──────────────────────────────────────────────────────────────────────

def test_draft_model_distributions(*, device, **_kw):
    """
    Verify trigram and bigram draft models produce valid, peaked distributions.
    """
    from benchmarks.speculative_decoding import BigramDraftModel
    from benchmarks.trigram_speculative_decoding import TrigramDraftModel

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
    results["test1_recompute_vs_kv"] = test_recompute_vs_kv_cache(model, **kwargs)

    # --- Tests that don't need the model ---
    results["test8_draft_distributions"] = test_draft_model_distributions(**kwargs)

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
