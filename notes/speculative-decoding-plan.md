# Speculative Decoding — Implementation Plan & Hints

## The Problem You're Solving

Standard autoregressive decoding is **sequential**: each token requires one full forward pass of the model. Even with KV caching, you're bottlenecked by the number of serial forward passes.

```
Step 1: forward(token_0) → token_1
Step 2: forward(token_1) → token_2
Step 3: forward(token_2) → token_3
...
N tokens = N forward passes
```

**Speculative decoding** breaks this by using a cheap **draft model** to guess K tokens ahead, then **verifying** all K guesses in a single forward pass of the real (target) model. If the guesses are good, you get K+1 tokens for the cost of ~1 target forward pass.

```
Draft:  guess token_1, token_2, token_3, token_4  (cheap, ~free)
Verify: forward([token_0, token_1, token_2, token_3, token_4]) → check all at once
Accept: token_1 ✓, token_2 ✓, token_3 ✗ → resample token_3 from target
Result: 3 tokens from 1 target forward pass!
```

**Key guarantee:** The output distribution is **mathematically identical** to the target model alone. Speculative decoding never degrades quality — it only speeds things up (or, in the worst case, matches normal decoding speed).

---

## What You Already Have (Starting Point)

From your NanoGPT notebooks:

- ✅ `GPTLanguageModel` with stateless `Head.forward()` and KV cache support
- ✅ `model.forward(idx, pos=pos, past_kvs=past_kvs)` returns `(logits, loss, new_kvs)`
- ✅ The model can process multiple tokens at once: `idx` shape `(B, T)` where `T > 1`
- ✅ Training data (`input.txt` — Tiny Shakespeare) for building a bigram table
- ✅ `encode()` / `decode()` for tokenization

What's missing: a draft model, the speculative loop, and the rejection sampling logic.

---

## Hint 1: The Draft Model — A Bigram Table

At 210K params, your GPT is already tiny. A "smaller GPT" would be absurd. Instead, use a **bigram model**: a simple lookup table that predicts the next token based only on the current token.

```python
class BigramDraftModel:
    """
    Draft model for speculative decoding.
    Predicts P(next_token | current_token) from training data statistics.
    """
    def __init__(self, train_data, vocab_size, device):
        # Count bigram frequencies: how often does token B follow token A?
        counts = torch.zeros(vocab_size, vocab_size, device=device)
        for i in range(len(train_data) - 1):
            counts[train_data[i], train_data[i + 1]] += 1

        # Convert counts to probabilities (add smoothing to avoid zeros)
        counts += 1  # Laplace smoothing
        self.probs = counts / counts.sum(dim=1, keepdim=True)  # (vocab_size, vocab_size)

    def get_probs(self, token_id):
        """Return P(next | token_id) as a (vocab_size,) distribution."""
        return self.probs[token_id]

    def sample(self, token_id):
        """Sample one token given the current token."""
        probs = self.get_probs(token_id)
        return torch.multinomial(probs, num_samples=1).item(), probs
```

**Why bigram?** It's essentially free — a single table lookup vs. a full transformer forward pass. The draft quality will be mediocre (bigrams can't capture long-range dependencies), but that's fine. Even a 30% acceptance rate means you sometimes get 2-3 tokens per target forward pass instead of 1.

**Why Laplace smoothing?** Without it, some bigram entries are zero. If the draft assigns probability 0 to a token that the target model wants, the rejection sampling math breaks (division by zero).

---

## Hint 2: The Speculative Decode Loop

The core algorithm has three phases per step: **draft**, **verify**, **accept/reject**.

```python
def speculative_generate(target_model, draft_model, prompt_tokens, max_new_tokens, K=4):
    """
    Generate tokens using speculative decoding.

    Args:
        target_model:  your GPTLanguageModel (the "real" model)
        draft_model:   BigramDraftModel (the cheap guesser)
        prompt_tokens: list of ints — the encoded prompt
        max_new_tokens: how many tokens to generate
        K:             number of draft tokens per speculation step (gamma in the paper)

    Returns:
        generated_tokens: list of ints
    """
    generated = []
    # You'll need to manage the target model's KV cache here.
    # Start by prefilling the prompt through the target model.

    while len(generated) < max_new_tokens:
        # 1. DRAFT: generate K candidate tokens using the cheap model
        # 2. VERIFY: run all K candidates through the target model in ONE forward pass
        # 3. ACCEPT/REJECT: compare draft vs target probabilities, accept or resample

        # ... (see Hints 3-5 for details)
        pass

    return generated[:max_new_tokens]
```

**The K parameter (speculation length):** This is how many tokens the draft model guesses ahead. Typical values are 3-5. Too small = not enough speedup. Too large = most guesses are rejected (wasting draft compute). At nanoGPT scale with a bigram draft, K=4 is a good starting point.

---

## Hint 3: The Draft Phase — Generating K Candidates

The draft phase is simple: autoregressively sample K tokens from the bigram model, storing the draft probabilities for each.

```python
def draft_tokens(draft_model, current_token, K):
    """
    Generate K speculative tokens from the draft model.

    Returns:
        candidates:  list of K token ids
        draft_probs: list of K probability distributions (each is (vocab_size,))
    """
    candidates = []
    draft_probs = []
    tok = current_token

    for _ in range(K):
        next_tok, probs = draft_model.sample(tok)
        candidates.append(next_tok)
        draft_probs.append(probs)
        tok = next_tok

    return candidates, draft_probs
```

This is nearly instant — K table lookups. No GPU compute.

---

## Hint 4: The Verification Phase — One Target Forward Pass

This is the key insight: **the target model can verify all K candidates in a single forward pass** because of the causal attention structure.

You feed the target model the sequence `[current_token, candidate_0, candidate_1, ..., candidate_{K-1}]` as a `(1, K+1)` input. The model produces logits at every position. The logit at position `i` gives the target model's distribution for what should come *after* position `i`.

```python
def verify_candidates(target_model, current_token, candidates, past_kvs):
    """
    Run the target model on [current_token] + candidates in one forward pass.

    Returns:
        target_probs: list of K+1 probability distributions
        new_kvs:      updated KV cache
    """
    # Build input: [current_token, c0, c1, ..., c_{K-1}]
    all_tokens = [current_token] + candidates
    input_ids = torch.tensor([all_tokens], dtype=torch.long, device=device)

    # Position indices continue from where the cache left off
    cache_len = 0  # number of tokens already in the KV cache
    if past_kvs is not None:
        # Get T_past from the first layer, first head's key tensor
        cache_len = past_kvs[0][0][0].shape[1]
    positions = torch.arange(cache_len, cache_len + len(all_tokens), device=device).unsqueeze(0)

    # Single forward pass!
    logits, _, new_kvs = target_model(input_ids, pos=positions, past_kvs=past_kvs)
    # logits shape: (1, K+1, vocab_size)

    # Convert to probabilities
    target_probs = []
    for i in range(len(all_tokens)):
        probs = F.softmax(logits[0, i, :], dim=-1)
        target_probs.append(probs)

    return target_probs, new_kvs
```

**Why does this work?** Position `i` in the output only attends to positions `≤ i` (causal masking). So:
- `target_probs[0]` = P(next | prompt, current_token) — what the target thinks should follow `current_token`
- `target_probs[1]` = P(next | prompt, current_token, c0) — what should follow `c0`
- `target_probs[i]` = P(next | prompt, current_token, c0, ..., c_{i-1}) — what should follow `c_{i-1}`

This lets you check each candidate against what the target model actually wanted.

---

## Hint 5: The Accept/Reject Phase — Rejection Sampling

This is the mathematically precise part. For each draft token, you decide whether to accept it based on how well the draft model's prediction matches the target model's prediction.

```python
def accept_reject(candidates, draft_probs, target_probs):
    """
    Apply rejection sampling to decide which draft tokens to accept.

    Args:
        candidates:   list of K draft token ids
        draft_probs:  list of K draft probability distributions
        target_probs: list of K+1 target probability distributions
                      (target_probs[i] is the target's distribution for position i)

    Returns:
        accepted_tokens: list of accepted tokens (1 to K+1 tokens)
    """
    accepted = []

    for i in range(len(candidates)):
        token = candidates[i]
        q = draft_probs[i][token]    # draft model's probability for this token
        p = target_probs[i][token]   # target model's probability for this token

        # Accept with probability min(1, p/q)
        if torch.rand(1, device=p.device).item() < (p / q).clamp(max=1.0).item():
            accepted.append(token)
        else:
            # Rejected! Resample from the adjusted distribution
            # The adjusted distribution ensures we match the target exactly
            adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            adjusted = adjusted / adjusted.sum()
            resampled = torch.multinomial(adjusted, num_samples=1).item()
            accepted.append(resampled)
            return accepted  # Stop here — don't check further candidates

    # All K candidates accepted! Sample one bonus token from the target
    bonus = torch.multinomial(target_probs[len(candidates)], num_samples=1).item()
    accepted.append(bonus)
    return accepted
```

### Why This Works — The Intuition

Think of it as a filter:

- **p ≥ q** (target likes the token at least as much as draft): Accept always. The target agrees with or is even more confident than the draft.
- **p < q** (target likes it less than draft): Accept with probability `p/q`. The bigger the disagreement, the more likely we reject.
- **On rejection**: We don't just use the target's raw distribution. We use `max(0, p - q)` normalized — this is the "residual" probability mass that the target wanted but the draft didn't provide. This mathematical trick is what guarantees the output distribution is identical to the target model alone.

### The Bonus Token

If ALL K candidates are accepted, we get a free extra token from `target_probs[K]` — the target's prediction for position K+1. This means a perfect speculation step yields K+1 tokens.

---

## Hint 6: KV Cache Management

The tricky part: after accept/reject, you need to **trim the KV cache** to match only the accepted tokens. The verification forward pass cached KV entries for all K+1 tokens, but if you only accepted 2 of them, the cache entries for positions 3, 4, 5 are invalid.

```python
def trim_kv_cache(new_kvs, num_accepted, cache_len_before_verify):
    """
    Trim the KV cache to only include accepted tokens.

    After verification, the cache has entries for all K+1 speculative tokens.
    We need to keep only the first `num_accepted` new entries.

    Args:
        new_kvs:     KV cache from the verification forward pass
        num_accepted: how many tokens were accepted
        cache_len_before_verify: KV cache length before the verify call
    """
    keep = cache_len_before_verify + num_accepted
    trimmed = []
    for layer_kv in new_kvs:
        layer_trimmed = []
        for (k, v) in layer_kv:
            layer_trimmed.append((k[:, :keep, :], v[:, :keep, :]))
        trimmed.append(layer_trimmed)
    return trimmed
```

**Why trim?** If you accepted tokens [A, B] but rejected C (and resampled C'), your KV cache from the verify pass contains entries computed assuming the sequence was [..., A, B, C]. But the actual sequence is [..., A, B, C']. The KV entries for C are wrong — they were computed with the wrong token. You must discard them.

The resampled token C' will be processed in the **next** speculation step, where it gets a fresh KV entry.

---

## Hint 7: Putting It All Together

```python
@torch.no_grad()
def speculative_generate(target_model, draft_model, prompt_tokens, max_new_tokens, K=4):
    target_model.eval()
    generated = []

    # 1. Prefill: run the full prompt through the target model
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), device=device).unsqueeze(0)
    logits, _, past_kvs = target_model(input_ids, pos=positions)

    # Sample the first token from the prefill output
    probs = F.softmax(logits[0, -1, :], dim=-1)
    current_token = torch.multinomial(probs, num_samples=1).item()
    generated.append(current_token)

    # 2. Speculative decode loop
    while len(generated) < max_new_tokens:
        cache_len = past_kvs[0][0][0].shape[1]  # current KV cache length

        # How many tokens to speculate (don't overshoot max_new_tokens)
        k = min(K, max_new_tokens - len(generated))

        # DRAFT
        candidates, draft_probs = draft_tokens(draft_model, current_token, k)

        # VERIFY
        target_probs, new_kvs = verify_candidates(
            target_model, current_token, candidates, past_kvs
        )

        # ACCEPT/REJECT
        accepted = accept_reject(candidates, draft_probs, target_probs)

        # TRIM KV CACHE (keep only accepted tokens)
        past_kvs = trim_kv_cache(new_kvs, len(accepted), cache_len)

        # Update state
        generated.extend(accepted)
        current_token = accepted[-1]

    return generated[:max_new_tokens]
```

---

## Hint 8: Measuring the Speedup

The metric that matters is **acceptance rate** — what fraction of draft tokens get accepted on average.

```python
def benchmark_speculative(target_model, draft_model, prompts, max_new_tokens=50, K=4):
    """Compare speculative vs standard decoding."""

    # Count target model forward passes
    target_calls_standard = 0
    target_calls_speculative = 0
    total_tokens = 0
    total_accepted = 0
    total_drafted = 0

    # ... run both approaches, count forward passes ...

    print(f"Standard decoding: {target_calls_standard} target forward passes")
    print(f"Speculative decoding: {target_calls_speculative} target forward passes")
    print(f"Acceptance rate: {total_accepted / total_drafted:.1%}")
    print(f"Tokens per target call: {total_tokens / target_calls_speculative:.2f}")
    print(f"Speedup: {target_calls_standard / target_calls_speculative:.2f}x")
```

With a bigram draft on Shakespeare text, expect:
- **Acceptance rate**: ~20-40% (bigrams are weak predictors)
- **Tokens per target call**: ~1.3-1.6
- **Speedup**: ~1.2-1.5x in forward pass count (actual wall-clock speedup depends on model size)

---

## Test Scenarios

### Test 1: Output equivalence (greedy)

With greedy decoding (argmax instead of sampling), speculative decoding should produce **exactly the same output** as standard decoding. This validates correctness.

```python
# Use temperature=0 (greedy) for deterministic comparison
standard_output = greedy_generate(target_model, prompt, max_new_tokens=50)
spec_output = speculative_generate_greedy(target_model, draft_model, prompt, max_new_tokens=50)
assert standard_output == spec_output, "Speculative output differs from standard!"
```

### Test 2: Rejection sampling correctness

When the draft model and target model are the **same model**, every token should be accepted (p == q always, so acceptance probability is 1).

```python
# Use target model as its own draft — should accept everything
# This validates your rejection sampling math
```

### Test 3: All-reject scenario

Use a draft model that always predicts a uniform distribution. Most tokens should be rejected, but the output should still be valid (resampled from the target).

### Test 4: Acceptance rate tracking

Verify that your acceptance rate counter matches manual counting.

---

## Summary of New Components

| Component | What It Does |
|-----------|-------------|
| `BigramDraftModel` | Cheap next-token predictor from bigram statistics |
| `draft_tokens()` | Generate K candidates from the draft model |
| `verify_candidates()` | Run K+1 tokens through target model in one forward pass |
| `accept_reject()` | Rejection sampling to decide which tokens to keep |
| `trim_kv_cache()` | Discard KV entries for rejected tokens |
| `speculative_generate()` | Main loop tying it all together |

**What doesn't change:** `Head`, `MultiHeadAttention`, `Block`, `GPTLanguageModel`, training.

---

## Recommended Build Order

```
1. BigramDraftModel              ← build from training data, verify probs sum to 1
2. draft_tokens()                ← sample K tokens, store draft probs
3. verify_candidates()           ← single target forward pass with KV cache
4. accept_reject()               ← rejection sampling (hardest part conceptually)
5. trim_kv_cache()               ← trim to accepted length
6. speculative_generate()        ← main loop
7. Test: greedy equivalence      ← validate correctness
8. Benchmark: acceptance rate    ← measure speedup
```

---

## Gotchas

1. **Don't forget to handle the `current_token` in the verify input.** The verify pass processes `[current_token, c0, c1, ..., c_{K-1}]` — that's K+1 tokens, not K. The `current_token` is needed so the target model produces its own prediction for position 0, which you compare against `c0`.

2. **KV cache trimming is critical.** If you skip it, the cache contains KV entries computed with rejected tokens. The next speculation step will attend to those wrong entries and produce garbage.

3. **The adjusted distribution on rejection must be non-negative.** `torch.clamp(target_probs - draft_probs, min=0)` handles this, but if the result sums to 0 (extremely rare), fall back to the target distribution directly.

4. **Position indices must be contiguous with the cache.** After trimming the cache to length `L`, the next verify pass must use positions starting at `L`.

5. **Greedy decoding simplifies rejection sampling.** With argmax (temperature=0), acceptance becomes a simple equality check: did the draft pick the same token as the target's argmax? This is much easier to debug with — start here before implementing full rejection sampling.

6. **The `block_size=32` position embedding limit still applies.** Total sequence length (prompt + generated) can't exceed 32. Keep prompts short and `max_new_tokens` modest.
