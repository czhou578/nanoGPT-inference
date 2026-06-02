# Trigram Draft Table for Speculative Decoding -- Implementation Plan & Hints

## The Problem You're Solving

The current speculative decoding implementation uses a **bigram draft model**:

```text
P(next_token | current_token)
```

That is cheap, but it only sees one token of context. For character-level Tiny Shakespeare,
many next-token choices depend on at least two previous characters:

```text
"th" -> "e" is much stronger than "h" -> "e"
"qu" -> "e" is much stronger than "u" -> "e"
"\n\n" -> capital letter is stronger than "\n" alone
```

A **trigram table** upgrades the draft model to:

```text
P(next_token | previous_token, current_token)
```

The speculative decoding algorithm does not change. The target model still verifies the
draft tokens and preserves the target distribution. Only the cheap draft distribution `q`
gets better, which should raise acceptance rate and improve tokens emitted per verification
step.

---

## What Changes

Keep these pieces unchanged:

- `verify_candidates()`
- `accept_reject()`
- `trim_kv_cache()`
- the target `GPTLanguageModel`
- the KV-cache invariant
- the benchmark comparison between `kv_decode` and `spec_decode`

Change the draft model and the draft phase:

| Current | New |
|---------|-----|
| `BigramDraftModel` | `TrigramDraftModel` |
| `get_probs(token_id)` | `get_probs(prev_token_id, token_id)` |
| `sample(token_id)` | `sample(prev_token_id, token_id)` |
| draft loop tracks one rolling token | draft loop tracks two rolling tokens |

The important idea: the verifier only cares that `draft_probs[i]` is the exact distribution
that produced `candidates[i]`. If the draft uses two-token context, store that two-token
distribution in `draft_probs[i]` and the rejection math remains valid.

---

## Hint 1: Build Counts for `P(c | a, b)`

The direct trigram count tensor is:

```python
counts[a, b, c] += 1
```

For this repo's character-level vocabulary, `vocab_size` is small, so a dense
`(vocab_size, vocab_size, vocab_size)` tensor is fine.

```python
class TrigramDraftModel:
    """
    Draft model for speculative decoding.
    Predicts P(next_token | prev_token, current_token) from training data.
    """

    def __init__(self, token_ids, vocab_size, device, fallback_bigram=None):
        counts = torch.zeros(vocab_size, vocab_size, vocab_size, dtype=torch.float32)

        ids = torch.as_tensor(token_ids, dtype=torch.long).flatten().cpu()
        for a, b, c in zip(ids[:-2].tolist(), ids[1:-1].tolist(), ids[2:].tolist()):
            if 0 <= a < vocab_size and 0 <= b < vocab_size and 0 <= c < vocab_size:
                counts[a, b, c] += 1.0

        # Save row mass before smoothing so sparse contexts can fall back.
        self.context_counts = counts.sum(dim=-1)
        counts += 1.0
        self.probs = counts / counts.sum(dim=-1, keepdim=True)
        self.probs = self.probs.to(device)
        self.context_counts = self.context_counts.to(device)
        self.fallback_bigram = fallback_bigram
```

### Why Not Just Smooth Everything?

Laplace smoothing makes every trigram context legal, but it also makes unseen two-token
contexts nearly uniform. A uniform draft is safe, but weak. For sparse contexts, a bigram
fallback is usually better:

```text
If (prev, current) was seen often: use P(next | prev, current)
If (prev, current) was rare/unseen: use P(next | current)
```

This keeps the trigram model from becoming overconfident on tiny counts.

---

## Hint 2: Add a Bigram Fallback

The fallback can reuse the existing `BigramDraftModel`, or the trigram model can build its
own bigram table. Reusing the existing class is the smallest implementation.

```python
def get_probs(self, prev_token_id, token_id, temperature=1.0):
    prev_token_id = int(prev_token_id)
    token_id = int(token_id)

    # Tune this threshold. 1 means "use trigram if the context ever appeared".
    min_context_count = 2
    if self.context_counts[prev_token_id, token_id] < min_context_count:
        probs = self.fallback_bigram.get_probs(token_id, temperature=temperature)
    else:
        probs = self.probs[prev_token_id, token_id]

    if temperature == 1.0:
        return probs

    scaled = probs.clamp_min(1e-12).pow(1.0 / temperature)
    return scaled / scaled.sum()

def sample(self, prev_token_id, token_id, *, temperature=1.0, generator=None):
    probs = self.get_probs(prev_token_id, token_id, temperature=temperature)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator).item()
    return next_token, probs
```

**Hint:** make `BigramDraftModel.get_probs()` accept `temperature` first if the benchmark
version already does. In `nanogpt-spec-decode.py`, it currently does not, so either add the
optional parameter there or keep the fallback call temperature-free in that file.

---

## Hint 3: Draft Tokens Need Two Starting Tokens

The draft phase now needs `(prev_token, current_token)`, not just `current_token`.

```python
def draft_tokens(draft_model, prev_token, current_token, K):
    candidates = []
    draft_probs = []

    a = prev_token
    b = current_token

    for _ in range(K):
        next_tok, probs = draft_model.sample(a, b)
        candidates.append(next_tok)
        draft_probs.append(probs)

        # Slide the two-token window forward.
        a, b = b, next_tok

    return candidates, draft_probs
```

For the benchmark helper:

```python
def _draft_tokens(draft_model, prev_token, current_token, k, *, temperature, generator):
    candidates = []
    draft_probs = []
    a, b = prev_token, current_token

    for _ in range(k):
        next_token, probs = draft_model.sample(
            a,
            b,
            temperature=temperature,
            generator=generator,
        )
        candidates.append(next_token)
        draft_probs.append(probs)
        a, b = b, next_token

    return candidates, draft_probs
```

The returned `draft_probs[i]` still has shape `(vocab_size,)`, so `accept_reject()` does not
need to know whether the distribution came from a bigram or trigram table.

---

## Hint 4: Track `prev_token` in Request State

The current speculative loop tracks the last emitted token as `current_token` or
`req.last_token`. A trigram draft needs one more piece of state.

For the single-request script:

```python
# After prompt prefill and first sampled token:
prev_token = prompt_tokens[-1]
current_token = first_generated_token

# After accept/reject:
for tok in accepted:
    prev_token, current_token = current_token, tok
```

For the benchmark request state, add:

```python
prev_token: int | None = None
last_token: int | None = None
```

Then update the token recorder:

```python
def _record_token(req, token_id):
    now = time.perf_counter()
    if req.last_token is not None:
        req.prev_token = req.last_token
    req.generated_tokens.append(int(token_id))
    req.last_token = int(token_id)
    if req.first_token_at_s is None:
        req.first_token_at_s = now
```

During prefill, before recording the first sampled token:

```python
req.prev_token = req.spec.prompt_tokens[-1]
_record_token(req, first_token.item())
```

That gives every speculation step a valid `(prev_token, last_token)` context.

---

## Hint 5: Handle One-Token Prompts

Trigram drafting needs two previous tokens. If a prompt has fewer than two tokens, use the
bigram fallback until two tokens exist.

```python
if prev_token is None:
    candidates, draft_probs = draft_tokens_bigram(
        fallback_bigram,
        current_token,
        k,
    )
else:
    candidates, draft_probs = draft_tokens(
        trigram_model,
        prev_token,
        current_token,
        k,
    )
```

In this repo's benchmark workloads, prompts are likely longer than one character, but the
guard is worth adding because it keeps the draft model total.

---

## Hint 6: Update Benchmark Construction

In `benchmarks/speculative_decoding.py`, replace:

```python
draft_model = BigramDraftModel(
    token_ids,
    vocab_size,
    device,
    draft_noise=draft_noise,
)
```

with:

```python
bigram_fallback = BigramDraftModel(
    token_ids,
    vocab_size,
    device,
    draft_noise=draft_noise,
)
draft_model = TrigramDraftModel(
    token_ids,
    vocab_size,
    device,
    fallback_bigram=bigram_fallback,
    draft_noise=draft_noise,
)
```

If you keep `draft_noise`, blend after selecting the base trigram distribution:

```python
uniform = torch.full_like(probs, 1.0 / self.vocab_size)
probs = (1.0 - noise) * probs + noise * uniform
```

**Recommendation:** first implement trigram with `draft_noise=0.0`. Once correctness is
stable, add the noise path back so existing benchmark rows still run.

---

## Hint 7: Keep the Verification Input Exactly the Same

Do not change this part:

```python
verify_tokens = [req.last_token] + candidates
```

The target model verifies the same candidate sequence. The extra previous token is only for
the draft table's lookup. It is already represented in the target model's KV cache.

This is the most common mental slip:

```text
Draft context:  [prev_token, current_token] -> candidate_0
Verify input:                 [current_token, candidate_0, candidate_1, ...]
```

`prev_token` is not inserted into the verification input because it is already cached.

---

## Test Scenarios

### Test 1: Probability Rows Sum to 1

```python
draft = TrigramDraftModel(train_data, vocab_size, device, fallback_bigram=bigram)
row = draft.get_probs(encode("t")[0], encode("h")[0])
assert torch.allclose(row.sum(), torch.tensor(1.0, device=row.device))
```

### Test 2: Trigram Beats Bigram on Known Contexts

Pick common character contexts from Tiny Shakespeare:

```python
context = encode("th")
e = encode("e")[0]
trigram_p = trigram.get_probs(context[0], context[1])[e]
bigram_p = bigram.get_probs(context[1])[e]
print(trigram_p, bigram_p)
```

You do not need this assertion to always be true for every context, but it should be true for
some common contexts. This test proves the table is actually using two-token history.

### Test 3: Sparse Fallback Works

Use a rare or impossible two-token context. Confirm `get_probs(a, b)` returns the fallback
bigram distribution instead of a near-uniform smoothed trigram row.

```python
if trigram.context_counts[a, b] < trigram.min_context_count:
    assert torch.allclose(trigram.get_probs(a, b), bigram.get_probs(b))
```

### Test 4: Speculative Correctness Still Holds

Run a short generation with the same random seed using the bigram and trigram drafts. The
text does not need to match between draft models under sampling, but both should:

- terminate at `max_new_tokens`
- avoid NaNs in `accept_reject()`
- keep the KV cache length aligned with accepted tokens
- preserve the target model's rejection-sampling path

### Test 5: Benchmark Acceptance Rate

Run the same suite with bigram and trigram drafts:

```python
run_kv_vs_speculative_decoding_benchmark(..., speculation_len=4, draft_noise=0.0)
```

Track:

- `accept`
- `avg_verify`
- `target_calls`
- `target_tok`
- `gen_tok/s`

Expected result: the trigram draft should usually increase acceptance rate. It may or may
not improve wall-clock throughput on CPU because the model is tiny and the benchmark includes
Python overhead.

---

## Gotchas

1. **The draft distribution must match the sampled token.** If candidate `c_i` was sampled
   from `P(next | a, b)`, then `draft_probs[i]` must be that same row. Recomputing it with
   the wrong rolling context breaks rejection sampling.

2. **Do not feed `prev_token` into verification.** It is already in the target KV cache.
   Verification still starts from `[current_token] + candidates`.

3. **Sparse trigram rows can hurt.** A smoothed row from a context seen once may be worse
   than a bigram row seen thousands of times. Use `min_context_count`.

4. **Dense trigram storage is okay here, but not always.** Character vocab is tiny. For a
   50k-token LLM vocabulary, a dense trigram tensor would be impossible. Production n-gram
   proposers use sparse maps, suffix arrays, or retrieval over observed prompt/document
   spans.

5. **Keep probabilities strictly non-zero.** Rejection sampling divides by `q`. Smoothing,
   fallback, and `clamp_min(1e-12)` in accept/reject are all useful guardrails.

6. **Update state after every emitted token.** A rejection can emit a resampled token that
   was not in the candidate chain. The next draft context must use the actual emitted token,
   not the rejected candidate.

---

## Implementation Checklist & Order

**Step 1: Add `TrigramDraftModel`**
- Build `counts[a, b, c]`.
- Normalize into `self.probs[a, b]`.
- Track `self.context_counts[a, b]`.
- Wire in a bigram fallback and `min_context_count`.

**Step 2: Update Draft Sampling**
- Change `draft_tokens()` / `_draft_tokens()` to accept `prev_token` and `current_token`.
- Slide `(prev_token, current_token)` forward with every sampled candidate.
- Keep returning `candidates` and `draft_probs` in the same shape as before.

**Step 3: Track Previous Token State**
- Add `prev_token` to benchmark request state.
- Initialize it from the prompt before the first generated token.
- Update it whenever a token is recorded.

**Step 4: Swap the Benchmark Draft**
- Construct `BigramDraftModel` as fallback.
- Construct `TrigramDraftModel` as the active draft model.
- Keep benchmark metric names clear, e.g. `k4_trigram_draft`.

**Step 5: Validate Correctness**
- Test probability normalization.
- Test fallback behavior.
- Run a short speculative generation with a fixed seed.
- Confirm no changes were needed in verification or rejection sampling.

**Step 6: Compare Results**
- Run bigram and trigram rows with the same workload and seed.
- Compare acceptance rate first, throughput second.
- Add a short benchmark-writeup note if the trigram improves `accept` or `avg_verify`.

---

## Success Criteria

The trigram implementation is successful if:

- speculative decoding still generates exactly `max_new_tokens`
- no NaNs or zero-probability division issues appear
- `accept_reject()` remains unchanged or nearly unchanged
- the benchmark can report a `trigram_draft` row
- acceptance rate is higher than the bigram draft on at least one comparable workload

The best outcome is not just higher throughput. The learning goal is to show how draft
quality changes speculative decoding: a slightly smarter near-free proposer can reduce target
forward calls without changing the target model at all.
