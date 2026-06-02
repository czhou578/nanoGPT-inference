# Adaptive Speculative Decoding -- Implementation Plan & Hints

This plan builds on [`nanogpt-trigram-spec-decode.py`](../../nanogpt-trigram-spec-decode.py) and the trigram benchmark in [`benchmarks/trigram_speculative_decoding.py`](../../benchmarks/trigram_speculative_decoding.py).

The goal is to replace a fixed speculation length `K` with an adaptive `K` that changes during generation based on recent draft quality.

Keep this as the guiding idea:

```text
If the draft is being accepted often, speculate farther.
If the draft is being rejected often, speculate less.
```

Do not change the target verification math. Do not change the accept/reject distribution correction. Adaptive speculative decoding should only decide **how many draft tokens to propose next**.

---

## The Problem You're Solving

The current trigram speculative decoder uses a fixed `K`:

```text
K = 2, 4, or 6 for the whole run
```

The benchmark results show the tradeoff clearly:

| K | What Happens |
|---|---|
| Small K | Higher acceptance, less wasted verification work, but fewer chances to emit many tokens per target call. |
| Large K | Fewer target calls, but lower acceptance and more extra target-token work. |

There is no single best `K` for every point in generation.

Some contexts are easy for the trigram draft:

```text
"th" -> "e"
"qu" -> "e"
", " -> common next letters
```

Other contexts are uncertain:

```text
rare names
line breaks
punctuation transitions
low-count trigrams
```

Adaptive speculation lets each request respond to local draft quality.

---

## What Should Stay Fixed

Keep these pieces unchanged:

- target `GPTLanguageModel`
- prompt prefill
- KV-cache update and trimming
- verification input shape: `[current_token] + candidates`
- target probability computation
- accept/reject logic
- bonus token behavior
- resampled token behavior

The correctness guarantee depends on the draft probabilities matching the distribution that sampled each candidate. Adaptive K is safe because it only changes the number of candidates.

Hint:

> Changing K is scheduling. Changing `accept_reject()` is math. Be much more careful with the math.

---

## What Should Change

Add a small controller that chooses `K` each speculative iteration.

Current loop:

```text
remaining = max_new_tokens - generated
k = min(fixed_K, remaining)
draft k tokens
verify
accept/reject
repeat
```

Adaptive loop:

```text
remaining = max_new_tokens - generated
k = controller.choose_k(remaining, recent_stats)
draft k tokens
verify
accept/reject
controller.observe(result)
repeat
```

The controller does not need to know anything about transformer internals. It only watches outcomes.

---

## Hint 1: Track Per-Request Adaptive State

Start with per-request state, not global state.

Why:

- Different prompts have different predictability.
- A request may move through easy and hard text regions.
- Global K can overreact to one request and hurt another.

Useful state:

| Field | Meaning |
|---|---|
| `current_k` | Speculation length to try next. |
| `min_k` | Lower bound, usually `1` or `2`. |
| `max_k` | Upper bound, maybe `6` or `8` for this tiny model. |
| `recent_accepts` | Number of accepted draft tokens in the recent window. |
| `recent_proposed` | Number of proposed draft tokens in the recent window. |
| `recent_resamples` | Recent rejection/correction count. |
| `recent_bonus` | Recent full-chain accept count. |
| `verify_steps` | Number of speculative verification steps observed. |

Hint:

> Put this state next to request state in the benchmark first. Once it works there, port the same idea into `nanogpt-trigram-spec-decode.py`.

---

## Hint 2: Start With A Simple Window

Use a small rolling window over the last few verification steps.

Good starting points:

```text
window_size = 8 verification steps
initial_k = 4
min_k = 1
max_k = 6
```

Avoid adapting after every single token with no smoothing. One rejection can happen by chance. A window prevents K from bouncing around too much.

Hint:

> You want K to feel like a thermostat, not a light switch.

---

## Hint 3: Use Acceptance Rate First

The simplest signal is:

```text
acceptance_rate = accepted_draft_tokens / proposed_draft_tokens
```

Possible thresholds:

```text
if acceptance_rate > 0.60:
    increase K
elif acceptance_rate < 0.30:
    decrease K
else:
    keep K
```

For your trigram results, acceptance was roughly **25.7% to 39.9%**, so these thresholds may be too optimistic. A more realistic first pass for the trigram table might be:

```text
high_accept = 0.40
low_accept = 0.25
```

Hint:

> Choose thresholds from your own benchmark results, not from what production papers use for neural draft models.

---

## Hint 4: Watch Tokens Per Verify, Not Just Acceptance

Acceptance rate can be misleading by itself.

Example:

```text
K=2 accepts 50% -> emits about 1-2 tokens per verify
K=6 accepts 25% -> may still emit more tokens per verify
```

Track:

```text
tokens_emitted_this_verify = len(accepted_tokens)
avg_tokens_per_verify = recent_emitted / recent_verify_steps
```

This is close to the metric printed as `avg_verify` in your benchmark.

Good adaptation rule:

```text
increase K only if acceptance is decent AND avg_tokens_per_verify is improving
decrease K if resamples are common AND avg_tokens_per_verify is low
```

Hint:

> The controller's real job is not to maximize acceptance. It is to maximize useful tokens per target call without creating too much wasted verification work.

---

## Hint 5: Use Hysteresis

Do not let K increase and decrease on adjacent steps.

Add small guardrails:

```text
only adapt after at least N verify steps
only change K by 1 at a time
wait M steps after changing K before changing again
```

Suggested defaults:

```text
adapt_every = 4 verify steps
cooldown = 2 verify steps
step_size = 1
```

Hint:

> If K changes too often, your benchmark will be harder to interpret and the controller may chase random sampling noise.

---

## Hint 6: Cap By Remaining Tokens

Even if the controller wants `K=6`, the request may only need 2 more tokens.

Always compute:

```text
k = min(controller.current_k, remaining)
```

Also think about context length:

```text
remaining_context = block_size - cache_len
k = min(k, remaining_context - 1)
```

The `-1` is because verification uses:

```text
[current_token] + candidates
```

Hint:

> A good adaptive controller should never be able to request an invalid verification shape.

---

## Hint 7: Consider Draft Confidence

Acceptance is a delayed signal. Draft confidence is available before verification.

Cheap confidence signals:

| Signal | Meaning |
|---|---|
| max draft probability | High if the trigram table strongly prefers one next token. |
| entropy | Low entropy means concentrated distribution. |
| context count | High trigram count means this two-token context was seen often. |
| fallback used | If using bigram/unigram fallback, confidence is lower. |

Possible use:

```text
if trigram context count is low:
    do not increase K
if entropy is high:
    cap K at 2 or 3
```

Hint:

> Start with acceptance-based adaptation. Add confidence only after you have a working baseline.

---

## Hint 8: Add A Trigram Backoff Before Getting Fancy

Your current benchmark trigram table uses smoothing. For rare contexts, that can be weak.

Before making the adaptive controller complicated, consider improving draft quality:

```text
if trigram context count >= threshold:
    use trigram
elif bigram context exists:
    use bigram
else:
    use unigram
```

This can make adaptive K more meaningful because the controller can trust context-count information.

Hint:

> Better draft probabilities often help more than clever K tuning.

---

## Hint 9: Add Metrics Before Tuning

Add adaptive-specific metrics to the benchmark output.

Useful columns:

| Metric | Meaning |
|---|---|
| `avg_k` | Average chosen K. |
| `min_k` / `max_k` | Range actually used. |
| `k_changes` | Number of times controller changed K. |
| `k_up` / `k_down` | Direction of changes. |
| `accept_by_k` | Acceptance rate for each K value. |
| `verify_by_k` | Avg emitted tokens per verify for each K. |
| `target_tok_by_k` | Target-token work caused by each K. |

Hint:

> Without `accept_by_k`, you will not know whether adaptive K chose well or just got lucky.

---

## Hint 10: Benchmark Against Fixed K

Do not only compare adaptive speculative decoding against KV decode.

Compare:

```text
kv_decode
fixed_k2_trigram
fixed_k4_trigram
fixed_k6_trigram
adaptive_trigram
```

The real question is:

```text
Does adaptive K beat the best fixed K, or only beat KV decode?
```

Your current fixed-K results show:

```text
K=2: highest acceptance, strong ratio but slower absolute spec tok/s than K=6
K=4: balanced
K=6: best clean absolute throughput, but high target-token overhead
```

Hint:

> Adaptive decoding is useful only if it avoids bad fixed-K choices while matching or beating the good ones.

---

## Hint 11: Start In The Benchmark File

Implement adaptive decoding first in:

```text
benchmarks/trigram_speculative_decoding.py
```

Then port the working logic into:

```text
nanogpt-trigram-spec-decode.py
```

Why:

- The benchmark already has clean request state.
- It already tracks acceptance, proposed tokens, bonus tokens, and resamples.
- You can compare fixed K and adaptive K in one place.
- You avoid mixing experimental controller logic with the demo script too early.

Hint:

> Treat the benchmark as your lab bench. Once the controller behaves, move it into the main script.

---

## Hint 12: Keep The First Version Boring

A good first adaptive controller:

```text
initial_k = 4
min_k = 1
max_k = 6
window = last 8 verifies
if enough observations and not in cooldown:
    if acceptance_rate >= 0.40 and avg_tokens_per_verify >= current_k * 0.55:
        K += 1
    elif acceptance_rate <= 0.25 or avg_tokens_per_verify <= 1.3:
        K -= 1
```

This is intentionally simple. Tune later.

Hint:

> A boring controller with good metrics is better than a clever controller you cannot explain.

---

## Common Mistakes To Avoid

### Mistake 1: Counting Resampled Tokens As Accepted Draft Tokens

A resampled token is emitted, but it was not accepted from the draft.

Keep separate:

```text
accepted_draft_tokens
resampled_tokens
bonus_tokens
emitted_tokens
```

### Mistake 2: Adapting Based On Total Generated Tokens Only

A step can emit tokens because of resampling even when the draft was bad.

Use draft acceptance and verify efficiency, not just generated token count.

### Mistake 3: Forgetting Remaining Tokens

Near the end of a request, K naturally shrinks because fewer output tokens remain.

Do not interpret these clipped final steps as evidence that large K is bad.

### Mistake 4: Changing Draft Probabilities After Sampling

The accept/reject math requires:

```text
draft_probs[i] == distribution that sampled candidates[i]
```

If you apply temperature, fallback, smoothing, or confidence adjustments, store the final distribution actually used.

### Mistake 5: Comparing Single Runs Too Strongly

Speculative decoding is stochastic. Adaptive K is even more stochastic.

Use multiple seeds before claiming the controller is better.

---

## Suggested Build Order

1. Add an `AdaptiveKController` concept in the benchmark file.
2. Track per-request `current_k`.
3. Feed `current_k` into the existing trigram draft step.
4. After accept/reject, record proposed, accepted, resampled, bonus, and emitted tokens.
5. Update the controller from the recent window.
6. Print adaptive metrics: `avg_k`, `k_changes`, `accept_by_k`.
7. Add a benchmark suite row called `adaptive_trigram`.
8. Compare against fixed K=2, K=4, and K=6.
9. Tune thresholds only after the metrics are visible.
10. Port the controller into `nanogpt-trigram-spec-decode.py`.

---

## Success Criteria

The adaptive version is successful if it:

- completes the same number of requests and generated tokens,
- preserves the same accept/reject correctness path,
- reduces target calls compared with KV decode,
- avoids excessive target-token overhead compared with fixed K=6,
- matches or beats fixed K=4/K=6 throughput over multiple seeds,
- reports enough metrics to explain why K changed.

Hint:

> If adaptive K is not faster yet but its K choices are explainable, that is still a useful milestone.

---

## Final Mental Model

Fixed-K speculative decoding asks:

```text
How far should every request speculate?
```

Adaptive speculative decoding asks:

```text
How far should this request speculate right now?
```

That is the whole upgrade.
