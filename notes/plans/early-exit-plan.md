# Early Exit Heads — Implementation Plan

## Base File

**Start from:** [nanogpt-kv-cache.py](file:///home/colin-zhou/Projects/nanoGPT-inference/nanogpt-kv-cache.py)

This is the right base because:
- It has the KV cache infrastructure (prefill/decode split) needed for cache backfill
- It has a clean `Block` → `GPTLanguageModel` architecture without extra complexity (no batching, no scheduling, no paging)
- The `Head` class stores `key_cache`/`value_cache` per head, which we need to manipulate for early-exited tokens
- The `generate_kv_cache()` function provides the decode loop we'll modify

Do NOT use `nanogpt-fused-attention.py` or more complex files - those add complications (fused QKV, continuous batching) that obscure what early exit is doing.

**New file to create:** `nanogpt-early-exit.py`

See also: [notes/concepts/early-exit-heads.md](file:///home/colin-zhou/Projects/nanoGPT-inference/notes/concepts/early-exit-heads.md) for the conceptual background.

---

## The Problem You're Solving

In a standard transformer, every token - easy or hard - passes through every layer.
With 4 layers, the word "the" gets the same compute as a rare word completing a complex phrase.
This is wasteful.

```
Standard decode (current):
  token → Layer 0 → Layer 1 → Layer 2 → Layer 3 → lm_head → prediction
                                                     ↑
                                              4 layers always
```

With early exit heads, the model can bail out early when it's confident:

```
Early exit decode (this plan):
  token → Layer 0 → [exit_head_0] → confident? → YES → prediction (1 layer)
                         ↓ NO
          Layer 1 → [exit_head_1] → confident? → YES → prediction (2 layers)
                         ↓ NO
          Layer 2 → [exit_head_2] → confident? → YES → prediction (3 layers)
                         ↓ NO
          Layer 3 → lm_head → prediction (4 layers, same as before)
```

For a 4-layer model this saves at most 75% of compute per token.
In practice, common tokens (spaces, punctuation, frequent words) should consistently exit at layer 0 or 1, while rare tokens use the full depth.

---

## What You Already Have

From `nanogpt-kv-cache.py`:

- ✅ `Head` with per-head `key_cache` / `value_cache`
- ✅ `Block` with `sa` (MultiHeadAttention) + `ffwd` (FeedForward) + layer norms
- ✅ `GPTLanguageModel` with `self.blocks` (Sequential of Blocks), `self.lm_head`
- ✅ `generate_kv_cache()` - prefill/decode loop with KV cache
- ✅ `clear_kv_cache()` - resets all caches
- ✅ `start_pos` for correct positional embeddings during single-token decode

What's missing: exit heads, confidence checking, the training loss modification, KV cache backfill for early-exited tokens, and per-layer exit statistics tracking.

---

## Phase 1 — Model Architecture Changes

### 1.1 Exit Head Module

Add a lightweight prediction head after each transformer block.
Each exit head is a single linear layer that maps hidden states to vocabulary logits, identical in shape to `lm_head`:

```python
class ExitHead(nn.Module):
    """Lightweight prediction head at an intermediate layer."""
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)       # exit-specific layer norm
        self.linear = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        # x: (B, T, n_embd) — hidden state at this layer
        return self.linear(self.ln(x))        # (B, T, vocab_size)
```

Why include a LayerNorm?
The main `lm_head` operates on `ln_f(x)` - the final layer-normed hidden state.
Intermediate hidden states have different scale/distribution at each layer.
Without a per-exit LayerNorm, the linear projection would need to simultaneously learn the scale normalization and the token prediction, which hurts training.

### 1.2 Modified GPTLanguageModel

Change `self.blocks` from `nn.Sequential` to `nn.ModuleList` so we can run blocks one at a time and check confidence between them:

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # ModuleList instead of Sequential — we need per-layer control
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Exit heads — one per block (except the last, which uses lm_head)
        self.exit_heads = nn.ModuleList([
            ExitHead(n_embd, vocab_size) for _ in range(n_layer - 1)
        ])

        self.apply(self._init_weights)
```

Why `n_layer - 1` exit heads?
The last block feeds into `ln_f` → `lm_head`, which is already the "exit" at full depth.
Adding an exit head there would be redundant.

### 1.3 Modified Forward Pass

The forward pass must return exit head logits during training (for the joint loss) and support early exit during inference:

```python
def forward(self, idx, targets=None, start_pos=0,
            exit_threshold=None):
    """
    Args:
        exit_threshold: float or None.
            - None = run all layers (training mode, or no early exit)
            - float = confidence threshold for early exit during inference
              If max(softmax(exit_logits)) > threshold, return early.
    Returns:
        logits: (B, T, vocab_size) — from whichever layer exited
        loss: scalar or None
        exit_layer: int — which layer produced the output (0-indexed)
                    n_layer - 1 means "used all layers" (no early exit)
        all_exit_logits: list of (B, T, vocab_size) — exit head outputs
                         at each layer (only populated during training)
    """
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(
        torch.arange(start_pos, start_pos + T, device=device)
    )
    x = tok_emb + pos_emb

    all_exit_logits = []

    for layer_idx, block in enumerate(self.blocks):
        x = block(x)

        # Check exit heads (all layers except the last)
        if layer_idx < n_layer - 1:
            exit_logits = self.exit_heads[layer_idx](x)

            if self.training:
                all_exit_logits.append(exit_logits)

            elif exit_threshold is not None:
                # Confidence check: can we exit early?
                probs = F.softmax(exit_logits[:, -1, :], dim=-1)
                confidence = probs.max(dim=-1).values.item()
                if confidence > exit_threshold:
                    # Early exit!
                    return exit_logits, None, layer_idx, all_exit_logits

    # Full depth — use the standard lm_head
    x = self.ln_f(x)
    logits = self.lm_head(x)

    # Compute loss if targets provided (training)
    loss = None
    if targets is not None:
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

    return logits, loss, n_layer - 1, all_exit_logits
```

---

## Phase 2 — Training with Joint Loss

### 2.1 The Joint Loss Function

During training, the loss is a weighted sum of the final `lm_head` loss and all exit head losses:

```
L_total = L_final + Σ α_l * L_exit_l
```

All exit heads are trained to predict the same target as the final head.
This encourages intermediate representations to be "good enough" for prediction at every layer.

```python
def compute_joint_loss(model, idx, targets):
    """
    Run the full model, collect exit head logits, and compute
    the joint loss = final_loss + sum(alpha * exit_losses).
    """
    logits, _, _, all_exit_logits = model(idx, targets=targets)

    B, T, C = logits.shape
    final_loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

    # Weight for exit head losses — uniform for simplicity
    # Can tune: earlier exits get lower weight because they're harder
    alpha = 0.3  # each exit head contributes 0.3× the final loss

    exit_loss = 0.0
    for exit_logits in all_exit_logits:
        exit_loss += F.cross_entropy(
            exit_logits.view(B*T, C), targets.view(B*T)
        )

    total_loss = final_loss + alpha * exit_loss
    return total_loss
```

### 2.2 Training Loop Change

Replace the standard training loop's loss computation:

```python
# Before:
logits, loss = model(xb, yb)

# After:
loss = compute_joint_loss(model, xb, yb)
```

Everything else (optimizer, eval, etc.) stays the same.

### 2.3 What Alpha Controls

- `alpha = 0.0` — exit heads are never trained; they stay random.
  No early exit is possible.
- `alpha = 0.3` — moderate signal. Exit heads learn to predict but don't distort the main model's representations.
  This is the recommended starting point.
- `alpha = 1.0` — exit heads get equal weight. Intermediate layers are strongly pushed to be predictive.
  This can slightly hurt the final layer's quality because it constrains how freely layer representations can evolve.

---

## Phase 3 — Generation with Early Exit

### 3.1 The Confidence Threshold

During decode, after each block, the exit head's logits are converted to probabilities.
If `max(softmax(exit_logits))` exceeds the threshold, we exit early.

```python
exit_threshold = 0.9   # only exit if top token has >90% probability
```

Why 0.9?
- Lower threshold (0.7) → more aggressive early exit, more tokens skip layers, but higher risk of false exits
- Higher threshold (0.95) → conservative, few tokens exit early, but those that do are almost certainly correct
- 0.9 is a good starting point for a 4-layer model with a 65-character vocabulary

### 3.2 The KV Cache Problem

This is the hard part and the most interesting design challenge.

When a token exits at layer L, its KV cache only has entries for layers 0 through L.
But the NEXT token might need all 4 layers - and attention at layer 3 needs KV entries from ALL previous tokens at layer 3.

```
Token "t":  exits at layer 1
  KV cache: layer 0 ✓, layer 1 ✓, layer 2 ✗, layer 3 ✗

Token "h":  needs all 4 layers
  Layer 3 attention needs K/V from "t" at layer 3 — but it doesn't exist!
```

**Three strategies, in order of simplicity:**

#### Strategy A: Always backfill KV (recommended for this implementation)

After early exit, continue running the remaining layers on the hidden state to populate the KV cache, but DON'T use the deeper layers' prediction.
The prediction comes from the exit head; the deeper layers only run to fill KV.

```python
# Pseudocode for decode with backfill
x = tok_emb + pos_emb

for layer_idx, block in enumerate(self.blocks):
    x = block(x)  # populates KV cache as a side effect

    if layer_idx < n_layer - 1:
        exit_logits = self.exit_heads[layer_idx](x)
        probs = F.softmax(exit_logits[:, -1, :], dim=-1)
        confidence = probs.max(dim=-1).values.item()

        if confidence > exit_threshold:
            # Record that we exited early (for statistics),
            # but continue running remaining blocks to fill KV cache.
            early_exit_logits = exit_logits
            early_exit_layer = layer_idx

            # Run remaining blocks ONLY for KV cache population
            for remaining_block in self.blocks[layer_idx + 1:]:
                x = remaining_block(x)

            return early_exit_logits, early_exit_layer
```

Wait — this runs all layers anyway! Where's the savings?

The savings are real in production (large models):
1. The exit head's logits are returned immediately - the prediction is fast
2. The KV backfill can be done asynchronously on a separate CUDA stream
3. For batched decode, only the exited tokens need backfill; non-exited tokens already computed everything

For NanoGPT's tiny 4-layer model, the savings are small.
But the POINT is to demonstrate the mechanism and measure per-layer exit rates.

#### Strategy B: Skip KV entirely (simpler, quality trade-off)

Simply don't populate missing KV entries.
When a future token at layer 3 looks for token "t"'s KV at layer 3, the entry is missing.

Implementation: pad the KV cache with zeros for skipped layers.

```python
if exited_early:
    for remaining_layer in range(exit_layer + 1, n_layer):
        # Insert zero K/V entries so cache shape is consistent
        for head in self.blocks[remaining_layer].sa.heads:
            zero_k = torch.zeros_like(head.key_cache[:, :1, :])
            head.key_cache = torch.cat([head.key_cache, zero_k], dim=1)
            zero_v = torch.zeros_like(head.value_cache[:, :1, :])
            head.value_cache = torch.cat([head.value_cache, zero_v], dim=1)
```

This is simpler but degrades quality — attention at deep layers will see zero entries for early-exited tokens, effectively ignoring them.
Good for experimentation, bad for production.

**Recommendation:** Start with Strategy A (backfill). It's the most correct and demonstrates the real-world approach.
Once it works, experiment with Strategy B to see how much quality degrades.

### 3.3 Modified generate_kv_cache()

```python
def generate_early_exit(model, idx, max_new_tokens,
                        exit_threshold=0.9):
    """Generate with confidence-gated early exit."""
    model.eval()
    clear_kv_cache(model)

    # Track exit statistics
    exit_counts = [0] * n_layer  # how many tokens exited at each layer

    # Prefill: full forward pass (no early exit during prefill)
    logits, _, _, _ = model(idx)

    for step in range(max_new_tokens):
        logits_last = logits[:, -1, :]
        probs = F.softmax(logits_last, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        # Decode: single-token forward with early exit
        logits, _, exit_layer, _ = model(
            idx_next,
            start_pos=idx.shape[1] - 1,
            exit_threshold=exit_threshold,
        )
        exit_counts[exit_layer] += 1

    model.train()

    # Print exit statistics
    total = sum(exit_counts)
    print("\n--- Early Exit Statistics ---")
    for layer, count in enumerate(exit_counts):
        label = f"Layer {layer}" if layer < n_layer - 1 else f"Layer {layer} (full)"
        pct = 100.0 * count / total if total > 0 else 0
        print(f"  {label}: {count:4d} tokens ({pct:5.1f}%)")

    return idx
```

---

## Phase 4 — Benchmarking & Analysis

### 4.1 Exit Rate Distribution

The most important measurement: what fraction of tokens exit at each layer?

```
--- Early Exit Statistics ---
  Layer 0:   47 tokens (23.5%)   ← spaces, punctuation, "the"
  Layer 1:   31 tokens (15.5%)   ← common words
  Layer 2:   18 tokens ( 9.0%)   ← medium-frequency tokens
  Layer 3 (full):  104 tokens (52.0%)   ← everything else
```

If layer 0 + layer 1 captures >30% of tokens, early exit is working.
If almost everything falls through to the final layer, the threshold is too high or the exit heads didn't train well.

### 4.2 Threshold Sweep

Sweep `exit_threshold` from 0.5 to 0.99 and measure:

```python
thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
for t in thresholds:
    # Generate with this threshold
    # Record: exit rate per layer, output text, generation time
```

Plot:
- X-axis: threshold
- Y-axis (left): % of tokens that exit early (sum of layers 0 through n_layer-2)
- Y-axis (right): output quality (visual inspection or perplexity on a held-out set)

### 4.3 Compute Savings Estimation

For each generation run, compute the theoretical compute saving:

```python
# Weighted average layers used
avg_layers = sum(
    (layer + 1) * count for layer, count in enumerate(exit_counts)
) / sum(exit_counts)

compute_fraction = avg_layers / n_layer
speedup_estimate = n_layer / avg_layers

print(f"Average layers used: {avg_layers:.2f} / {n_layer}")
print(f"Theoretical compute: {100 * compute_fraction:.1f}%")
print(f"Theoretical speedup: {speedup_estimate:.2f}×")
```

### 4.4 Quality Comparison

Compare the output text with and without early exit to verify quality:

```python
# Same seed, same prompt
torch.manual_seed(42)
output_full = generate_kv_cache(model, prompt, 200)  # no early exit

torch.manual_seed(42)
output_exit = generate_early_exit(model, prompt, 200, exit_threshold=0.9)

# Compare token-by-token
matches = sum(a == b for a, b in zip(output_full, output_exit))
match_rate = matches / len(output_full)
print(f"Token match rate vs full model: {100*match_rate:.1f}%")
```

---

## Phase 5 — Stretch: Exit Heads as Draft Models

This is a natural connection to your speculative decoding work.

Instead of using a separate `BigramDraftModel`, use the layer-0 exit head as the draft "model":

- Draft: run layer 0 only, use exit_head_0 to predict K tokens
- Verify: run all 4 layers on the K candidates in one batched pass
- Accept/reject: standard rejection sampling

The advantage: the draft and target share KV cache for layer 0.
No separate draft model weights.
No separate draft model memory.

This is exactly how Medusa-style approaches work in production.

**Don't implement this in Phase 1.** Get basic early exit working first.
This is listed here for awareness and as a future extension.

---

## Implementation Checklist

```
Phase 1: Architecture
  [ ] Copy nanogpt-kv-cache.py → nanogpt-early-exit.py
  [ ] Add ExitHead class
  [ ] Change self.blocks to ModuleList
  [ ] Add self.exit_heads (n_layer - 1 heads)
  [ ] Modify forward() to return exit_layer and all_exit_logits
  [ ] Add confidence-gated early exit logic in forward()

Phase 2: Training
  [ ] Implement compute_joint_loss()
  [ ] Replace training loop loss with joint loss
  [ ] Train and verify exit heads produce reasonable distributions
  [ ] Experiment with alpha values (0.1, 0.3, 0.5, 1.0)

Phase 3: Generation
  [ ] Implement KV cache backfill (Strategy A)
  [ ] Implement generate_early_exit() with exit statistics tracking
  [ ] Print per-layer exit distribution after generation
  [ ] Verify output quality matches full-depth generation at high thresholds

Phase 4: Benchmarking
  [ ] Threshold sweep (0.5 → 0.99)
  [ ] Exit rate distribution plot
  [ ] Compute savings estimation
  [ ] Token match rate vs full model
  [ ] Wall-clock timing comparison (with and without early exit)
```

---

## Expected Behavior

With a 4-layer model trained on Shakespeare (65-char vocab, 57K params):

- **Spaces and punctuation** (`, . \n ! ? ;`) should exit at layer 0 with >90% confidence.
  These are highly predictable from local context.
- **Common words** (`the`, `and`, `of`, `to`) should exit at layer 0 or 1.
- **Less common words** and **mid-word continuations** should exit at layer 1 or 2.
- **Rare tokens** and **start-of-word positions** should use all 4 layers.

Expected exit distribution with `threshold=0.9`:

```
Layer 0: ~20-30%
Layer 1: ~10-20%
Layer 2: ~5-15%
Layer 3 (full): ~40-60%
```

This translates to roughly 60-80% of full compute on average, or a theoretical 1.25-1.7x speedup.
The actual wall-clock speedup on a tiny model will be smaller due to Python overhead dominating, but the exit distribution itself is the meaningful result.
