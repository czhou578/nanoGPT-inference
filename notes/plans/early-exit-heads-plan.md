# Early Exit Heads — Implementation Plan & Hints

## The Problem You're Solving

In a standard transformer, every token must pass through **all N layers** before producing a prediction. But not every token is equally hard to predict. After seeing "First Citi", the model probably knows "zen" is coming well before layer 6. Meanwhile "to be or not to ___" might genuinely need deep reasoning.

**Early exit heads** solve this by attaching a lightweight classifier at **every intermediate layer**. During inference, if an intermediate head's prediction is "confident enough" (above a threshold), you can skip the remaining layers entirely.

```
Layer 0: x₀ → exit_head₀(x₀) → confidence 0.3 → not confident, continue
Layer 1: x₁ → exit_head₁(x₁) → confidence 0.4 → not confident, continue
Layer 2: x₂ → exit_head₂(x₂) → confidence 0.92 → CONFIDENT, exit early!
Layer 3: (skipped)
Layer 4: (skipped)
Layer 5: (skipped)
```

Result: 50% compute savings for "easy" tokens. The final layer's head is still the normal `lm_head`, so quality never degrades for hard tokens — they just take the full path.

### Why This Matters for Inference

- **Decode is memory-bandwidth-bound.** Each decode step feeds 1 token through all layers. If half the tokens can exit at layer 2 of 6, that's ~3× fewer FLOPs per "easy" token.
- **Naturally complements speculative decoding.** Draft models guess "easy" tokens. Early exit heads serve the same population — tokens the model is confident about. You could even use early exit confidence to *decide when to speculate*.
- **Production relevance.** Google's CALM (Confident Adaptive Language Modeling), Meta's LayerSkip, and Microsoft's SkipDecode all implement variants of this idea.

---

## Base File

**Start from: [`nanogpt-kv-cache.py`](../../nanogpt-kv-cache.py)**

This is the right base because:
- It has the simplest model architecture with KV cache support (prefill/decode split)
- The `Head` class has stateful `key_cache`/`value_cache` — simple to reason about
- `Block` is straightforward: `x = x + self.sa(self.ln1(x)); x = x + self.ffwd(self.ln2(x))`
- The model uses `nn.Sequential` for blocks, which you'll need to replace with `nn.ModuleList` (to get intermediate hidden states)
- No batching, scheduling, or paging complexity — early exit is a **model architecture** change, not a serving optimization
- `generate_kv_cache()` is a clean single-request generate loop you can adapt
- Training loop is simple (no request queues, no radix trees)

You'll copy this file as `nanogpt-early-exit.py` and modify it.

**Why not `nanogpt-continuous-batching.py`?** Early exit makes batching much harder — different requests in the batch want to exit at different layers, which breaks the neat "all requests go through all layers" assumption. Start without batching, get correctness right, then figure out batching later.

**Why not `nanogpt.py`?** It lacks KV cache support. You need the cache for meaningful inference benchmarks.

---

## Hint 1: Understand the Architecture

Each intermediate exit head is a small classifier that maps the hidden state at layer `i` to vocabulary logits:

```python
class EarlyExitHead(nn.Module):
    """Lightweight classifier at an intermediate layer."""
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.proj = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        return self.proj(self.ln(x))
```

**Why the LayerNorm?** Intermediate hidden states aren't normalized the way the final output is (the final `ln_f` does that). Without per-exit LayerNorm, the logits from early layers are poorly calibrated, making confidence thresholds useless.

The modified `GPTLanguageModel` attaches one of these to **every** layer:

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # final exit (unchanged)
        
        # NEW: intermediate exit heads (one per layer, excluding the last)
        self.exit_heads = nn.ModuleList([
            EarlyExitHead(n_embd, vocab_size) for _ in range(n_layer - 1)
        ])
```

**Key design decision:** The last layer doesn't need an exit head — it already has `lm_head`. So you have `n_layer - 1` exit heads for a model with `n_layer` layers.

---

## Hint 2: Modify the Forward Pass

The forward pass needs two modes:

### Training mode: compute ALL exit logits for the distillation loss
```python
def forward(self, idx, targets=None, start_pos=0):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(
        torch.arange(start_pos, start_pos + T, device=device)
    )
    x = tok_emb + pos_emb

    exit_logits = []  # NEW: collect logits from each exit

    for i, block in enumerate(self.blocks):
        x = block(x)
        if i < n_layer - 1:
            exit_logits.append(self.exit_heads[i](x))  # intermediate exit

    x = self.ln_f(x)
    final_logits = self.lm_head(x)
    exit_logits.append(final_logits)  # final exit

    # Loss computation changes — see Hint 4
    ...
    return final_logits, loss, exit_logits
```

### Inference mode: exit early if confident
```python
# During inference, check confidence after each layer
for i, block in enumerate(self.blocks):
    x = block(x)  # KV cache version
    
    if i < n_layer - 1:
        logits_i = self.exit_heads[i](x)
        probs_i = F.softmax(logits_i[:, -1, :], dim=-1)
        confidence = probs_i.max(dim=-1).values.item()
        
        if confidence > threshold:
            return logits_i, None  # EXIT EARLY
```

**Question to ask yourself:** What happens to the KV cache when you exit early? Answer: The layers you *did* run still produce KV entries. The layers you skipped produce no KV entries. You need to handle this asymmetry on the next decode step. See Hint 5.

---

## Hint 3: The Training Loss — Self-Distillation

You can't just train the exit heads with the same cross-entropy loss as the final head. The intermediate representations haven't been refined by deeper layers yet — they're too raw for accurate prediction.

Instead, use **self-distillation**: the final layer's soft predictions (teacher) train the intermediate heads (students). Combined with the standard cross-entropy on the final output:

```python
def compute_early_exit_loss(exit_logits_list, targets, temperature=2.0, alpha=0.5):
    """
    Combined loss: cross-entropy on final head + distillation on intermediate heads.
    
    Args:
        exit_logits_list: list of (B*T, vocab_size) logits, one per layer
        targets: (B*T,) ground truth token ids
        temperature: softmax temperature for distillation (higher = softer targets)
        alpha: weight between CE loss and distillation loss
    """
    # Final head: standard cross-entropy
    final_logits = exit_logits_list[-1]
    ce_loss = F.cross_entropy(final_logits, targets)
    
    # Distillation: intermediate heads learn to match the final head's soft predictions
    teacher_probs = F.softmax(final_logits.detach() / temperature, dim=-1)
    
    distill_loss = 0.0
    for logits_i in exit_logits_list[:-1]:
        student_log_probs = F.log_softmax(logits_i / temperature, dim=-1)
        distill_loss += F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    distill_loss = distill_loss / max(len(exit_logits_list) - 1, 1)
    distill_loss = distill_loss * (temperature ** 2)  # scale by T² (standard trick)
    
    return alpha * ce_loss + (1 - alpha) * distill_loss
```

**Why self-distillation?** Training intermediate heads with raw cross-entropy makes them compete with the final head for gradient signal. Self-distillation is gentler — the intermediate heads learn the final head's *soft distribution*, not hard labels. This works much better in practice.

**The temperature parameter:** Higher temperature (2-4) produces softer probability distributions, which give the intermediate heads more gradient signal for near-correct tokens. Temperature 1.0 would be equivalent to just matching the argmax.

---

## Hint 4: KV Cache with Early Exit

This is the hardest part. When token A exits at layer 2 but token B goes through all 6 layers, the KV cache state is inconsistent:

```
Token A (exited at layer 2):
  Layer 0: KV ✓    Layer 1: KV ✓    Layer 2: KV ✓    Layer 3: KV ✗    Layer 4: KV ✗    Layer 5: KV ✗

Token B (full depth):
  Layer 0: KV ✓    Layer 1: KV ✓    Layer 2: KV ✓    Layer 3: KV ✓    Layer 4: KV ✓    Layer 5: KV ✓
```

When the *next* token needs to attend to token A at layer 4, there's no KV entry! Two approaches:

### Approach A: Copy-down (simpler, recommended for v1)
After early exit, copy the last computed hidden state down to fill the remaining layers' KV caches:

```python
# After exiting at layer exit_layer:
for remaining_layer in range(exit_layer + 1, n_layer):
    # Reuse the exit layer's K/V for all deeper layers
    head.key_cache = exit_k.clone()
    head.value_cache = exit_v.clone()
```

This is approximate but works surprisingly well — the hidden state at layer 2 is a reasonable approximation for layers 3-5 when the prediction is already confident.

### Approach B: Lazy computation (correct but complex)
Track which layers each token has KV entries for. When a future token needs to attend to a partially-computed token at a deeper layer, compute the missing layers on demand. This is what production systems (CALM) actually do, but it's significantly more complex.

**Recommendation:** Start with Approach A. It's simple, works well enough to see the speedup, and lets you focus on getting the confidence thresholds right.

---

## Hint 5: The Confidence Threshold

The threshold determines the quality/speed tradeoff. Too low → exits too early, quality degrades. Too high → never exits early, no speedup.

```python
def should_exit(logits, threshold=0.8):
    """Check if the model is confident enough to exit early."""
    probs = F.softmax(logits[:, -1, :], dim=-1)
    max_prob = probs.max(dim=-1).values.item()
    return max_prob > threshold
```

Good threshold values depend on the model and data:
- **0.9+**: Very conservative. Only exits on trivial predictions (spaces, punctuation). Small speedup, near-zero quality loss.
- **0.7-0.8**: Moderate. Exits on common words and obvious continuations. Noticeable speedup, slight quality change.
- **0.5-0.6**: Aggressive. Exits frequently. Big speedup but measurable quality degradation.

**Tip:** Log the exit layer distribution during benchmarks. If 80% of tokens exit at the same layer, your threshold is too loose (or too tight).

---

## Hint 6: The Early Exit Generate Loop

```python
@torch.no_grad()
def generate_early_exit(model, idx, max_new_tokens, threshold=0.8):
    """Generate with early exit — skip remaining layers when confident."""
    model.eval()
    clear_kv_cache(model)
    
    # Prefill: must run ALL layers (no early exit during prefill)
    logits, _ = model(idx)
    
    exit_layers = []  # track which layer each token exits at
    
    for step in range(max_new_tokens):
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Forward pass with early exit check
        logits, _, exit_layer = model.forward_with_early_exit(
            idx_next, start_pos=idx.shape[1] - 1, threshold=threshold
        )
        exit_layers.append(exit_layer)
    
    return idx, exit_layers
```

**Important:** Prefill should NOT exit early. During prefill, you're building the KV cache for the entire prompt. Exiting early would leave gaps. Only apply early exit during **decode** (single token steps).

---

## Hint 7: Measuring the Speedup

The right metrics are:

```python
def benchmark_early_exit(model, prompts, max_new_tokens=50, threshold=0.8):
    # 1. Average exit layer (lower = more compute saved)
    avg_exit = sum(exit_layers) / len(exit_layers)
    
    # 2. Exit layer distribution (histogram)
    for layer in range(n_layer):
        count = exit_layers.count(layer)
        print(f"  Layer {layer}: {count} tokens ({count/len(exit_layers)*100:.1f}%)")
    
    # 3. Theoretical speedup
    # If avg_exit = 2.0 and n_layer = 6, speedup ≈ 6/2 = 3×
    theoretical_speedup = n_layer / avg_exit
    
    # 4. Quality comparison: perplexity with vs without early exit
    # Run the same prompts with threshold=1.0 (no early exit) as baseline
```

---

## Test Scenarios

### Test 1: No Early Exit Baseline (threshold=1.0)
With threshold=1.0, no token ever exits early. The output should be **identical** to the normal `generate_kv_cache()`. This validates that the exit heads don't affect the normal forward path.

### Test 2: Forced Early Exit (threshold=0.0)
With threshold=0.0, every token exits at layer 0. The output will be low quality, but the model shouldn't crash. This validates the KV cache copy-down logic.

### Test 3: Quality vs Speedup Curve
Sweep thresholds from 0.5 to 0.95 and plot:
- x-axis: average exit layer (proxy for compute cost)
- y-axis: perplexity (quality)

You should see a smooth curve — not a cliff. If quality drops sharply at a particular threshold, something is wrong with the copy-down logic.

### Test 4: Exit Layer Distribution
Log which layer each token exits at. You should see a distribution, not a spike at one layer. If all tokens exit at the same layer, the exit heads aren't learning useful distinctions.

---

## Summary of New Components vs KV Cache

| Component | What's New |
|-----------|-----------|
| `EarlyExitHead` class | New — lightweight `LayerNorm + Linear` classifier per layer |
| `GPTLanguageModel.__init__` | Modified — adds `self.exit_heads` ModuleList |
| `GPTLanguageModel.forward` | Modified — `nn.Sequential` → `nn.ModuleList` loop, collects exit logits |
| `forward_with_early_exit()` | New — inference path with confidence-based layer skipping |
| `compute_early_exit_loss()` | New — self-distillation training loss |
| `generate_early_exit()` | New — generation loop with exit tracking |
| KV cache copy-down | New — fills missing KV entries for skipped layers |
| `Head`, `Block`, training data | **Unchanged** |

The key insight: **early exit is a model architecture change**, not a serving change. The `Block` and `Head` classes don't need modification. All the new logic lives in `GPTLanguageModel` and the training/generate loops above it.

---

## Recommended Implementation Order

1. **Step 1: Copy `nanogpt-kv-cache.py` → `nanogpt-early-exit.py`**
   - Change `nn.Sequential` to `nn.ModuleList` for `self.blocks`
   - Update the forward pass to loop through blocks explicitly
   - Verify output is identical to the original (no exit heads yet)

2. **Step 2: Add `EarlyExitHead` class and attach to model (Hint 1)**
   - Create exit heads for layers 0 through n_layer-2
   - Forward pass collects exit logits from each layer
   - Verify: output unchanged (exit heads exist but aren't used for prediction)

3. **Step 3: Implement self-distillation training loss (Hint 3)**
   - Replace the standard CE loss with the combined loss
   - Train the model and verify exit heads produce increasingly better predictions at deeper layers

4. **Step 4: Implement `forward_with_early_exit()` (Hints 2 & 5)**
   - Add confidence checking after each layer
   - Implement KV cache copy-down for skipped layers (Hint 4)
   - Test with threshold=1.0 (no exit) for correctness

5. **Step 5: Build `generate_early_exit()` (Hint 6)**
   - Only apply early exit during decode, not prefill
   - Track exit layer per token

6. **Step 6: Build the benchmark suite (Hint 7)**
   - Threshold sweep: quality vs speedup curve
   - Exit layer distribution histogram
   - Wall-clock timing comparison

7. **Step 7: Correctness verification (Test 1)**
   - threshold=1.0 → identical output to normal generate
   - threshold=0.0 → model doesn't crash

---

## Connection to Production Systems

| Your Implementation | Production System |
|---|---|
| `EarlyExitHead` (LN + Linear) | CALM uses learned "halting" classifiers |
| Confidence = max softmax prob | CALM uses learned confidence predictors |
| KV cache copy-down | CALM computes missing layers lazily |
| Threshold = fixed | Production uses calibrated per-layer thresholds |
| Single-request | LayerSkip uses early exit with batched speculation |

The architectural pattern is the same — you're just using a simpler confidence metric (max probability) instead of a learned confidence predictor.

---

## Gotchas

1. **Don't exit early during prefill.** The KV cache for the full prompt must be built at full depth. Early exit only makes sense during decode (1-token steps).

2. **The exit heads add parameters.** Each `EarlyExitHead` is `LayerNorm(n_embd) + Linear(n_embd, vocab_size)`. For `n_embd=384, vocab_size=65`, that's ~25K params × 5 layers = ~125K params. Modest compared to the 210K-param model, but not free.

3. **Self-distillation requires `.detach()` on the teacher logits.** If you forget this, the gradients from the distillation loss flow back through the final head and mess up training.

4. **The copy-down approximation degrades for deep models.** For a 4-6 layer NanoGPT it's fine. For a 32-layer model, copying layer 4's hidden state to layer 31 would be very wrong. Production systems handle this with lazy computation.

5. **Confidence calibration matters.** Raw softmax max probability is often poorly calibrated — the model can be "confident" but wrong. For a demo, it's fine. For production, you'd train a separate confidence predictor.

6. **The `block_size` position embedding limit still applies.** Total sequence length (prompt + generated) can't exceed `block_size`.
