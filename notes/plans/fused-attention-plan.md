# Fused Multi-Head Attention — Implementation Plan & Hints

## Base File: `nanogpt-kv-cache.py`

**Output file:** `nanogpt-fused-attention.py`

### Why this base file?

`nanogpt-kv-cache.py` is the right starting point because it's the simplest file that has the KV cache infrastructure you need to refactor:

- **Separate `Head` modules** — `nn.ModuleList([Head(head_size) for _ in range(num_heads)])` with three independent `nn.Linear` projections per head. This is the thing we're fusing.
- **Per-head KV caches** — `self.key_cache` / `self.value_cache` stored inside each `Head`. The fused version will store these as `(n_head, T, head_size)` tensors per layer instead.
- **Prefill/decode split** — Already separates full-prompt processing from single-token decoding. The fused attention must preserve this split.
- **No batching complexity** — No scheduler, no request queue, no paged attention. The cleanest possible environment to get the attention math right before layering complexity on top.

We don't need continuous batching, chunked prefill, or scheduling — those add noise. The goal is to understand how and why production transformers fuse QKV projections and multi-head computation into single operations.

---

## The Problem You're Solving

Right now, your `MultiHeadAttention` looks like this:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

Each `Head` has its own `nn.Linear` for Q, K, and V. With 4 heads, that's **12 separate linear layers** just for attention. Each one launches its own CUDA kernel (or CPU op), each one reads `x` from memory independently.

This has three problems:

1. **Kernel launch overhead** — 12 small matmuls instead of 3 large ones. On GPU, each kernel launch costs ~5-10μs of CPU-side overhead. At 4 layers × 12 ops = 48 tiny kernel launches per forward pass just for Q/K/V projections.

2. **Memory bandwidth waste** — Each `Head` reads the full input `x` from memory to compute its own Q, K, V. That's 12 reads of the same `(B, T, n_embd)` tensor instead of 3 (or even 1 if you do a fused QKV).

3. **No path to `scaled_dot_product_attention`** — PyTorch's `F.scaled_dot_product_attention` expects tensors shaped `(B, n_head, T, head_size)`. Your separate-head architecture produces `n_head` individual `(B, T, head_size)` tensors. You'd have to stack them anyway.

### How every production transformer does it

GPT-2, LLaMA, Mistral, and every model in HuggingFace Transformers use a single `CausalSelfAttention` module:

```
One nn.Linear(n_embd, 3 * n_embd)  →  split into Q, K, V  →  reshape to (B, n_head, T, head_size)  →  attention  →  reshape back
```

One matmul replaces twelve. The KV cache becomes a single `(B, n_head, T, head_size)` tensor per layer. This is the standard architecture, and understanding why it's better is fundamental.

---

## Hint 1: Replace `Head` + `MultiHeadAttention` with `CausalSelfAttention`

Your new class should have a single linear projection for Q, K, and V combined. Think about:

- What shape should the weight matrix be if you want one matmul to produce Q, K, and V simultaneously?
- After that matmul, how do you split the output into three equal parts?
- How do you reshape from `(B, T, n_embd)` to `(B, n_head, T, head_size)` so that each head's data is contiguous?

The key operations you'll need:
- `tensor.split(split_size, dim=...)` — splits a tensor along a dimension
- `tensor.view(B, T, n_head, head_size)` — reshapes without copying
- `tensor.transpose(1, 2)` — swaps the `T` and `n_head` dimensions

The output projection (`self.proj`) stays the same — it's already `nn.Linear(n_embd, n_embd)`.

**Question to think about:** Why do we transpose to put `n_head` before `T`? What does that do to memory layout, and why does batched matmul (`@`) need it that way?

---

## Hint 2: The Attention Computation Changes Shape

With separate heads, each head computes:
```
q @ k.T → (B, T, T)     # one head's attention weights
```

With fused heads, you're computing all heads simultaneously:
```
q @ k.T → (B, n_head, T, T)     # all heads' attention weights at once
```

Think about:
- The causal mask (`tril`) now needs to broadcast correctly over the `(B, n_head, T, T)` shape. Does `self.tril[:T, :T]` still work? (Hint: broadcasting rules.)
- The softmax is still over `dim=-1`.
- The final output after `attn_weights @ v` will be `(B, n_head, T, head_size)`. How do you get it back to `(B, T, n_embd)` for the output projection?

You'll need `tensor.transpose(1, 2).contiguous().view(B, T, n_embd)` — think about why `.contiguous()` is needed here.

---

## Hint 3: Refactor the KV Cache

Currently, each `Head` object stores its own `key_cache` and `value_cache` as `(B, T_cached, head_size)` tensors. With fused attention, you need to decide on a new cache structure.

Two options:
1. **Per-layer cache as `(B, n_head, T_cached, head_size)`** — a single tensor per layer holding all heads' cached K and V.
2. **Same dict structure** but keyed by `layer_idx` only, with values shaped `(B, n_head, T, head_size)`.

Think about:
- During decode (T=1), you `torch.cat` the new K/V onto the cached K/V along the `T` dimension. Which dimension is `T` in the fused layout?
- The cache should live in the `CausalSelfAttention` module (not inside separate heads, since those no longer exist).
- `clear_kv_cache()` needs to be updated — it currently walks all `Head` instances via `isinstance(module, Head)`.

---

## Hint 4: Training vs. Inference Paths

Your current `Head.forward()` has two branches:

```python
if not self.training:
    # KV cache path (prefill or decode)
else:
    # Standard causal attention with tril mask
```

Your fused `CausalSelfAttention` needs the same split, but now you can also consider using PyTorch's built-in:

```python
torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```

This function:
- Handles the causal mask internally (no need for `self.tril`)
- Automatically dispatches to FlashAttention or memory-efficient attention on CUDA when available
- Expects inputs shaped `(B, n_head, T, head_size)`

**You don't have to use it** — implementing attention manually is more educational. But try adding it as an optional path and benchmarking the difference. On GPU, `F.scaled_dot_product_attention` with FlashAttention should be noticeably faster for prefill (it avoids materializing the full `(T, T)` attention matrix).

**Important caveat for the decode path:** When you have a KV cache (past K/V concatenated with new K/V), you can't simply use `is_causal=True` because the query length (1) doesn't match the key length (T_past + 1). You'll need to either:
- Use `attn_mask` parameter instead of `is_causal` during decode
- Or keep manual attention for decode and only use `F.scaled_dot_product_attention` for prefill/training

---

## Hint 5: Weight Compatibility — Can You Load the Old Weights?

This is subtle. Your old model has:
```
blocks.0.sa.heads.0.key.weight    → (head_size, n_embd)
blocks.0.sa.heads.0.query.weight  → (head_size, n_embd)
blocks.0.sa.heads.0.value.weight  → (head_size, n_embd)
blocks.0.sa.heads.1.key.weight    → (head_size, n_embd)
...
```

Your new model has:
```
blocks.0.sa.c_attn.weight         → (3 * n_embd, n_embd)
blocks.0.sa.c_proj.weight         → (n_embd, n_embd)
```

These are mathematically equivalent but have different `state_dict` keys. You have two choices:

1. **Just retrain** — The model trains in ~5 seconds at 120 iterations. Simplest approach.
2. **Write a weight conversion function** — Stack the old per-head Q/K/V weights into the fused weight matrix. This is a great exercise in understanding how the fused layout maps to per-head weights.

If you go with option 2, think about the ordering: does the fused weight interleave heads (`Q0, K0, V0, Q1, K1, V1, ...`) or group by projection type (`Q0, Q1, ..., K0, K1, ..., V0, V1, ...`)? GPT-2 uses the latter. This affects how you `split()` the output.

---

## Hint 6: Verify Equivalence

The most important test: **fused attention must produce identical logits to separate-head attention** (given the same weights).

Design a test:
1. Train the old `nanogpt-kv-cache.py` model, save the `state_dict`
2. Write a weight conversion that loads old weights into the new fused model
3. Run the same input through both models with the same random seed
4. Assert `torch.allclose(old_logits, new_logits, atol=1e-6)`

If they don't match:
- Check your reshape/transpose order — `(B, T, n_head, head_size)` vs. `(B, n_head, T, head_size)` is a common source of bugs
- Check your split order — are you splitting QKV in the right order?
- Check `.contiguous()` — transposes create non-contiguous views that can silently produce wrong results in some ops

---

## Hint 7: Benchmark — Why Fusion Matters

Design benchmarks that show the three benefits:

### Benchmark 1: Forward Pass Latency (Prefill)
Time a single forward pass with a full prompt (e.g., 32 tokens). Compare old vs. fused. On GPU, the fused version should be faster due to fewer kernel launches.

### Benchmark 2: Decode Step Latency
Time a single decode step (1 token, with cached K/V). This is where kernel launch overhead dominates — each of the 12 small matmuls in the old version has non-trivial launch cost relative to the computation.

### Benchmark 3: Tokens/Second End-to-End
Run the full `generate_kv_cache()` loop for 50 tokens. Measure total throughput.

### Benchmark 4: `F.scaled_dot_product_attention` vs. Manual
If you implement both paths (Hint 4), compare them. On GPU, SDPA should win on prefill. On CPU, the difference may be small.

**Expected results at this scale (210K params):**
- On CPU: fusion helps modestly (maybe 1.1-1.3×) because kernel launch overhead is less of a factor
- On GPU: fusion helps more (maybe 1.3-2×) because the kernels are tiny and launch overhead dominates
- With SDPA on GPU: prefill could be significantly faster if FlashAttention kicks in

---

## Summary of Changes vs. `nanogpt-kv-cache.py`

| Component | What Changes |
|-----------|-------------|
| `Head` class | **Deleted entirely** |
| `MultiHeadAttention` class | **Replaced** by `CausalSelfAttention` |
| `CausalSelfAttention` | **New** — fused QKV projection, multi-head reshape, attention, output projection, KV cache |
| `Block.forward()` | Calls `self.sa(self.ln1(x))` — same interface, different internals |
| `GPTLanguageModel.forward()` | Should remain unchanged (blocks still return `(x, loss)` or `(logits, loss)`) |
| `clear_kv_cache()` | Update to walk `CausalSelfAttention` modules instead of `Head` |
| Training loop | **No changes** — the model trains identically |
| `generate_kv_cache()` | **No changes** — the cache is internal to `CausalSelfAttention` |

The key insight: **the rest of the model doesn't know anything changed**. The `Block` still calls `self.sa(x)` and gets back attention output. The fused attention is a drop-in replacement with the same external interface, different (better) internal implementation.

---

## Recommended Implementation Order

1. **Step 1: Copy `nanogpt-kv-cache.py` → `nanogpt-fused-attention.py`**
   - Update the module docstring.

2. **Step 2: Write the `CausalSelfAttention` class (Hints 1-2)**
   - Single `c_attn = nn.Linear(n_embd, 3 * n_embd)` for fused QKV.
   - Single `c_proj = nn.Linear(n_embd, n_embd)` for output projection.
   - Manual attention: compute `q @ k.T`, apply causal mask, softmax, apply to `v`.
   - Get the training path working first (no cache).

3. **Step 3: Add the KV cache (Hint 3)**
   - Store `key_cache` and `value_cache` as `(B, n_head, T, head_size)`.
   - Handle the `torch.cat` along the T dimension during decode.
   - Implement prefill vs. decode branching.

4. **Step 4: Update `clear_kv_cache()` and `Block` (Hint 3)**
   - Block now uses `CausalSelfAttention` instead of `MultiHeadAttention`.
   - Delete the old `Head` and `MultiHeadAttention` classes.

5. **Step 5: Verify correctness (Hint 6)**
   - Train from scratch, generate text, confirm it looks like Shakespeare.
   - Optionally: weight conversion + logit comparison with old model.

6. **Step 6: Add `F.scaled_dot_product_attention` as optional path (Hint 4)**
   - Use it for the training/prefill path where `is_causal=True` works.
   - Keep manual attention for the decode path (or figure out the `attn_mask` approach).

7. **Step 7: Benchmark (Hint 7)**
   - Compare old vs. fused on prefill latency, decode latency, and end-to-end throughput.

---

## The Conceptual Map

```
Separate-head architecture (current):

  Input x: (B, T, n_embd=32)
       │
       ├──→ Head 0: W_q0(8,32) → q0(B,T,8)    W_k0 → k0    W_v0 → v0
       ├──→ Head 1: W_q1(8,32) → q1(B,T,8)    W_k1 → k1    W_v1 → v1
       ├──→ Head 2: W_q2(8,32) → q2(B,T,8)    W_k2 → k2    W_v2 → v2
       └──→ Head 3: W_q3(8,32) → q3(B,T,8)    W_k3 → k3    W_v3 → v3
                     ───────────────────
                     12 separate matmuls
                     12 kernel launches
                     12 reads of x from memory

  Each head computes attention independently: q_i @ k_i.T → softmax → @ v_i
  Then: cat(out0, out1, out2, out3) → proj → output

─────────────────────────────────────────────────────────

Fused architecture (target):

  Input x: (B, T, n_embd=32)
       │
       └──→ c_attn: W_qkv(96, 32) → qkv(B, T, 96)    ← ONE matmul
                     │
                     split into q(B,T,32), k(B,T,32), v(B,T,32)
                     │
                     reshape each to (B, n_head=4, T, head_size=8)
                     │
                     q @ k.T → (B, 4, T, T)    ← ONE batched matmul for ALL heads
                     │
                     causal mask → softmax → @ v → (B, 4, T, 8)
                     │
                     reshape to (B, T, 32) → c_proj → output

  Total: 3 matmuls (QKV projection, attention, output projection)
  3 kernel launches instead of 14
  1 read of x instead of 12
```
