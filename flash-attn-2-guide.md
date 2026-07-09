# Flash Attention 2 Integration Guide

**Target hardware:** Blackwell (GB10 / CUDA 13.0 / Arch 12.1)  
**Foundation file:** `nanogpt-cuda-graph.py` — the most complete implementation in the repo. It has fused QKV projections, pre-allocated KV cache buffers, static buffer protocol, `cache_pos` parameter, `decode_cached()` graph-safe path, and `decode_one_token()`.

---

## Why FA-2

Standard attention (including your fused QKV path) reads K and V from HBM, computes QK^T, does softmax, then multiplies by V — reading V **twice** from memory. FA-2 eliminates that by tiling the computation: it reads K and V once into SRAM, computes partial attention, and writes out a single result. On Blackwell this gives:

- **2–3× speedup** on prefill (compute-heavy, long sequences)
- **~1.2× speedup** on decode (still memory-bound, but fewer HBM reads)
- **Better numerical stability** with online softmax and max-tracking
- **No precision loss** — it's an algorithm change, not quantization

---

## Step 0 — Install `flash-attn`

```bash
pip install flash-attn
```

This installs the kernel from [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). On CUDA 13.0 / Blackwell, make sure you have a recent version (`flash-attn >= 2.7`):

```bash
pip install --upgrade flash-attn
```

If direct install fails, build from source:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install build
python -m build --wheel
pip install dist/*.whl
```

---

## Step 1 — Model structure changes (minimal)

### What changes

In `CausalSelfAttention.__init__`, you already have:

```python
self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
```

**Nothing in init changes.** The fused QKV weight shape stays the same.

### What changes in `forward()`

**Before (lines 129–153 of cuda-graph.py):**
```python
qkv = self.qkv(x)
q, k, v = qkv.split(n_embd, dim=2)
q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
# ... then q @ k.transpose(-2,-1) with mask, softmax, @ v
```

**After — pack into a single (B, T, 3, n_head, head_size) tensor:**
```python
qkv = self.qkv(x)  # (B, T, 3 * n_embd)
qkv = qkv.view(B, T, 3, self.num_heads, self.head_size)  # (B, T, 3, n_head, hs)
qkv = qkv.transpose(1, 2)  # (B, 3, T, n_head, hs) — FLASH-ATTN expects this shape
```

That's it for the reshape. The FA-2 kernel takes exactly this `(B, T, 3, n_head, head_size)` tensor.

### What changes in `decode_cached()`

**Before:** Separate QKV split, manual cache write, manual attention with mask.
**After:** Pack QKV the same way, pass to FA-2. The mask logic changes — FA-2 uses a `cu_seqlens` (sequence length) parameter instead of a boolean mask, which is actually simpler for decode because T=1.

---

## Step 2 — The two attention paths

You need **two** attention implementations: one for training/prefill (variable T, with masks), one for decode (T=1, graph-safe).

### Path A: Prefill — `forward()` with FA-2

FA-2's varlen kernel supports variable-length sequences AND attention masks:

```python
from flash_attn import flash_attn_varlen_qkvpacked_func

def forward(self, x, targets=None, start_pos=0, cache_pos=None):
    B, T, C = x.shape
    qkv = self.qkv(x)
    qkv = qkv.view(B, T, 3, self.num_heads, self.head_size).transpose(1, 2)
    # qkv: (B, 3, T, n_head, head_size)

    # ── Inference with cache ──
    if not self.training and cache_pos is not None:
        # Write K/V into cache (same as before)
        self.key_cache[:, :, cache_pos:cache_pos + T, :] = k
        self.value_cache[:, :, cache_pos:cache_pos + T, :] = v

        # Gather cached K/V and new K/V into a single qkv tensor
        # ... see Step 3 for details ...

        # Call FA-2 with the full cached + new qkv
        # You need cu_seqlens = [0, T] for a single sequence
        out = flash_attn_varlen_qkvpacked_func(
            qkv_packed,              # (1, 3, total_T, n_head, head_size)
            cu_seqlens=torch.tensor([0, total_T], dtype=torch.int32, device=device),
            max_seqlen=total_T,
        )
        # reshape out back to (B, T, C)
```

**Key insight:** During prefill + cache write, you still write to the KV cache buffers first, then gather everything back into a `(B, 3, T, n_head, hs)` tensor to call FA-2. This is the "recompute" style — you write K/V out, then read them back in for attention. For decode (T=1), the FA-2 library also has a KV-cache-aware kernel (`flash_attn_kvpacked_func` or `flash_attn_with_kvcache` in newer versions).

### Path B: Decode — `decode_cached()` with FA-2

For decode, use `flash_attn_with_kvcache` (FA-2 + KV cache in one call):

```python
from flash_attn.layers.rotary import RotaryEmbedding  # if using RoPE; for sinusoidal, skip

def decode_cached(self, x, cache_pos):
    B, T, C = x.shape  # T is always 1
    qkv = self.qkv(x)
    qkv = qkv.view(B, T, 3, self.num_heads, self.head_size).transpose(1, 2)
    # qkv: (B, 3, 1, n_head, head_size)

    # For T=1 decode, call the KV cache version:
    out = flash_attn_with_kvcache(
        qkv[:, 0:1, :, :, :],   # q: (B, 1, n_head, head_size)
        self.key_cache,          # (1, n_head, block_size, head_size)
        self.value_cache,        # (1, n_head, block_size, head_size)
        cache_pos=cache_pos,     # which slot to write into
        # softmax_scale = self.head_size ** -0.5,  # auto-computed if omitted
    )
    # out: (B, 1, n_embd) — already has the output projection applied internally

    out = out.transpose(1, 2).contiguous().view(B, 1, C)
    out = self.attn_proj(out)
    return out
```

Wait — `flash_attn_with_kvcache` takes q, k, v **separately** (not packed). So for decode:

```python
def decode_cached(self, x, cache_pos):
    B, T, C = x.shape
    qkv = self.qkv(x)
    q, k, v = qkv.split(n_embd, dim=2)
    q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, 1, n_head, hs)
    k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, 1, n_head, hs)
    v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, 1, n_head, hs)

    # Write K/V to cache
    self.key_cache.index_copy_(2, cache_pos.view(1), k)
    self.value_cache.index_copy_(2, cache_pos.view(1), v)

    # Call FA-2 with cached K/V
    out = flash_attn_with_kvcache(
        q,                               # (B, 1, n_head, hs)
        self.key_cache,                  # cached K, already updated
        self.value_cache,                # cached V, already updated
        cache_pos=cache_pos,             # scalar or (B,) tensor
        # softmax_scale is auto = 1/sqrt(head_size)
    )
    # out: (B, 1, n_embd)
    out = out.transpose(1, 2).contiguous().view(B, 1, C)
    out = self.attn_proj(out)
    return out
```

---

## Step 3 — Training path (unchanged attention)

**Training doesn't use FA-2 in this guide.** Keep the standard PyTorch attention for training:

```python
else:
    # Training path — standard PyTorch attention (unchanged)
    scale = self.head_size ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    out = attn @ v
```

Reason: FA-2 is inference-only for this project. Training benefits are real but unnecessary complexity — the eval harness already validates output quality, and your training data is tiny.

---

## Step 4 — Positional embeddings

**FA-2 does NOT handle positional embeddings.** Sinusoidal (RoPE is different — that's a rotary embedding on the Q/K values themselves). Your current positional embedding is sinusoidal, added to token embeddings **before** the attention layer. This is correct and unchanged — FA-2 never touches positional embeddings.

```python
# In Block.forward() — NO CHANGE NEEDED
def forward(self, x):
    x = x + self.sa(self.ln1(x))   # sa now uses FA-2 internally
    x = x + self.ffwd(self.ln2(x))
    return x
```

---

## Step 5 — PagedAttention / KV cache integrations

If you're also using paged attention (`nanogpt-paged-attention.py`) or radix tree (`nanogpt-radix-tree.py`), the FA-2 integration point is the **gather/kv_cache assembly**. FA-2's `flash_attn_with_kvcache` expects a contiguous `(B, n_head, seq_len, head_size)` tensor. For paged attention, you'd gather from the block pool into a contiguous tensor first, then call FA-2, then scatter back. The rest of the paged/radix logic stays the same.

---

## Step 6 — CUDA graph compatibility

**FA-2 replay is graph-compatible.** When you capture the decode step, the FA-2 kernel is a single GPU kernel that gets recorded into the graph. The `decode_cached()` method still reads from the same static buffers and writes to the same fixed-address tensors.

The key change: instead of recording ~50 small kernels (matmul, softmax, etc.), the graph captures one big FA-2 kernel. This is actually **faster** than the current CUDA graph capture because:

1. Less CPU-side graph reconstruction overhead
2. One kernel launch instead of many
3. The graph replay is just `graph.replay()` — unchanged

---

## Step 7 — Naming the new file

Copy `nanogpt-cuda-graph.py` and name the new file:

```
nanogpt-flash-attn2.py
```

Or if you want to combine FA-2 + CUDA graphs:

```
nanogpt-flash-attn2-cuda-graph.py
```

The second name signals that it has **both** FA-2 attention **and** CUDA graph capture/replay. This is the most complete version.

---

## File structure comparison

```
BEFORE (nanogpt-cuda-graph.py)              AFTER (nanogpt-flash-attn2-cuda-graph.py)
─────────────────────────                    ─────────────────────────────────────────
CausalSelfAttention                          CausalSelfAttention
  self.qkv = nn.Linear(…, 3*…)               self.qkv = nn.Linear(…, 3*…)  ← UNCHANGED
  self.attn_proj = nn.Linear(…, …)           self.attn_proj = nn.Linear(…, …)  ← UNCHANGED
  self.key_cache = zeros(...)                self.key_cache = zeros(...)  ← UNCHANGED
  self.value_cache = zeros(...)              self.value_cache = zeros(...)  ← UNCHANGED
  self.kv_indices = arange(...)              self.kv_indices = arange(...)  ← UNCHANGED

  forward(x, cache_pos):                     forward(x, cache_pos):
    qkv = self.qkv(x)                        qkv = self.qkv(x)
    q,k,v = split + reshape                  qkv.view + transpose → (B,3,T,n_head,hs)
    write K,V to cache                       write K,V to cache  ← same
    q @ K.T with mask                        qkv → flash_attn_varlen_qkvpacked_func
    softmax + @V                             reshape + attn_proj
    attn_proj                                ← DONE

  decode_cached(x, cache_pos):               decode_cached(x, cache_pos):
    qkv = self.qkv(x)                        qkv = self.qkv(x)
    split + reshape                          split: q,k,v (B,1,n_head,hs)
    index_copy_ K,V                          index_copy_ K,V  ← same
    q @ K.T with mask                        q → flash_attn_with_kvcache(k_cache, v_cache, cache_pos)
    softmax + @V                             reshape + attn_proj
    attn_proj                                ← DONE

GPTLanguageModel.decode_one_token()          GPTLanguageModel.decode_one_token()
  block.decode_cached(x, pos)                block.decode_cached(x, pos)  ← UNCHANGED
  (FA-2 runs inside decode_cached)

generate_cuda_graph()                        generate_cuda_graph()
  same 3-phase structure                     same 3-phase structure  ← UNCHANGED
  (graph captures FA-2 kernel instead of     (one kernel instead of ~50)
   ~50 small kernels)
```

---

## Quick verification checklist

After implementing, run through these in order:

1. **Training loss matches:** Train with the FA-2 model and verify training loss is identical (within float32 tolerance) to the non-FA-2 version. Training uses standard attention, so this is a sanity check that FA-2 doesn't leak into training.

2. **Decode logits match:** Run one decode step with standard attention, save the logits. Run one decode step with FA-2 `decode_cached()`, save the logits. They should match within ~1e-4 (FA-2 uses slightly different numerical paths — online softmax vs standard — so tiny differences are expected).

3. **CUDA graph captures:** `torch.cuda.CUDAGraph(graph, stream=s)` succeeds without errors during the capture phase. If it fails, the FA-2 kernel may be doing dynamic allocations — use `torch.cuda.empty_cache()` before capture and ensure `q, k, v` tensors are reshaped views (not copies) of the static buffers.

4. **End-to-end generation:** `generate_cuda_graph()` produces identical (or near-identical) text output with `torch.manual_seed(42)`.

5. **Benchmark:** Add an `flash_attn2_benchmark_runs.py` file. Compare:
   - prefill latency (50-token prompt) — expect 2–3× speedup
   - decode throughput (tokens/sec over 64 tokens) — expect 1.2–1.5× speedup on decode
   - overall TTFT — expect 2–3× faster because prefill dominates TTFT

---

## Key references

| Resource | Why it helps |
|----------|-------------|
| [flash-attn GitHub](https://github.com/Dao-AILab/flash-attention) | API docs, example code, FAQ |
| `flash_attn.ops.flash_attn_func.flash_attn_varlen_qkvpacked_func` | Varlen packed QKV — your prefill path |
| `flash_attn.ops.flash_attn_func.flash_attn_with_kvcache` | KV cache version — your decode path |
| FA-2 paper (triton demo) | The original implementation shows the tiling algorithm in ~50 lines of Triton — great for understanding how it works under the hood |
| Your `notes/concepts/` — "Hardware Fundamentals" article | Memory hierarchy context for *why* FA-2 is faster |

---

## Optional: Implement FA-2 from scratch

If you want to understand the algorithm deeply (rather than just use the library), implement the core tiling loop yourself. The paper's algorithm is ~50 lines:

```python
# Pseudocode for a simplified FA-2 kernel (no tiling, no parallel reduction)
def flash_attn_simple(q, k, v, softmax_scale):
    # q,k,v: (B, n_head, T, head_size)
    s = q @ k.transpose(-2, -1) * softmax_scale  # (B, n_head, T, T)
    # Online softmax — tracks running max and sum
    m = torch.full((B, n_head, T), -math.inf, device=q.device)
    l = torch.zeros_like(m)
    out = torch.zeros_like(q)
    for j in range(T):  # tile across KV dimension
        row_s = s[:, :, :, j:j+1]  # (B, n_head, T, 1)
        new_m = torch.maximum(m, row_s)
        new_l = torch.exp(m - new_m) * l + torch.exp(row_s - new_m)
        out = out * (torch.exp(m - new_m) / new_l) + (v[:, :, j:j+1] * torch.exp(row_s - new_m) / new_l)
        m, l, out = new_m, new_l, out
    return out
```

The real FA-2 adds:
- **SRAM tiling** — process chunks of K/V at a time to fit in L2
- **Recomputation** — recompute the softmax numerator on the fly instead of storing the full S matrix (saves HBM writes)
- **Parallel reduction** — use thread blocks to reduce across tiles
- **FlashAttention-3** — even more optimization using tensor cores and async copy on Hopper+

For the DGX Spark, the library kernel already exploits Blackwell's tensor cores optimally. A hand-written kernel would be an interesting exercise but unlikely to beat the library version.