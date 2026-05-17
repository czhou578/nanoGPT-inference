# INT8 Quantization — Implementation Plan

## What You're Actually Doing

First, a terminology fix: `torch.quantization.quantize_dynamic` is **dynamic quantization**,
not static. They're related but distinct:

| | Dynamic Quantization | Static (Post-Training) Quantization |
|---|---|---|
| **Weights** | Pre-quantized to INT8 at load time | Pre-quantized to INT8 at load time |
| **Activations** | Quantized **at runtime**, per-batch | Quantized with **fixed scales** from calibration |
| **Calibration required?** | ❌ No | ✅ Yes — run representative data first |
| **PyTorch API** | `quantize_dynamic(model, ...)` | `prepare()` → calibrate → `convert()` |
| **When to use** | LSTM, Linear-heavy models; memory bound | CNNs, models where activation range is stable |
| **Speedup source** | Less memory to load weights (bandwidth) | Fewer memory ops + INT8 GEMM kernels |

This notebook will implement **both**, so you understand the tradeoff.

---

## Why You Won't See a Speedup on nanoGPT (210K params)

The real bottleneck at this scale is **not** memory bandwidth — it's Python overhead and
the attention computation itself. Quantization wins when:

1. The model is large enough that **weight loading from DRAM is the bottleneck** (>1B params)
2. The hardware has fast **INT8 GEMM** kernels (e.g. NVIDIA Tensor Cores, modern ARM CPUs)

On a 210K model on a T4 GPU, you may even see a *slowdown* — the INT8 dequantization
overhead can exceed the savings. That's fine and expected. **The goal here is to learn the
API and the concepts**, not to hit a perf number.

---

## Notebook Structure

The notebook already has the trained `GPTLanguageModel`. You'll add 4 new sections after
the training cell:

### Section 1 — Baseline Benchmark (FP32)

Measure FP32 inference latency and model size before touching anything.

```python
import os, time, copy

def benchmark_generate(model, context, n_tokens=200, n_trials=5):
    """Returns mean latency in ms over n_trials runs."""
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_trials):
            t0 = time.perf_counter()
            model.generate(context, n_tokens)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return sum(times) / len(times)

def model_size_mb(model):
    """Size of model parameters in MB."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 1e6

context = torch.zeros((1, 1), dtype=torch.long, device=device)
fp32_ms  = benchmark_generate(model, context)
fp32_mb  = model_size_mb(model)
print(f"FP32 | size: {fp32_mb:.2f} MB | latency: {fp32_ms:.1f} ms")
```

> **Important:** `quantize_dynamic` only works on **CPU**. If you're on CUDA, you'll need
> `model_cpu = copy.deepcopy(model).cpu()` before quantizing, and run inference on CPU too.

---

### Section 2 — Dynamic Quantization

One-liner API. Quantizes `nn.Linear` weights to INT8 at model-load time.
Activations are quantized dynamically per forward pass.

```python
import torch.quantization

model_dq = copy.deepcopy(model).cpu()
model_dq.eval()

model_dq = torch.quantization.quantize_dynamic(
    model_dq,
    {nn.Linear},   # which layer types to quantize
    dtype=torch.qint8
)
```

**What to inspect after this:**
```python
# The Linear layers are now QuantizedLinear — look at one:
print(model_dq.blocks[0].sa.proj)
# Output: DynamicQuantizedLinear(in=64, out=64, dtype=torch.qint8, qscheme=per_tensor_affine)

# The weight is now a packed INT8 tensor, not a float32 tensor:
print(model_dq.blocks[0].sa.proj.weight().dtype)  # torch.qint8
```

**Benchmark it:**
```python
context_cpu = torch.zeros((1, 1), dtype=torch.long)  # CPU tensor
dq_ms  = benchmark_generate(model_dq, context_cpu)
dq_mb  = model_size_mb(model_dq)   # Note: may not reflect INT8 savings accurately
print(f"DQ INT8 | size: {dq_mb:.2f} MB | latency: {dq_ms:.1f} ms")
```

**Verify the output is still coherent:**
```python
torch.manual_seed(42)
out = model_dq.generate(context_cpu, max_new_tokens=100)
print(decode(out[0].tolist()))  # Should still be Shakespeare-ish
```

The output **will differ** from FP32 (INT8 introduces rounding error) but should still be
grammatically sensible. This is the accuracy/efficiency tradeoff.

---

### Section 3 — Static (Post-Training) Quantization

This requires 3 steps: **fuse → prepare → calibrate → convert**.

#### Step 3a: Fuse layers

PyTorch can fuse adjacent `Linear → ReLU` into a single quantized op. Your `FeedForward`
has exactly this pattern:

```python
model_sq = copy.deepcopy(model).cpu()
model_sq.eval()

# Fuse Linear + ReLU in each FeedForward block
for block in model_sq.blocks:
    torch.quantization.fuse_modules(
        block.ffwd.net,
        [['0', '1']],   # indices of [Linear, ReLU] in the Sequential
        inplace=True
    )
```

#### Step 3b: Set quantization config and prepare

```python
model_sq.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # CPU
# 'fbgemm' = for x86 CPUs; use 'qnnpack' for ARM (e.g. Apple Silicon, Android)

torch.quantization.prepare(model_sq, inplace=True)
```

After `prepare()`, the model has **observer hooks** inserted before/after each quantized op.
These accumulate statistics (min, max) on your calibration data.

#### Step 3c: Calibrate

Run a few batches of representative data through the model. The observers record activation
statistics — this is how static quantization knows the right INT8 scale/zero-point for each
layer's activations.

```python
# Calibration — run ~100 batches of real data (no gradient needed)
model_sq.eval()
with torch.no_grad():
    for _ in range(100):
        xb, _ = get_batch('val')
        xb_cpu = xb.cpu()
        model_sq(xb_cpu)   # observers collect min/max stats
```

#### Step 3d: Convert

Replace float ops with quantized INT8 ops, bake in the scales from calibration:

```python
torch.quantization.convert(model_sq, inplace=True)

print(model_sq.blocks[0].sa.proj)
# QuantizedLinear — now with fixed scale/zero_point from calibration
```

**Benchmark:**
```python
sq_ms = benchmark_generate(model_sq, context_cpu)
sq_mb = model_size_mb(model_sq)
print(f"SQ INT8 | size: {sq_mb:.2f} MB | latency: {sq_ms:.1f} ms")
```

---

### Section 4 — Comparison Table

```python
print(f"{'Method':<20} {'Size (MB)':>10} {'Latency (ms)':>14} {'Speedup':>10}")
print("-" * 56)
print(f"{'FP32 (baseline)':<20} {fp32_mb:>10.2f} {fp32_ms:>14.1f} {'1.00x':>10}")
print(f"{'Dynamic INT8':<20} {dq_mb:>10.2f} {dq_ms:>14.1f} {fp32_ms/dq_ms:>9.2f}x")
print(f"{'Static INT8':<20} {sq_mb:>10.2f} {sq_ms:>14.1f} {fp32_ms/sq_ms:>9.2f}x")
```

Expected output (rough — your numbers will vary):
```
Method               Size (MB)   Latency (ms)    Speedup
────────────────────────────────────────────────────────
FP32 (baseline)           0.84          312ms      1.00x
Dynamic INT8              0.22           --ms      ~0.8x   ← slower! overhead > savings
Static INT8               0.22           --ms      ~0.9x   ← still marginal at 210K
```

The size reduction is real (4x: FP32→INT8). The latency won't improve at this scale.

---

## Key Concepts to Internalize

### Why dynamic quantization is still useful even without activation calibration

Dynamic quantization computes the activation scale **on-the-fly per input tensor**. This
makes it robust to out-of-distribution inputs — you don't need calibration data, and the
scale adapts to whatever activations your input produces. The tradeoff: that per-tensor
scale computation adds runtime overhead.

### What `qscheme=per_tensor_affine` means

Each tensor gets **one** scale and zero-point.
`per_channel` quantization gives each *output channel* its own scale — more accurate but
more storage overhead. For weights, per-channel is almost always better.

### The `fbgemm` vs `qnnpack` split

- `fbgemm` — optimized for x86 servers (your Colab T4 host CPU uses this)
- `qnnpack` — optimized for ARM (Apple Silicon, phone chips)

PyTorch dispatches to different INT8 GEMM kernels depending on which you select.

---

## What's NOT Quantized (and Why)

- `nn.Embedding` — lookup tables are already memory-efficient; quantizing embeddings
  requires special handling (not in the standard pipeline)
- `nn.LayerNorm` — too sensitive to quantization error; typically left in FP32
- `nn.Dropout` — no parameters; irrelevant
- The KV cache tensors — those are activations, not parameters; handled separately

In production (e.g. llm.int8 by Tim Dettmers), embeddings and LayerNorm stay FP32 while
only the big Linear projection weights go INT8. That's essentially what `quantize_dynamic`
does here.

---

## Gotchas

1. **CUDA vs CPU**: `quantize_dynamic` and `prepare/convert` only work on CPU. Always
   `model.cpu()` first.

2. **`model_size_mb` undercounts after quantization**: PyTorch's quantized tensors store
   INT8 but `p.element_size()` for a `QInt8` parameter still returns different values
   depending on how it's packed. Use `torch.save` + `os.path.getsize` for accurate disk size.

3. **`generate()` must also run on CPU**: Your `generate` method calls
   `torch.multinomial` and does tensor ops — make sure your context tensor and all
   intermediate tensors are on CPU too, or you'll get device mismatch errors.

4. **Static quantization fails if your model has control flow that changes with input**:
   The observer assumes the computational graph is stable across your calibration runs.
   Your `Head.forward` branches on `past_k is not None` — make sure your calibration
   always uses the same branch (e.g. always pass or always omit `past_kvs`).
