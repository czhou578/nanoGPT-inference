# FP8 Kernels in LLM Inference: From First Principles

---

## 1. What Is FP8? Starting From Number Representation

### How computers represent decimal numbers

Every number in a computer is stored as a sequence of bits. For floating-point numbers, those bits are divided into three fields:

```
[Sign (1 bit)] [Exponent (E bits)] [Mantissa (M bits)]
```

- **Sign**: 0 = positive, 1 = negative
- **Exponent**: Determines the magnitude (how big or small the number is)
- **Mantissa (fraction)**: Determines the precision (how many significant digits)

The value represented is:

$$
\text{value} = (-1)^{\text{sign}} \times 2^{\text{exponent} - \text{bias}} \times (1 + \text{mantissa})
$$

### The precision hierarchy

| Format | Total bits | Exponent bits | Mantissa bits | Dynamic range | Precision |
|---|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | ±3.4 × 10³⁸ | ~7 decimal digits |
| FP16 | 16 | 5 | 10 | ±65,504 | ~3.3 decimal digits |
| BF16 | 16 | 8 | 7 | ±3.4 × 10³⁸ | ~2.4 decimal digits |
| **FP8 (E4M3)** | **8** | **4** | **3** | **±448** | **~1.7 decimal digits** |
| **FP8 (E5M2)** | **8** | **5** | **2** | **±57,344** | **~1.2 decimal digits** |
| INT8 | 8 | — | — | -128 to +127 | Integer only |
| INT4 | 4 | — | — | -8 to +7 | Integer only |

### The two FP8 formats

FP8 comes in two variants that trade off range vs. precision:

**E4M3** (4 exponent bits, 3 mantissa bits):
- Range: ±448
- Precision: Can distinguish values like 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875
- **Best for weights and activations** — needs precision to distinguish similar values
- 8 distinct values per power-of-2 interval

**E5M2** (5 exponent bits, 2 mantissa bits):
- Range: ±57,344
- Precision: Can distinguish values like 1.0, 1.25, 1.5, 1.75 (only 4 values per interval)
- **Best for gradients** (during training) — needs range more than precision because gradients can be very large or very small
- Rarely used for inference

**For LLM inference, E4M3 is the default FP8 format.** When people say "FP8 inference," they mean E4M3.

---

## 2. Why FP8 Matters for Inference: The Memory Bandwidth Argument

### The bottleneck (from Pope's framework)

Recall from the roofline model:

$$
t_{\text{mem}} = \frac{N_{\text{total}} \times \text{bytes per param}}{\text{mem\_bw}}
$$

During decode, the GPU's primary bottleneck is reading model weights from HBM. The time to read all weights is directly proportional to **bytes per parameter**.

| Precision | Bytes per param | Weight size (70B model) | HBM read time (H100, 3.35 TB/s) |
|---|---|---|---|
| FP32 | 4 bytes | 280 GB | 83.6 ms |
| FP16/BF16 | 2 bytes | 140 GB | 41.8 ms |
| **FP8** | **1 byte** | **70 GB** | **20.9 ms** |
| INT4 | 0.5 bytes | 35 GB | 10.4 ms |

**FP8 halves the memory bandwidth cost compared to FP16.** This directly translates to:
- 2× faster decode steps (in the memory-bound regime)
- 2× higher throughput (tokens per second)
- 2× more capacity — a 70B model in FP8 fits in 70 GB vs. 140 GB in FP16, leaving more HBM for KV cache (= more concurrent users)

### The compute argument

FP8 also doubles compute throughput on supported hardware:

| Hardware | FP16 TFLOP/s | FP8 TFLOP/s | Speedup |
|---|---|---|---|
| H100 (SXM) | 990 | 1,979 | 2.0× |
| H200 | 990 | 1,979 | 2.0× |
| B200 | 2,250 | 4,500 | 2.0× |
| MI300X | 1,300 | 2,600 | 2.0× |

The Tensor Cores (NVIDIA) or Matrix Cores (AMD) can process FP8 operands at twice the rate of FP16 operands, because each core packs twice as many FP8 multiplications into the same silicon area and clock cycle.

**Combined effect**: FP8 gives you 2× memory bandwidth savings AND 2× compute throughput. Both terms in the roofline model improve:
- t_memory halved (fewer bytes to load)
- t_compute halved (more FLOP/s available)

The critical batch size B* remains roughly the same (both numerator and denominator of FLOPs/BW double), but total throughput at any batch size approximately doubles.

---

## 3. How FP8 Kernels Work: The Technical Details

### 3.1 Quantization: Converting FP16 → FP8

Weights are stored in FP8 and must be converted from the original FP16/BF16 training precision. The process:

**Per-tensor scaling:**
1. Find the maximum absolute value in the tensor: $\text{amax} = \max(|W|)$
2. Compute a scale factor that maps amax to the FP8 representable range: $\text{scale} = \frac{\text{FP8\_MAX}}{\text{amax}} = \frac{448}{\text{amax}}$
3. Multiply all values by the scale: $W_{\text{scaled}} = W \times \text{scale}$
4. Round each value to the nearest FP8 representable value: $W_{\text{fp8}} = \text{round\_to\_fp8}(W_{\text{scaled}})$
5. Store $W_{\text{fp8}}$ (1 byte each) and the scale factor (1 FP32 number per tensor)

**Per-channel scaling (more precise):**
- Instead of one scale factor for the entire tensor, compute a separate scale factor per output channel (row of the weight matrix)
- This accommodates the fact that different channels may have very different value ranges
- Costs slightly more storage (one FP32 scale per channel) but significantly improves accuracy

**Static vs. dynamic quantization:**
- **Static**: Scale factors computed once (during calibration with a representative dataset) and fixed at inference time. Fastest, but may miss outlier activations.
- **Dynamic**: Scale factors recomputed every forward pass based on the actual activation values. Slightly slower (requires a max-reduce operation), but handles input-dependent outlier activations correctly.

### 3.2 The FP8 GEMM (Matrix Multiply) Kernel

The core computation in every transformer layer is a **GEMM** (General Matrix Multiply):

$$
Y = X \cdot W^T
$$

Where X is the activation matrix (batch × hidden_dim) and W is the weight matrix (output_dim × hidden_dim).

An FP8 GEMM works as follows:

```
1. Inputs: X_fp8 (FP8 activations), W_fp8 (FP8 weights), scale_X, scale_W
2. Hardware computes: Y_fp32_accumulator = X_fp8 ⊗ W_fp8  (in FP8, accumulated in FP32)
3. Rescale: Y_output = Y_fp32_accumulator / (scale_X × scale_W)
4. Optionally cast Y_output to FP8 for the next layer, computing a new scale factor
```

The key detail: **the multiplication is done in FP8, but the accumulation (summing the products) is done in FP32.** This is critical because:

- Each individual multiply produces a low-precision result (FP8 × FP8 → FP16-ish)
- But summing thousands of these low-precision products can accumulate errors
- FP32 accumulation prevents this error accumulation
- The final result is nearly as accurate as FP16 × FP16 → FP32 accumulation

This is analogous to how INT8 quantization works — the multiplies are cheap (INT8 × INT8), but the sums use a wider accumulator (INT32).

### 3.3 The Dequantization Overhead

After the GEMM, the FP32 result must be:
1. Rescaled (undo the input scale factors)
2. Added to biases (if any)
3. Optionally re-quantized to FP8 for the next layer

This rescaling is a lightweight element-wise operation — essentially one multiply per output element. The cost is negligible compared to the GEMM itself (~0.1% overhead).

**Fused kernels** perform the rescaling inside the GEMM kernel, eliminating the separate rescaling pass entirely. This is what production FP8 kernels (cuBLAS, CUTLASS, Triton) do.

### 3.4 Where FP8 Quantization Is Applied in the Transformer

Not every operation benefits equally from FP8:

| Component | FP8 benefit | Notes |
|---|---|---|
| **Linear layers (Q, K, V, O projections)** | High — these are GEMMs | Primary target, largest compute cost |
| **FFN layers (up, gate, down projections)** | High — also GEMMs | Equally important target |
| **Attention scores (Q × K^T)** | Moderate | Can use FP8 but precision matters for softmax input |
| **Attention × V multiplication** | Moderate | FP8 possible, some quality sensitivity |
| **Layer normalization** | Low — small compute | Usually kept in FP16/FP32 for stability |
| **Softmax** | None — must be high precision | Always FP32 (exp() needs precision) |
| **Residual connections** | None — element-wise adds | Kept in FP16/BF16 to maintain residual stream quality |

In practice, FP8 is applied to **all linear layers** (which account for ~95% of the model's FLOPs) while keeping normalization, softmax, and residual connections in higher precision. This is called **mixed-precision FP8 inference**.

---

## 4. Quality Impact: How Much Accuracy Do You Lose?

### Empirical results

For well-calibrated FP8 quantization (per-channel static scaling):

| Benchmark | FP16 baseline | FP8 (E4M3) | Degradation |
|---|---|---|---|
| MMLU (knowledge) | 81.2% | 80.9% | -0.3% |
| HumanEval (code) | 67.1% | 66.5% | -0.6% |
| GSM8K (math) | 84.7% | 84.1% | -0.6% |
| Perplexity (lower = better) | 5.12 | 5.15 | +0.6% |

**FP8 typically degrades quality by less than 1% across major benchmarks.** This is dramatically better than INT4 quantization (which often shows 2–5% degradation) because FP8 preserves the floating-point representation — it has a dynamic range (via the exponent) that adapts to the scale of the values rather than being fixed like integer.

### Why FP8 is surprisingly good

The key insight is that **neural network weights are approximately normally distributed** — most values cluster near zero, with few outliers. FP8's logarithmic spacing (from the exponent field) naturally allocates more precision near zero (where most values live) and less precision for large values (which are rare).

This matches the distribution of weights far better than linear INT8 spacing, which wastes half its representable values on the large-magnitude region where few weights exist.

### When FP8 degrades

- **Outlier-heavy models**: Some models (especially older or poorly trained ones) have large weight outliers that clip in FP8's limited range. Per-channel scaling mitigates this.
- **Long-generation accuracy drift**: Small per-token errors can compound over thousands of tokens. This is more noticeable in reasoning chains where each step depends on previous ones.
- **Very small models**: Models with fewer parameters have less redundancy to absorb quantization errors. FP8 works best on 7B+ parameter models.

---

## 5. FP8 in Practice: What the Software Stack Looks Like

### 5.1 Model Conversion Pipeline

```
Pre-trained FP16 model
    ↓ Calibration (run ~512 representative samples)
    ↓ Compute per-channel scale factors for each linear layer
    ↓ Quantize weights to FP8 (E4M3)
    ↓ Store quantized model + scale factors
    = FP8 inference-ready model
```

This takes ~10 minutes for a 70B model on a single GPU. It's a one-time cost.

### 5.2 Framework Support

| Framework | FP8 support | Notes |
|---|---|---|
| **TensorRT-LLM** | Full production support | NVIDIA's primary FP8 inference path |
| **vLLM** | Supported via FP8 weight loading | Uses CUTLASS/cuBLAS FP8 kernels |
| **SGLang** | Supported | Similar to vLLM |
| **llama.cpp** | Limited (GGUF FP8 quantization) | CPU/GPU hybrid, less optimized |
| **PyTorch (native)** | torch.float8 dtype, torch.compile FP8 | Experimental but improving rapidly |

### 5.3 Hardware Requirements

FP8 requires **hardware Tensor Core support** for the 2× compute speedup:
- **NVIDIA**: H100, H200, B200, GB200 (Hopper and Blackwell architectures). Ada Lovelace (RTX 4090) also supports FP8 but with lower throughput.
- **AMD**: MI300X (CDNA 3 architecture)
- **Not supported**: A100, V100, T4 (these can store FP8 but compute it at FP16 speed, losing the compute benefit while keeping the bandwidth benefit)

On unsupported hardware, you can still store weights in FP8 (saving memory and bandwidth) but dequantize to FP16 for computation. You get the bandwidth benefit but not the compute benefit.

---

## 6. FP8 vs. Other Quantization Methods

| Method | Bits per weight | Compute speed | Quality loss | Complexity |
|---|---|---|---|---|
| FP16 (baseline) | 16 | 1× | None | None |
| BF16 | 16 | 1× | Negligible | None |
| **FP8 (E4M3)** | **8** | **2×** | **<1%** | **Low** |
| INT8 (W8A8) | 8 | 2× | <1% | Medium (calibration) |
| INT4 (W4A16) | 4 (weights only) | ~1.5× | 2–5% | High (grouping, scales) |
| GPTQ (INT4) | 4 | ~1.5× | 1–3% | High (calibration + optimization) |
| AWQ (INT4) | 4 | ~1.5× | 1–3% | High |

**FP8's sweet spot**: It offers the best quality-to-speedup ratio. INT4 is faster per byte but loses more quality. FP16 has perfect quality but is 2× slower. FP8 hits the inflection point where the quality loss is imperceptible but the speedup is meaningful.

**Why not just use INT4 for everything?**
- INT4 uses **integer** arithmetic — no exponent, fixed scaling. This means it handles weight distributions with different scales poorly (unless complex grouping/per-channel scaling is used).
- FP8 naturally handles multi-scale distributions because the exponent field gives it **logarithmic** resolution.
- INT4 requires more complex calibration (group sizes, asymmetric quantization) and is more sensitive to outliers.
- For activations, INT4 is particularly bad because activation distributions shift dramatically per input. FP8 with dynamic scaling handles this gracefully.

---

## 7. The Future of FP8 Kernels

### 7.1 FP8 Becomes the Default Precision for Inference

The trajectory is clear:

| Year | Default inference precision | Adoption |
|---|---|---|
| 2022 | FP16/BF16 | Universal |
| 2023 | FP16 with optional INT8/INT4 | Early adopters |
| 2024 | FP8 for H100 deployments, FP16 elsewhere | Growing |
| 2025 | FP8 as the standard for all Hopper/Blackwell deployments | Mainstream |
| 2026+ | FP8 default, FP4/INT4 for aggressive optimization | Universal |

As Hopper and Blackwell become the majority of the inference fleet, FP8 will be the default precision — not an optimization you opt into, but the baseline you start from.

### 7.2 FP4 on the Horizon

NVIDIA's Blackwell architecture (B200) introduces **FP4** Tensor Core support:
- 4 bits per value (E2M1 format)
- 4× compute throughput vs. FP16 (vs. 2× for FP8)
- 4× memory bandwidth savings
- Quality loss: ~2–4% on major benchmarks (worse than FP8, but improving with better calibration and per-block scaling)

FP4 will likely be used for weight-only quantization (W4A8: FP4 weights, FP8 activations), similar to how INT4 weight-only quantization is used today but with better hardware support.

### 7.3 Training in FP8 → Inference in FP8 (No Conversion Needed)

DeepSeek-V3 was trained in FP8 from the start. This eliminates the quantization step entirely:
- The model's weights are already in FP8 format
- No calibration needed, no conversion accuracy loss
- The model was trained to be robust to FP8 precision from day one

As more models adopt FP8 training, the quantization conversion pipeline becomes obsolete. Models ship in FP8 natively, and the serving stack consumes them directly.

### 7.4 Mixed-Precision Per Layer (Adaptive Quantization)

Not all layers have the same sensitivity to precision reduction:
- Attention projection layers: Relatively robust to FP8
- FFN layers: Very robust to FP8 (even FP4 may work)
- First and last transformer layers: More sensitive (embeddings and unembeddings have higher information density)

Future systems will use **per-layer precision selection**:
- Sensitive layers: FP8 or even FP16
- Robust layers: FP4
- Average effective precision: ~6 bits per parameter
- Quality: Nearly identical to uniform FP8

This requires kernel dispatch systems that handle mixed-precision GEMMs efficiently — the scheduler must select the right kernel variant for each layer.

---

## 8. The Investor Lens

FP8 kernels sit at the intersection of the **Compilation/Optimization Layer** and the **Hardware Layer** in the inference stack. They are the primary mechanism for converting hardware generational improvements into actual inference cost reduction.

### Core Thesis

> **FP8 is the single most impactful, lowest-risk inference optimization available today. It delivers a ~2× speed improvement with <1% quality loss, requires minimal engineering effort (one-time calibration), and is supported by all modern inference hardware. Any inference deployment not using FP8 on Hopper/Blackwell hardware is leaving half its capital efficiency on the table. FP8 is not an optimization — it's the new baseline.**

### Primary Investment Implications

#### 1. FP8 Is the New Default (Not an Optimization)

The mental model shift: FP8 is not "quantization you apply to make things faster." It's "the native precision of modern inference hardware." Running in FP16 on an H100 is like running in FP32 on the first Tensor Core GPUs — technically possible, but you're using the hardware wrong.

Every major inference deployment will move to FP8 as the baseline:
- vLLM, TRT-LLM, SGLang all support it
- DeepSeek-V3 was trained in FP8 natively
- NVIDIA's own benchmarks assume FP8 as the default

**Investor takeaway**: When evaluating inference provider benchmarks, check whether they're reporting FP8 or FP16 numbers. FP16 "performance" is misleading on Hopper/Blackwell — it's leaving half the hardware's capability unused. Any provider still defaulting to FP16 on H100 has not fully optimized their stack.

#### 2. FP8 Doubles Effective GPU Supply

From a market perspective, FP8 adoption is equivalent to doubling the supply of GPUs:
- A fleet of 1,000 H100s running FP8 produces approximately the same token throughput as 2,000 H100s running FP16
- This means existing GPU deployments become 2× more productive without buying new hardware
- New deployments need half as many GPUs for the same capacity

**But the Jevons paradox applies**: The 2× cost reduction enables new use cases that were previously too expensive. Total demand grows by more than 2×, so GPU demand still increases. This has played out with every previous efficiency gain (FlashAttention, continuous batching, INT8 quantization).

**Investor takeaway**: FP8 adoption is deflationary for per-token costs but inflationary for total token demand. Long-term, this is bullish for NVIDIA and HBM suppliers because total compute demand grows faster than per-token cost shrinks. Short-term, FP8 adoption may cause some DeepSeek-V3 style "efficiency shock" moments where the market temporarily fears GPU demand will drop.

#### 3. Hardware-Software Co-design Creates NVIDIA Lock-In

FP8 kernels are tightly coupled to hardware:
- NVIDIA's FP8 Tensor Cores use specific E4M3/E5M2 formats
- Optimal FP8 GEMM kernels (CUTLASS, cuBLAS) are CUDA-only
- AMD's ROCm supports FP8 but with less mature kernel libraries
- Custom ASICs (Groq, Cerebras) have their own FP8 implementations (if any)

The FP8 kernel ecosystem creates a **hardware lock-in cycle**:
- Developers optimize for NVIDIA's FP8 implementation
- Kernels and models are tested primarily on NVIDIA hardware
- Switching to AMD or custom ASICs requires re-validation of accuracy and performance
- This increases switching costs and reinforces NVIDIA's ecosystem moat

**Investor takeaway**: FP8 adoption deepens NVIDIA's ecosystem moat. As the inference stack converges on FP8 as the default, every kernel, every model checkpoint, and every benchmark assumes NVIDIA's FP8 implementation. Competitors must match not just hardware performance but the entire software ecosystem. This is bullish for NVIDIA's inference-era pricing power.

#### 4. FP4 Is the Next Inflection Point (Blackwell)

B200 introduces FP4 Tensor Cores, potentially offering another 2× improvement over FP8:
- FP8 → FP4: 2× memory bandwidth savings, 2× compute throughput
- Total improvement from FP16 baseline: ~4× (FP16→FP8→FP4)
- Quality loss: Currently ~2–4%, expected to narrow with better scaling techniques

If FP4 reaches production quality (similar to how FP8 went from "experimental" to "default" in ~18 months), it represents another halving of inference costs. Blackwell hardware buyers get both the hardware improvement AND the FP4 precision improvement — a compounding advantage.

**Investor takeaway**: Track FP4 quality benchmarks on Blackwell. If FP4 achieves <1.5% quality loss with proper calibration (similar to FP8's trajectory), it becomes the Blackwell-era default, delivering another 2× cost reduction that triggers another Jevons paradox cycle. This would be the single largest efficiency event since FlashAttention.

#### 5. Models Trained in FP8 Change the Ecosystem

DeepSeek-V3's FP8 training demonstrates that quantization-aware training eliminates the accuracy gap entirely. If this becomes standard:
- Models ship in FP8 natively — no post-hoc quantization step
- The weights are inherently robust to FP8 precision (trained to be)
- Quality loss drops from <1% to ~0% (effectively free quantization)
- The "quantization tax" (accuracy cost of reducing precision) approaches zero

**Investor takeaway**: Model providers that train in FP8 have a structural cost advantage over those that train in FP16 and quantize post-hoc. DeepSeek's FP8 training is not just a training efficiency — it's a deployment efficiency that compounds at serving time. Watch for other labs adopting FP8 training as a signal that the full stack is converging on 8-bit as native precision.

### Risk Factors

**Risk 1 — Quality-sensitive applications refuse FP8.** For high-stakes domains (medical diagnosis, legal reasoning, financial analysis), even <1% quality degradation may be unacceptable. These applications may mandate FP16 or FP32 inference, creating a two-tier market where FP8 serves "good enough" workloads and full precision serves premium workloads.

**Risk 2 — FP8 may not be the final precision.** If FP4 or even FP2 methods achieve acceptable quality, FP8's advantage is temporary. However, the transition from FP8 → FP4 is much smaller than FP16 → FP8 (diminishing returns at lower precisions, and quality loss accelerates below 8 bits).

**Risk 3 — Non-IEEE FP8 formats on custom hardware.** If competing hardware vendors (Qualcomm, MediaTek, custom ASICs) adopt incompatible 8-bit formats, model portability suffers. This could fragment the ecosystem and increase deployment complexity.

### Summary Signal for Investors

> **FP8 is the "obviously correct" inference optimization — 2× performance at <1% quality loss, supported by all modern hardware, adopted by all major serving frameworks. It is no longer a competitive advantage; it is table-stakes. The investor signal is not "who uses FP8" (everyone will) but "who has moved beyond FP8" — tracking FP4 readiness on Blackwell and FP8-native training adoption as the next markers of infrastructure maturity. The durable insight is that each precision step (FP16→FP8→FP4) triggers a Jevons paradox cycle: lower cost → more demand → more GPU demand. Reduced precision is structurally bullish for inference volume.**
