# Memory Offload in LLM Inference: From First Principles

---

# 1. The Problem: Models Don't Fit

The fundamental constraint of LLM inference is that **GPU memory (HBM) is finite and expensive**, but the data required for inference keeps growing:

| Component | Size for a 70B model (FP16) | Size for a 70B model (INT4) |
|---|---|---|
| Model weights | ~140 GB | ~35 GB |
| KV cache (1 sequence, 4K context) | ~1.3 GB | ~0.3 GB |
| KV cache (100 concurrent sequences, 4K) | ~130 GB | ~33 GB |
| Activations (per batch) | ~1–5 GB | ~1–5 GB |

A single H100 GPU has 80 GB of HBM. A 70B model in FP16 doesn't even fit on one GPU. And even with INT4 quantization (35 GB for weights), 100 concurrent users' KV caches push total memory well beyond 80 GB.

There are only two solutions:
1. **Use more GPUs** (tensor/pipeline parallelism) — expensive
2. **Offload data to cheaper, larger memory** — that's memory offload

---

# 2. First Principles: The Memory-Speed Hierarchy

From the HBM vs SRAM notes, recall the hierarchy:

```
Registers (~MB)     → fastest, smallest
L1/Shared (~50 MB)  → very fast
L2 (~100 MB)        → fast
GPU HBM (~80–192 GB) → high bandwidth, THE bottleneck
↓ offload boundary ↓
CPU DRAM (~512 GB–2 TB) → slow relative to HBM, very large
NVMe SSD (~1–16 TB) → very slow, very large, persistent
```

Memory offload exploits the lower tiers of this hierarchy to handle data that doesn't fit in HBM but is needed during inference.

---

# 3. What Gets Offloaded (and Why)

Not all data is equally latency-sensitive. The offload strategy depends on access patterns:

### 3.1 Weight Offloading

**What**: Store model weights in CPU DRAM (or even disk) and load them into GPU HBM layer-by-layer as needed.

**Why it works**: During decode, the model processes one layer at a time. You only need **one layer's weights in GPU memory at a time**, not the full model. While computing on layer N, you can prefetch layer N+1's weights from CPU DRAM.

**Trade-off**: Each layer's weights must traverse the PCIe or NVLink bus:
- PCIe 5.0: ~64 GB/s
- NVLink (CPU–GPU): ~900 GB/s (only on some systems like Grace Hopper)

For a 70B model with ~4.4 GB per layer (32 layers, FP16):
- Over PCIe 5.0: ~69 ms per layer → ~2.2 seconds for full model pass → ~0.5 tokens/sec
- Over NVLink: ~5 ms per layer → ~160 ms for full model pass → ~6 tokens/sec

**Verdict**: Weight offloading over PCIe is too slow for interactive inference but acceptable for batch or offline processing. NVLink (Grace Hopper) makes it viable for moderate-throughput interactive use.

### 3.2 KV Cache Offloading

**What**: Move KV cache entries for inactive or low-priority sequences from GPU HBM to CPU DRAM. Bring them back when the sequence becomes active again.

**Why it works**: In a continuous batching system, not all sequences are equally active:
- Some are in decode (generating tokens) — need KV cache in HBM
- Some are paused (waiting for user input, preempted by scheduler) — KV cache can be offloaded
- Some are waiting to resume (long conversation, session persistence) — KV cache can go to disk

**How it integrates with PagedAttention**: vLLM's block manager already treats KV cache as virtual memory pages. Offloading extends this naturally:
- Active pages: GPU HBM
- Swapped pages: CPU DRAM
- Evicted pages: Disk (must re-prefill if needed later)

```
Scheduler decides to preempt sequence B:
  1. Copy B's KV cache pages from GPU → CPU DRAM (async DMA)
  2. Free those GPU blocks for new sequence C
  3. Later, when B resumes: copy pages back from CPU → GPU
```

**Trade-off**: Swap latency (PCIe transfer) adds to TTFT when a sequence resumes. For a 4K-context sequence with ~1.3 GB of KV cache:
- Swap out: ~20 ms over PCIe 5.0
- Swap in: ~20 ms

This is acceptable if preemption is rare, but becomes a problem under heavy load with frequent preemptions.

### 3.3 Activation Offloading

**What**: During prefill (forward pass over the prompt), intermediate activations can be offloaded to CPU to free GPU memory for more KV cache or a larger batch.

**Why it's usually not needed**: Activations are ephemeral — they exist only during the forward pass and are discarded after each layer. Unlike training (which needs activations for backprop), inference can discard them immediately. But in very large batch or very long prompt scenarios, even transient activation memory can be significant.

---

# 4. Offload Scheduling: How to Hide the Latency

The key engineering challenge is **hiding the transfer latency** behind useful computation. This connects directly to prefetch pipelines:

```
Layer N running on GPU:
  |
  |--- Simultaneously: PCIe DMA copies layer N+1 weights from CPU → GPU
  |--- Simultaneously: PCIe DMA copies preempted sequence's KV cache from GPU → CPU
  |
Layer N+1 weights arrive before layer N finishes
  → zero stall
```

This requires:
1. **Async DMA**: Non-blocking memory copies using CUDA streams
2. **Pinned (page-locked) CPU memory**: Prevents the OS from paging CPU-side buffers, enabling reliable DMA speed
3. **Multi-stream scheduling**: Separate CUDA streams for compute, host-to-device transfers, and device-to-host transfers, all running concurrently

```python
# Pseudocode for weight-offloaded inference
compute_stream = cuda.Stream()
transfer_stream = cuda.Stream()

# Pre-load first layer
transfer_stream.memcpy_h2d(gpu_buffer, cpu_weights[0])
transfer_stream.synchronize()

for layer_idx in range(num_layers):
    # Start loading NEXT layer (async, on transfer stream)
    if layer_idx + 1 < num_layers:
        transfer_stream.memcpy_h2d(gpu_buffer_next, cpu_weights[layer_idx + 1])
    
    # Compute CURRENT layer (on compute stream)
    compute_stream.run_layer(gpu_buffer, activations)
    
    # Wait for both streams before swapping
    cuda.synchronize()
    swap(gpu_buffer, gpu_buffer_next)
```

---

# 5. Real-World Systems That Use Memory Offload

| System | Offload Type | Mechanism |
|---|---|---|
| **vLLM** | KV cache swap to CPU | PagedAttention block manager with swap-in/swap-out |
| **DeepSpeed-Inference** | Weights + KV to CPU/NVMe | ZeRO-Inference with pipelined offloading |
| **llama.cpp** | Partial weight offload (GPU+CPU split) | Split layers between GPU VRAM and CPU RAM |
| **FlexGen** | Weights + KV + activations to CPU/SSD | Aggressive offloading for throughput-optimized batch inference on limited GPUs |
| **HuggingFace Accelerate** | Weight offload via device_map | Layer-wise placement across GPU/CPU/disk |

---

# 6. When Offloading Makes Sense vs. When It Doesn't

| Scenario | Offload? | Why |
|---|---|---|
| Model too large for one GPU, can't afford multi-GPU | ✅ Yes | Only option to run the model at all |
| Batch inference, no latency SLA | ✅ Yes | Throughput-per-dollar is the metric; offload maximizes utilization of cheap CPU DRAM |
| Interactive chatbot, latency-sensitive | ⚠️ Careful | KV cache swap adds TTFT spikes when sequences resume; weight offload caps ITL |
| Many concurrent long-context sessions | ✅ Yes (KV only) | Offload idle session KV to CPU, keep active sessions in HBM |
| Real-time voice (< 20ms ITL required) | ❌ No | Any PCIe transfer breaks the latency SLA |

---

# 7. The Future of Memory Offload

### 1. CXL-Attached Memory Pools
CXL (Compute Express Link) provides ~200–400 ns latency to pooled DRAM — 5–10× worse than HBM but 10–50× better than NVMe. This creates a new tier perfectly suited for KV cache offloading:

```
GPU HBM (hot) → CXL DRAM (warm) → NVMe (cold)
```

CXL memory can be shared across GPUs, enabling dynamic rebalancing of KV cache capacity based on per-GPU load.

### 2. Heterogeneous Memory Systems (Grace Hopper)
NVIDIA's Grace Hopper pairs a GPU with an ARM CPU via NVLink-C2C at 900 GB/s — close to HBM bandwidth for CPU↔GPU transfers. This makes weight offloading viable for interactive workloads and dramatically increases effective memory capacity (up to 624 GB unified memory).

### 3. Intelligent Offload Policies
Current offload policies are simple (FIFO, priority-based swap). Future systems will use ML-driven prediction:
- Which sequences will be needed next? (Pre-fetch their KV cache)
- Which sequences are unlikely to resume? (Evict to disk)
- How to dynamically size the GPU/CPU/disk allocation based on traffic patterns?

---

# 8. The Investor Lens (Aligned with the Inference Framework)

Memory offload sits at the intersection of the **Serving / Runtime Layer** and the **Hardware Layer**. It is the mechanism that decouples GPU memory capacity from serving capacity, enabling more concurrent users per GPU.

### Value Drivers

- **GPU ROI multiplier**: Without offloading, a GPU can serve only as many concurrent sequences as fit in HBM. With KV cache offloading, the same GPU serves 3–10× more concurrent sessions (limited by throughput, not memory). This directly improves GPU utilization and ROI.
- **CXL as a catalyst**: CXL-attached DRAM turns memory offloading from a "tolerable compromise" into a "near-transparent extension." Investors should watch CXL ecosystem companies (Astera Labs, Montage Technology) as the enabling layer for next-generation inference memory architectures.
- **Grace Hopper as a competitive signal**: NVIDIA's Grace Hopper design (unified NVLink memory between CPU and GPU) is a hardware-level acknowledgment that offloading is central to inference economics. It validates the thesis that future inference nodes will be defined by total memory capacity (HBM + DRAM + CXL), not just HBM alone.
- **The self-hosting enabler**: Memory offload is what makes running a 70B model on consumer hardware (a single RTX 4090 with 24 GB VRAM + 128 GB system RAM) possible at all. This directly enables the "edge AI" migration that threatens cloud inference revenue for commodity tasks.

### Risk Factor

If HBM capacity per GPU grows fast enough (80 GB → 192 GB → 288 GB → 512 GB+), the need for offloading to CPU DRAM diminishes for standard workloads. But KV cache growth (driven by longer contexts and more concurrent users) is growing faster than HBM capacity, so offloading will remain relevant for the foreseeable future.

### Summary Signal

> Memory offload is the democratization lever — it lets smaller hardware run bigger models. For investors, it accelerates the "build vs. buy" shift toward self-hosting and threatens cloud inference margins for commodity workloads. But it also sustains the inference hardware market by making every GPU more productive per dollar invested.
